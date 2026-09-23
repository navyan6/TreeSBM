#!/usr/bin/env python3
"""
Comprehensive evaluation of a single generated tree.

  A. Sequence quality: root divergence, leaf diversity, ESM PLL, post-prune survival
  B. Phylogenetic coherence: sibling/parent vs random-pair sequence identity
  C. Tree structure: bifurcating check, depth, branch lengths, Sackin index, cherry count
  D. GT comparison: depth/branch distributions, best-match seq identity

Usage:
    python scripts/eval_single_tree.py \
        --checkpoint checkpoints/best.pt \
        --data data/train \
        --group 1 \
        --n-steps 50 \
        --max-leaves 200 \
        --branch-rate-scale 6.0
"""

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForMaskedLM

from src.dataset import TreeDataset
from src.tree_state import TreeState
from src.treeencoder.node_encoder import NodeEncoder
from src.treeencoder.plm_embeddings import ESM2Embedder
from src.treeencoder.structural_features import compute_structural_features
from src.treeencoder.laplacian import compute_laplacian_pe
from src.treeencoder.edges import build_edges
from src.networks import TreeEncoder, RateHeads
from src.bridge.losses import _build_seq_indices
from src.bridge.fitness_tilt import (
    TILT_FULL_ESM,
    TILT_SITE_LOCAL,
    tilt_log_R0_by_fitness,
)
from src.bridge.mutation_sample import (
    mutate_sequence_independent,
    mutate_sequence_site_softmax,
)

AA_VOCAB  = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}


#utiliities

def seq_identity(a: str, b: str) -> float:
    L = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:L], b[:L])) / L if L else 0.0


def get_lm_logits(tokenizer, esm_model, aa_token_ids, sequences, max_seq_len, device):
    log_rates = torch.zeros(len(sequences), max_seq_len, 20, dtype=torch.float32, device=device)
    with torch.no_grad():
        tokens = tokenizer(sequences, return_tensors="pt", padding=True,
                           truncation=False).to(device)
        logits = esm_model(**tokens).logits
    seq_lens = tokens["attention_mask"].sum(dim=1)
    for i in range(len(sequences)):
        actual_L = int(seq_lens[i].item()) - 2
        aa_logits = logits[i, 1:actual_L + 1, :][:, aa_token_ids]
        log_probs = F.log_softmax(aa_logits, dim=-1)
        clip = min(actual_L, max_seq_len)
        log_rates[i, :clip, :] = log_probs[:clip]
    return log_rates


def _seq_keyed_cached_batch(
    sequences: list[str],
    cache: dict[str, torch.Tensor],
    compute_fn,
    device,
    stats: dict | None = None,
) -> torch.Tensor:
    """Exact recompute-on-miss cache keyed by AA string.

    ``compute_fn(miss_seqs)`` must return a tensor stacked in the same order as
    unique ``miss_seqs``. Hits are reused unchanged (no freeze-R0 approximation).
    Duplicate sequences within one batch are deduped before the forward.
    """
    if not sequences:
        raise ValueError("sequences must be non-empty")
    miss_seqs = list(dict.fromkeys(s for s in sequences if s not in cache))
    n_hit = sum(1 for s in sequences if s in cache)
    if miss_seqs:
        fresh = compute_fn(miss_seqs)
        if fresh.shape[0] != len(miss_seqs):
            raise RuntimeError(
                f"compute_fn returned {fresh.shape[0]} rows for {len(miss_seqs)} misses"
            )
        for j, s in enumerate(miss_seqs):
            cache[s] = fresh[j].detach().cpu()
    if stats is not None:
        stats["hits"] = stats.get("hits", 0) + n_hit
        stats["misses"] = stats.get("misses", 0) + len(miss_seqs)
        stats["unique"] = len(cache)
    return torch.stack([cache[s].to(device) for s in sequences], dim=0)


def embed_sequences_cached(
    embedder,
    sequences: list[str],
    cache: dict[str, torch.Tensor],
    device,
    stats: dict | None = None,
) -> torch.Tensor:
    """Seq-keyed wrapper around ``embedder.embed_sequences`` (exact on miss)."""
    return _seq_keyed_cached_batch(
        sequences,
        cache,
        lambda miss: embedder.embed_sequences(miss),
        device,
        stats=stats,
    )


def get_lm_logits_cached(
    tokenizer,
    esm_model,
    aa_token_ids,
    sequences: list[str],
    max_seq_len: int,
    device,
    cache: dict[str, torch.Tensor],
    stats: dict | None = None,
) -> torch.Tensor:
    """Seq-keyed wrapper around ``get_lm_logits`` (exact on miss)."""
    return _seq_keyed_cached_batch(
        sequences,
        cache,
        lambda miss: get_lm_logits(
            tokenizer, esm_model, aa_token_ids, miss, max_seq_len, device
        ),
        device,
        stats=stats,
    )


def esm_pll_seq(log_R0_i: torch.Tensor, seq: str, max_seq_len: int) -> float:
    vals = [log_R0_i[pos, AA_TO_IDX[aa]].item()
            for pos, aa in enumerate(seq[:max_seq_len]) if aa in AA_TO_IDX]
    return sum(vals) / len(vals) if vals else float("-inf")


def children_map(tree: TreeState) -> dict:
    cm = defaultdict(list)
    for p, c in tree.edges:
        cm[p].append(c)
    return dict(cm)


def get_leaves(tree: TreeState) -> list[str]:
    cm = children_map(tree)
    return [n for n in tree.node_ids if n not in cm]


def bfs_depths(tree: TreeState) -> dict[str, int]:
    depths = {tree.root_id: 0}
    queue = [tree.root_id]
    cm = children_map(tree)
    while queue:
        node = queue.pop(0)
        for child in cm.get(node, []):
            depths[child] = depths[node] + 1
            queue.append(child)
    return depths


def sackin_index(tree: TreeState, leaves: list[str]) -> int:
    depths = bfs_depths(tree)
    return sum(depths.get(l, 0) for l in leaves)


def cherry_count(tree: TreeState) -> int:
    cm = children_map(tree)
    leaf_set = set(get_leaves(tree))
    return sum(1 for ch in cm.values()
               if len(ch) == 2 and all(c in leaf_set for c in ch))


def section(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


#model loading

def load_models(checkpoint, device, max_seq_len):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint: epoch {ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', 0):.4f}")
    cfg = ckpt.get("config", {})
    node_enc = NodeEncoder(d_plm=320, d_struct=3, d_laplacian=8, d_node=128).to(device)
    tree_enc = TreeEncoder(d_model=128, n_layers=4, n_heads=8, dropout=0.1).to(device)
    r_heads  = RateHeads(
        d_model=128, max_seq_len=max_seq_len,
        use_pos_emb=cfg.get("use_pos_emb", False),
        use_site_entropy=cfg.get("use_site_entropy", False),
        deep_mut_head=cfg.get("deep_mut_head", False),
        use_mut_aa_emb=cfg.get("use_mut_aa_emb", False),
        d_aa=cfg.get("mut_aa_emb_dim", 16),
        use_pssm_gate=cfg.get("use_pssm_gate", False),
        pssm_gate_fixed_w=cfg.get("pssm_gate_fixed_w", None),
    ).to(device)
    node_enc.load_state_dict(ckpt["node_enc"])
    tree_enc.load_state_dict(ckpt["tree_enc"])
    r_heads.load_state_dict(ckpt["rate_heads"])
    for m in [node_enc, tree_enc, r_heads]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
    # Empirical column-entropy vector [L] if the model trained with
    # --entropy-source empirical; must be reused here or the mutation head sees
    # a different signal than it was trained on. None -> RateHeads falls back to
    # ESM self-entropy (matching an esm_self-trained model). Guard on the config
    # so an empirical model whose checkpoint lacks the vector fails loudly rather
    # than silently mismatching.
    col_entropy = ckpt.get("col_entropy", None)
    if col_entropy is not None:
        if isinstance(col_entropy, dict):
            col_entropy = {
                k: (v.to(device) if hasattr(v, "to") else v)
                for k, v in col_entropy.items()
            }
        else:
            col_entropy = col_entropy.to(device)
    elif cfg.get("entropy_source") == "empirical":
        raise RuntimeError(
            "checkpoint has entropy_source=empirical but no saved col_entropy; "
            "generation would fall back to ESM self-entropy and mismatch training."
        )
    log_pssm = ckpt.get("log_pssm", None)
    if log_pssm is not None:
        log_pssm = log_pssm.to(device)
    elif cfg.get("use_pssm_gate"):
        raise RuntimeError(
            "checkpoint has use_pssm_gate=True but no saved log_pssm; "
            "generation would mismatch training."
        )
    # Stash on module so generate_tree / callers keep the 4-tuple unpack stable.
    r_heads._train_log_pssm = log_pssm
    r_heads._fitness_beta = float(cfg.get("fitness_beta", 0.0))
    r_heads._fitness_score = cfg.get("fitness_score", "log_R0")
    r_heads._fitness_tilt_mode = cfg.get("fitness_tilt_mode", TILT_SITE_LOCAL)
    r_heads._fitness_esm_batch_size = int(cfg.get("fitness_esm_batch_size", 8))
    r_heads._fitness_esm_top_k = cfg.get("fitness_esm_top_k")
    return node_enc, tree_enc, r_heads, col_entropy


#generation

def generate_tree(root_seq, n_steps, max_seq_len, branch_rate_scale, max_leaves, mutation_rate_scale,
                  node_enc, tree_enc, rate_heads, embedder,
                  tokenizer, esm_model, aa_token_ids, device, col_entropy=None,
                  site_softmax_sample: bool = False,
                  site_temperature: float = 1.0,
                  fitness_beta: float | None = None,
                  fitness_score: str | None = None,
                  fitness_tilt_mode: str | None = None,
                  fitness_esm_batch_size: int | None = None,
                  fitness_esm_top_k: int | None = None,
                  cache_esm: bool = True,
                  esm_cache_stats: dict | None = None,
                  ablate_bridge: bool = False,
                  ablate_tree_context: bool = False,
                  branching_mode: str = "learned",
                  ref_lambda: float = 1.0,
                  ablate_branch_length_head: bool = False,
                  ablate_internal_node_seqs: bool = False,
                  ablate_site_entropy: bool = False,
                  r0_backend=None):
    """Generate a tree (Algorithm 4).

    ``ablate_site_entropy`` is accepted for call-site compatibility with
    ``eval_evescape_enrichment``; currently unused (entropy comes from ckpt).

    ``cache_esm`` (default True): keep in-memory seq→embedding and seq→log_R0
    caches across steps. Unchanged AA strings reuse cached tensors; misses run
    a full ESM forward (exact recompute, not freeze-R0).

    Generation ablations (no retrain):
      ablate_bridge: force log R_θ = log R0 (zero c_θ)
      ablate_tree_context: zero H_T before RateHeads
      branching_mode='poisson_ref': constant-λ Poisson branching (no seq-dep head)
      ablate_branch_length_head: use constant BL=dt instead of BL head
      ablate_internal_node_seqs: zero PLM embeddings for non-leaf nodes

    Pass ``r0_backend`` (from ``src.r0_backends.build_r0_backend``) to
    swap the frozen mutation prior (JTT / ESM-2-650M / ESM-C / …).
    """
    tree = TreeState.root_only(root_seq)
    node_birth_step = {tree.root_id: 0}
    dt = 1.0 / n_steps
    log_pssm = getattr(rate_heads, "_train_log_pssm", None)
    if fitness_beta is None:
        fitness_beta = getattr(rate_heads, "_fitness_beta", 0.0)
    if fitness_score is None:
        fitness_score = getattr(rate_heads, "_fitness_score", "log_R0")
    if fitness_tilt_mode is None:
        fitness_tilt_mode = getattr(rate_heads, "_fitness_tilt_mode", TILT_SITE_LOCAL)
    if fitness_esm_batch_size is None:
        fitness_esm_batch_size = getattr(rate_heads, "_fitness_esm_batch_size", 8)
    if fitness_esm_top_k is None:
        fitness_esm_top_k = getattr(rate_heads, "_fitness_esm_top_k", None)
    fitness_cache: dict = {}
    emb_cache: dict[str, torch.Tensor] = {}
    r0_cache: dict[str, torch.Tensor] = {}
    emb_stats = None
    r0_stats = None
    if esm_cache_stats is not None:
        emb_stats = esm_cache_stats.setdefault("emb", {"hits": 0, "misses": 0, "unique": 0})
        r0_stats = esm_cache_stats.setdefault("r0", {"hits": 0, "misses": 0, "unique": 0})
    fitness_scorer = None
    if fitness_tilt_mode == TILT_FULL_ESM and float(fitness_beta) != 0.0:
        from src.r0_backends import AA_TO_IDX as _AA_TO_IDX

        def fitness_scorer(sequences):
            L = max((len(s) for s in sequences), default=max_seq_len)
            seqs = list(sequences)
            if r0_backend is not None:
                log_R0 = r0_backend.log_mutation_rates(seqs, L, device=device)
                if torch.is_tensor(log_R0):
                    log_R0 = log_R0.detach().cpu()
            elif cache_esm:
                log_R0 = get_lm_logits_cached(
                    tokenizer, esm_model, aa_token_ids, seqs, L, device,
                    r0_cache, stats=r0_stats,
                ).cpu()
            else:
                log_R0 = get_lm_logits(
                    tokenizer, esm_model, aa_token_ids, seqs, L, device
                ).cpu()
            out = []
            for i, seq in enumerate(seqs):
                vals = []
                for pos, aa in enumerate(seq[:L]):
                    j = _AA_TO_IDX.get(aa)
                    if j is None:
                        continue
                    vals.append(float(log_R0[i, pos, j].item()))
                out.append(sum(vals) / len(vals) if vals else float("-inf"))
            return torch.tensor(out, dtype=torch.float32)

    for step in range(n_steps):
        t = step / n_steps
        if not tree.active_leaves:
            break

        node_ids_t  = tree.node_ids
        node_to_idx = {nid: i for i, nid in enumerate(node_ids_t)}
        active_leaves = list(tree.active_leaves)
        active_idx    = [node_to_idx[v] for v in active_leaves]
        node_times_dict = {nid: node_birth_step.get(nid, 0) / n_steps
                           for nid in node_ids_t}

        struct_t = compute_structural_features(tree, node_to_idx).to(device)
        lap_t    = compute_laplacian_pe(tree, node_to_idx, 8, device=device)
        edge_index_t, _, edge_attr_t = build_edges(tree, node_to_idx)
        edge_index_t  = edge_index_t.to(device)
        branch_lens_t = edge_attr_t.squeeze(-1).to(device)
        node_seqs = [tree.node_seqs[nid] for nid in node_ids_t]
        if ablate_internal_node_seqs:
            # Without internal-node sequences — only leaf PLM features
            # enter NodeEncoder/TreeEncoder; internals get zero embeddings.
            # Stack is a fresh tensor (cache stores CPU copies), so in-place
            # zeroing cannot poison emb_cache.
            leaf_set = set(active_leaves)
            plm_t = torch.zeros(len(node_ids_t), 320, device=device)
            leaf_pos = [i for i, nid in enumerate(node_ids_t) if nid in leaf_set]
            if leaf_pos:
                leaf_seqs_emb = [node_seqs[i] for i in leaf_pos]
                if cache_esm:
                    leaf_plm = embed_sequences_cached(
                        embedder, leaf_seqs_emb, emb_cache, device, stats=emb_stats
                    )
                else:
                    leaf_plm = embedder.embed_sequences(leaf_seqs_emb).to(device)
                plm_t[leaf_pos] = leaf_plm
        elif cache_esm:
            plm_t = embed_sequences_cached(
                embedder, node_seqs, emb_cache, device, stats=emb_stats
            )
        else:
            plm_t = embedder.embed_sequences(node_seqs).to(device)

        active_seqs = [tree.node_seqs[v] for v in active_leaves]
        if r0_backend is not None:
            if cache_esm:
                log_R0_mut = _seq_keyed_cached_batch(
                    active_seqs,
                    r0_cache,
                    lambda miss: r0_backend.log_mutation_rates(
                        miss, max_seq_len, device=device
                    ),
                    device,
                    stats=r0_stats,
                )
            else:
                log_R0_mut = r0_backend.log_mutation_rates(
                    active_seqs, max_seq_len, device=device
                )
            if not torch.is_tensor(log_R0_mut):
                raise TypeError("r0_backend.log_mutation_rates must return a tensor")
            log_R0_mut = log_R0_mut.to(device)
        elif cache_esm:
            log_R0_mut = get_lm_logits_cached(
                tokenizer, esm_model, aa_token_ids, active_seqs, max_seq_len,
                device, r0_cache, stats=r0_stats,
            )
        else:
            log_R0_mut = get_lm_logits(
                tokenizer, esm_model, aa_token_ids, active_seqs, max_seq_len, device
            )
        log_R0_mut = tilt_log_R0_by_fitness(
            log_R0_mut,
            beta=fitness_beta,
            score=fitness_score,
            mode=fitness_tilt_mode,
            sequences=active_seqs if fitness_tilt_mode == TILT_FULL_ESM else None,
            fitness_scorer=fitness_scorer,
            cache=fitness_cache,
            batch_size=int(fitness_esm_batch_size),
            top_k_aas=fitness_esm_top_k,
        )
        aa_indices = None
        if getattr(rate_heads, "use_mut_aa_emb", False):
            aa_indices = _build_seq_indices(active_seqs, max_seq_len, device)
        with torch.no_grad():
            h_t     = node_enc(plm_t, struct_t, lap_t)
            H_t, _  = tree_enc(h_t, node_ids_t, node_times_dict,
                                edge_index_t, branch_lens_t, t_scalar=t)
            if ablate_tree_context:
                H_t = torch.zeros_like(H_t)
            out     = rate_heads(
                H_t, active_idx, log_R0_mut,
                site_entropy=col_entropy,
                aa_indices=aa_indices,
                log_pssm=log_pssm,
            )
            if ablate_bridge:
                # Without bridge matching → pure R0 (no learned c_θ).
                out = dict(out)
                out["log_R_theta_mut"] = log_R0_mut

        new_node_seqs = dict(tree.node_seqs)

        for i, leaf_id in enumerate(active_leaves):
            seq     = tree.node_seqs[leaf_id]
            seq_len = min(len(seq), max_seq_len)
            log_R_i = out["log_R_theta_mut"][i]
            if site_softmax_sample:
                new_node_seqs[leaf_id] = mutate_sequence_site_softmax(
                    log_R_i, seq, seq_len, dt, mutation_rate_scale,
                    site_temperature=site_temperature,
                )
            else:
                new_node_seqs[leaf_id] = mutate_sequence_independent(
                    log_R_i, seq, seq_len, dt, mutation_rate_scale,
                )

            at_cap   = len(tree.active_leaves) >= max_leaves
            if branching_mode == "poisson_ref":
                from src.reference_process import sample_poisson_offspring
                n_ch = 0 if at_cap else min(sample_poisson_offspring(float(ref_lambda), dt), 2)
            else:
                lam      = out["branching_rate"][i].item() * branch_rate_scale
                p_branch = 1.0 - math.exp(-max(0.0, lam) * dt)
                n_ch     = 0 if at_cap else (2 if torch.rand(1).item() < p_branch else 0)

            if n_ch > 0:
                child_seqs = [new_node_seqs[leaf_id]] * n_ch
                tree = TreeState(
                    node_ids=tree.node_ids, root_id=tree.root_id,
                    edges=tree.edges, branch_lengths=tree.branch_lengths,
                    node_seqs=new_node_seqs, active_leaves=list(tree.active_leaves))
                tree = tree.branch_node(leaf_id, child_seqs)
                bl_pred = dt if ablate_branch_length_head else out["branch_length"][i].item()
                new_children = tree.get_children(leaf_id)
                tree = TreeState(
                    node_ids=tree.node_ids, root_id=tree.root_id,
                    edges=tree.edges,
                    branch_lengths={**tree.branch_lengths,
                                    **{(leaf_id, c): bl_pred for c in new_children}},
                    node_seqs=tree.node_seqs,
                    active_leaves=list(tree.active_leaves))
                new_node_seqs = dict(tree.node_seqs)
                for child_id in new_children:
                    node_birth_step.setdefault(child_id, step + 1)

        tree = TreeState(
            node_ids=tree.node_ids, root_id=tree.root_id,
            edges=tree.edges, branch_lengths=tree.branch_lengths,
            node_seqs=new_node_seqs, active_leaves=list(tree.active_leaves))

    return tree


#evaluation section results

def eval_sequence_quality(gen_tree, gen_leaves, root_seq, max_seq_len,
                          tokenizer, esm_model, aa_token_ids, device,
                          pll_prune_threshold):
    section("A. SEQUENCE QUALITY")

    # Root-to-leaf divergence
    divs = [1.0 - seq_identity(root_seq, gen_tree.node_seqs[l]) for l in gen_leaves]
    print(f"Root-to-leaf divergence (fraction of positions mutated from root):")
    print(f"  mean={sum(divs)/len(divs):.4f}  "
          f"min={min(divs):.4f}  max={max(divs):.4f}")

    # Leaf-to-leaf pairwise diversity
    sample = random.sample(gen_leaves, min(60, len(gen_leaves)))
    pairs  = [(a, b) for i, a in enumerate(sample) for b in sample[i+1:]]
    pairs  = random.sample(pairs, min(300, len(pairs)))
    leaf_div = [1.0 - seq_identity(gen_tree.node_seqs[a], gen_tree.node_seqs[b])
                for a, b in pairs]
    print(f"Leaf-to-leaf pairwise diversity ({len(pairs)} pairs):")
    if leaf_div:
        print(f"  mean={sum(leaf_div)/len(leaf_div):.4f}  "
              f"min={min(leaf_div):.4f}  max={max(leaf_div):.4f}")
    else:
        print("  (skipped: need >=2 leaves)")

    # ESM PLL
    leaf_seqs = [gen_tree.node_seqs[l] for l in gen_leaves]
    print(f"Computing ESM-2 PLL for {len(gen_leaves)} generated leaves")
    log_R0 = get_lm_logits(tokenizer, esm_model, aa_token_ids,
                            leaf_seqs, max_seq_len, device)
    plls = [esm_pll_seq(log_R0[i], leaf_seqs[i], max_seq_len)
            for i in range(len(gen_leaves))]
    print(f"ESM PLL (nats/position):")
    print(f"  mean={sum(plls)/len(plls):.4f}  "
          f"min={min(plls):.4f}  max={max(plls):.4f}")

    # Pruning
    surviving = [l for l, p in zip(gen_leaves, plls) if p >= pll_prune_threshold]
    print(f"Post-prune survival (PLL >= {pll_prune_threshold} nats/pos):")
    print(f"  {len(surviving)}/{len(gen_leaves)} leaves survive  "
          f"({100*len(surviving)/len(gen_leaves):.0f}%)")

    return plls, surviving


def eval_phylogenetic_coherence(gen_tree, gen_leaves):
    section("B. PHYLOGENETIC COHERENCE")
    print("Tests whether nearby nodes in the tree are more similar than random pairs.")

    cm = children_map(gen_tree)
    leaf_set = set(gen_leaves)

    # Sibling leaf pairs (both children of same parent are leaves)
    sibling_ids = []
    for node, children in cm.items():
        if len(children) == 2:
            a, b = children
            if a in leaf_set and b in leaf_set:
                sibling_ids.append(seq_identity(
                    gen_tree.node_seqs[a], gen_tree.node_seqs[b]))

    # Parent-child pairs where child is a leaf
    parent_child_ids = [
        seq_identity(gen_tree.node_seqs[p], gen_tree.node_seqs[c])
        for p, c in gen_tree.edges if c in leaf_set
    ]

    # Random leaf pairs (baseline)
    sample = random.sample(gen_leaves, min(60, len(gen_leaves)))
    pairs = [(a, b) for i, a in enumerate(sample) for b in sample[i+1:]]
    pairs = random.sample(pairs, min(300, len(pairs)))
    random_ids = [seq_identity(gen_tree.node_seqs[a], gen_tree.node_seqs[b])
                  for a, b in pairs]

    def fmt(vals, label):
        if not vals:
            return f"  {label}: N/A"
        m = sum(vals) / len(vals)
        return f"  {label} (n={len(vals):3d}): mean={m:.4f}  min={min(vals):.4f}  max={max(vals):.4f}"

    print(fmt(sibling_ids,      "Sibling leaf pairs    "))
    print(fmt(parent_child_ids, "Parent-child pairs    "))
    print(fmt(random_ids,       "Random leaf pairs     "))

    if sibling_ids and random_ids:
        delta = sum(sibling_ids)/len(sibling_ids) - sum(random_ids)/len(random_ids)
        result = "Pass!" if delta > 0 else "Fail"
        print(f"\n  Sibling vs random delta: {delta:+.4f}  [{result}]")


def eval_tree_structure(gen_tree, gen_leaves):
    section("C. TREE STRUCTURE")

    cm = children_map(gen_tree)
    internal = [n for n in gen_tree.node_ids if n in cm]

    # Bifurcating
    counts = [len(cm[n]) for n in internal]
    all_bif = all(c == 2 for c in counts)
    print(f"Strictly bifurcating: {all_bif}  (child counts seen: {sorted(set(counts))})")

    # Depth
    depths = bfs_depths(gen_tree)
    leaf_depths = [depths.get(l, 0) for l in gen_leaves]
    print(f"Leaf depth from root:")
    print(f"  mean={sum(leaf_depths)/len(leaf_depths):.1f}  "
          f"min={min(leaf_depths)}  max={max(leaf_depths)}")

    # Branch lengths
    bls = list(gen_tree.branch_lengths.values())
    if bls:
        print(f"Branch lengths ({len(bls)} edges, all positive: {all(b>0 for b in bls)}):")
        print(f"  mean={sum(bls)/len(bls):.6f}  "
              f"min={min(bls):.6f}  max={max(bls):.6f}")

    # Sackin + cherries
    sak = sackin_index(gen_tree, gen_leaves)
    cherries = cherry_count(gen_tree)
    print(f"Sackin index:  {sak}  (lower = more balanced)")
    print(f"Cherry count:  {cherries}  (sibling leaf pairs)")

    return leaf_depths, bls


def positional_recovery(root: str, gt: str, gen: str) -> dict:
    """
    Given root sequence and a GT leaf, split positions into:
      - conserved: root[i] == gt[i] -> model should keep root AA
      - mutating:  root[i] != gt[i]  -> model should reach gt AA

    Factorization (exact when wrong AA still counts as a site hit):
      mut_recovery     = P(gen==GT | root!=GT)
      site_recall      = P(gen!=root | root!=GT)
      aa_acc_given_hit = P(gen==GT | root!=GT & gen!=root)
      => mut_recovery = site_recall * aa_acc_given_hit  (when site_hits>0)

    Also reports site_precision = P(root!=GT | gen!=root).
    """
    L = min(len(root), len(gt), len(gen))
    mut_correct = mut_total = cons_correct = cons_total = 0
    site_hits = site_hit_correct = 0
    gen_mut_total = gen_mut_true = 0
    for i in range(L):
        r, g, m = root[i], gt[i], gen[i]
        if r == g:
            cons_total += 1
            if m == r:
                cons_correct += 1
        else:
            mut_total += 1
            if m == g:
                mut_correct += 1
            if m != r:
                site_hits += 1
                if m == g:
                    site_hit_correct += 1
        if m != r:
            gen_mut_total += 1
            if r != g:
                gen_mut_true += 1
    return {
        "mut_recovery":  mut_correct  / mut_total  if mut_total  else float("nan"),
        "cons_retention": cons_correct / cons_total if cons_total else float("nan"),
        "site_recall": site_hits / mut_total if mut_total else float("nan"),
        "aa_acc_given_hit": (
            site_hit_correct / site_hits if site_hits else float("nan")
        ),
        "site_precision": (
            gen_mut_true / gen_mut_total if gen_mut_total else float("nan")
        ),
        "mut_total":  mut_total,
        "cons_total": cons_total,
        "site_hits": site_hits,
        "gen_mut_total": gen_mut_total,
    }


def eval_gt_comparison(gen_tree, gen_leaves, gt_batch):
    section("D. GROUND TRUTH COMPARISON")

    gt_node_ids = gt_batch["node_ids"]
    gt_seqs = gt_batch["seqs"]
    gt_edges = gt_batch["edges"]
    gt_bls_dict = gt_batch["branch_lengths"]
    gt_root_id = gt_node_ids[gt_batch["root_index"]]

    gt_cm = defaultdict(list)
    for p, c in gt_edges:
        gt_cm[p].append(c)
    gt_leaf_set = set(n for n in gt_node_ids if n not in gt_cm)
    gt_leaves = list(gt_leaf_set)
    # Filter seqs to node_ids only (FASTA may have extra entries)
    gt_seqs_filtered = {nid: gt_seqs.get(nid, "") for nid in gt_node_ids}

    print(f"GT:  {len(gt_node_ids)} nodes, {len(gt_leaves)} leaves")
    print(f"Gen: {len(gen_tree.node_ids)} nodes, {len(gen_leaves)} leaves")

    # Best-match seq identity: each GT leaf -> closest generated leaf
    sample_gt = random.sample(gt_leaves, min(100, len(gt_leaves)))
    gt_root_seq = gt_seqs.get(gt_root_id, "")
    best_matches = []  # (gt_leaf, best_gen_leaf, identity)
    for gl in sample_gt:
        best_id, best_gen = max(
            (seq_identity(gt_seqs[gl], gen_tree.node_seqs[gn]), gn)
            for gn in gen_leaves)
        best_matches.append((gl, best_gen, best_id))
    best_ids = [m[2] for m in best_matches]
    print(f"\nBest-match identity (GT leaf → nearest gen leaf, n={len(sample_gt)}):")
    print(f"  mean={sum(best_ids)/len(best_ids):.4f}  "
          f"min={min(best_ids):.4f}  max={max(best_ids):.4f}")

    # Positional recovery: mutating vs conserved sites
    # For each (GT leaf → best gen leaf) pair, use GT root as anchor to classify positions
    all_mut_rec  = []
    all_cons_ret = []
    all_site_rec = []
    all_aa_acc = []
    for gl, best_gen, _ in best_matches:
        rec = positional_recovery(gt_root_seq, gt_seqs[gl], gen_tree.node_seqs[best_gen])
        if not (rec["mut_total"] == 0 and rec["cons_total"] == 0):
            all_mut_rec.append(rec["mut_recovery"])
            all_cons_ret.append(rec["cons_retention"])
            all_site_rec.append(rec["site_recall"])
            all_aa_acc.append(rec["aa_acc_given_hit"])

    def _mean(vals):
        finite = [v for v in vals if v == v]  # filter NaN
        return sum(finite) / len(finite) if finite else float("nan")

    print(f"\nPositional recovery (GT root as reference, n={len(all_mut_rec)} pairs):")
    print(f"  Mutating sites  (root→GT differs): "
          f"mean recovery   = {_mean(all_mut_rec):.4f}  "
          f"[fraction of GT mutations the model got right]")
    print(f"  Site recall     (gen!=root | root!=GT): "
          f"mean            = {_mean(all_site_rec):.4f}")
    print(f"  AA acc | hit    (gen==GT | hit): "
          f"mean            = {_mean(all_aa_acc):.4f}")
    print(f"  Conserved sites (root→GT same):    "
          f"mean retention  = {_mean(all_cons_ret):.4f}  "
          f"[fraction of conserved sites model left unchanged]")

    # Branch length distributions
    gt_bls_vals  = list(gt_bls_dict.values())
    gen_bls_vals = list(gen_tree.branch_lengths.values())
    if gt_bls_vals and gen_bls_vals:
        print(f"\nBranch length distribution:")
        print(f"  GT:  mean={sum(gt_bls_vals)/len(gt_bls_vals):.6f}  "
              f"max={max(gt_bls_vals):.6f}")
        print(f"  Gen: mean={sum(gen_bls_vals)/len(gen_bls_vals):.6f}  "
              f"max={max(gen_bls_vals):.6f}")

    # Depth distribution
    gt_tree_obj = TreeState(
        node_ids=gt_node_ids, root_id=gt_root_id,
        edges=gt_edges, branch_lengths=gt_bls_dict,
        node_seqs=gt_seqs_filtered, active_leaves=list(gt_leaf_set))
    gt_depths     = bfs_depths(gt_tree_obj)
    gt_leaf_depths = [gt_depths.get(l, 0) for l in gt_leaves]
    gen_depths_map = bfs_depths(gen_tree)
    gen_leaf_depths = [gen_depths_map.get(l, 0) for l in gen_leaves]

    print(f"\nLeaf depth from root:")
    print(f"  GT:  mean={sum(gt_leaf_depths)/len(gt_leaf_depths):.1f}  "
          f"min={min(gt_leaf_depths)}  max={max(gt_leaf_depths)}")
    print(f"  Gen: mean={sum(gen_leaf_depths)/len(gen_leaf_depths):.1f}  "
          f"min={min(gen_leaf_depths)}  max={max(gen_leaf_depths)}")

    # Sackin index + cherries
    gt_sak  = sackin_index(gt_tree_obj, gt_leaves)
    gen_sak = sackin_index(gen_tree, gen_leaves)
    gt_ch   = cherry_count(gt_tree_obj)
    gen_ch  = cherry_count(gen_tree)
    print(f"\nSackin index:  GT={gt_sak}  Gen={gen_sak}  "
          f"(ratio={gen_sak/gt_sak:.2f})" if gt_sak > 0 else "")
    print(f"Cherry count:  GT={gt_ch}    Gen={gen_ch}")


# main 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",          default="checkpoints/best.pt")
    parser.add_argument("--data",                default="data/train")
    parser.add_argument("--group",               type=int, required=True)
    parser.add_argument("--n-steps",             type=int,   default=50)
    parser.add_argument("--max-seq-len",         type=int,   default=566)
    parser.add_argument("--branch-rate-scale",   type=float, default=6.0)
    parser.add_argument("--max-leaves",          type=int,   default=200)
    parser.add_argument("--pll-prune-threshold",  type=float, default=-3.0)
    parser.add_argument("--mutation-rate-scale",  type=float, default=1.0,
                        help="Multiply CTMC mutation rate by this; >1 forces more mutations")
    parser.add_argument("--site-softmax-sample", action="store_true",
                        help="Sample mutating sites from a categorical over site "
                             "propensity (Σ_aa mut mass), then AA|site. Off by default.")
    parser.add_argument("--site-temperature", type=float, default=1.0,
                        help="Temperature on site-propensity logits (--site-softmax-sample).")
    parser.add_argument(
        "--fitness-beta", "--ref-tilt-beta",
        type=float, default=None, dest="fitness_beta",
        help="§4.2 R0 fitness tilt β. Default: checkpoint config. Alias: --ref-tilt-beta.",
    )
    parser.add_argument(
        "--fitness-score",
        choices=["log_R0", "log_softmax"],
        default=None,
        help="Fitness proxy for site_local tilting (default: checkpoint / log_R0).",
    )
    parser.add_argument(
        "--fitness-tilt-mode",
        choices=[TILT_SITE_LOCAL, TILT_FULL_ESM],
        default=None,
        help="site_local (Option A) or full_esm (Option B). Default: checkpoint.",
    )
    parser.add_argument("--fitness-esm-batch-size", type=int, default=None)
    parser.add_argument("--fitness-esm-top-k", type=int, default=None)
    parser.add_argument(
        "--r0-backend",
        default=None,
        help="Frozen R0 prior (esm2 / esm2_650m / esmc / jtt / wag / lg). "
             "Default: checkpoint config / ESM-2-8M logits path.",
    )
    parser.add_argument("--r0-model", default=None, help="Optional R0 model id override.")
    parser.add_argument(
        "--cache-esm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Incremental seq-keyed ESM emb + R0/MLM cache during generation "
             "(default: on). Use --no-cache-esm to disable.",
    )
    parser.add_argument("--seed",                type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    node_enc, tree_enc, rate_heads, col_entropy = load_models(args.checkpoint, device, args.max_seq_len)
    embedder = ESM2Embedder(device=device)

    model_id = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    esm_model = EsmForMaskedLM.from_pretrained(model_id).to(device)
    esm_model.eval()
    for p in esm_model.parameters():
        p.requires_grad = False
    aa_token_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(aa) for aa in AA_VOCAB], dtype=torch.long)

    dataset  = TreeDataset(args.data, max_seq_len=args.max_seq_len)
    group_idx = next((i for i in range(len(dataset))
                      if dataset.groups[i] == args.group), None)
    if group_idx is None:
        print(f"Group {args.group} not found"); sys.exit(1)

    gt_batch = dataset[group_idx]
    root_id  = gt_batch["node_ids"][gt_batch["root_index"]]
    root_seq = gt_batch["seqs"][root_id]
    print(f"\nGroup {args.group}: root_len={len(root_seq)}  "
          f"gt_nodes={len(gt_batch['node_ids'])}")
    print(f"Generating ({args.n_steps} steps, max_leaves={args.max_leaves}, "
          f"scale={args.branch_rate_scale})...")

    r0_live = None
    if args.r0_backend:
        from src.r0_backends import build_r0_backend, normalize_backend_name
        r0_name = normalize_backend_name(args.r0_backend)
        print(f"R0 backend override: {r0_name}")
        r0_live = build_r0_backend(r0_name, model_id=args.r0_model, device=device)

    gen_tree = generate_tree(
        root_seq, args.n_steps, args.max_seq_len,
        args.branch_rate_scale, args.max_leaves, args.mutation_rate_scale,
        node_enc, tree_enc, rate_heads, embedder,
        tokenizer, esm_model, aa_token_ids, device, col_entropy=col_entropy,
        site_softmax_sample=args.site_softmax_sample,
        site_temperature=args.site_temperature,
        fitness_beta=args.fitness_beta,
        fitness_score=args.fitness_score,
        fitness_tilt_mode=args.fitness_tilt_mode,
        fitness_esm_batch_size=args.fitness_esm_batch_size,
        fitness_esm_top_k=args.fitness_esm_top_k,
        cache_esm=args.cache_esm,
        r0_backend=r0_live,
    )
    gen_leaves = get_leaves(gen_tree)
    print(f"Generated: {len(gen_tree.node_ids)} nodes, {len(gen_leaves)} leaves")

    eval_sequence_quality(gen_tree, gen_leaves, root_seq, args.max_seq_len,
                          tokenizer, esm_model, aa_token_ids, device,
                          args.pll_prune_threshold)
    eval_phylogenetic_coherence(gen_tree, gen_leaves)
    eval_tree_structure(gen_tree, gen_leaves)
    eval_gt_comparison(gen_tree, gen_leaves, gt_batch)

    out_dir = Path("checkpoints")

    # Save FASTA (all nodes: root, internal, leaves)
    cm = children_map(gen_tree)
    out_fasta = out_dir / f"gen_group{args.group}.fasta"
    with open(out_fasta, "w") as f:
        for nid in gen_tree.node_ids:
            tag = ("root" if nid == gen_tree.root_id
                   else ("leaf" if nid not in cm else "internal"))
            f.write(f">{nid}|{tag}\n{gen_tree.node_seqs[nid]}\n")
    print(f"\nSequences saved to {out_fasta}")

    # Save Newick
    def _to_newick(nid: str) -> str:
        bl = gen_tree.branch_lengths.get(
            next(((p, nid) for p, c in gen_tree.edges if c == nid), (None, None)), 0.0)
        kids = cm.get(nid, [])
        if not kids:
            return f"{nid}:{bl:.8f}"
        return f"({','.join(_to_newick(c) for c in kids)}){nid}:{bl:.8f}"

    out_nwk = out_dir / f"gen_group{args.group}.nwk"
    out_nwk.write_text(_to_newick(gen_tree.root_id) + ";")
    print(f"Tree saved to    {out_nwk}")


if __name__ == "__main__":
    main()
