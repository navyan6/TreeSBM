#!/usr/bin/env python3
"""Controlled tree generation from a root sequence and TreeSBM checkpoint."""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

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
    make_sequence_pll_scorer,
    tilt_log_R0_by_fitness,
)
from src.r0_backends import build_r0_backend, normalize_backend_name
from src.reference_process import sample_poisson_offspring
from src.bridge.mutation_sample import (
    mutate_sequence_independent,
    mutate_sequence_site_softmax,
)

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}


def load_checkpoint(path, device, max_seq_len=566):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    node_enc  = NodeEncoder(d_plm=320, d_struct=3, d_laplacian=8, d_node=128).to(device)
    tree_enc  = TreeEncoder(d_model=128, n_layers=4, n_heads=8, dropout=0.1).to(device)
    r_heads   = RateHeads(
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
    node_enc.eval(); tree_enc.eval(); r_heads.eval()
    for m in [node_enc, tree_enc, r_heads]:
        for p in m.parameters():
            p.requires_grad = False
    col_entropy = ckpt.get("col_entropy", None)
    if col_entropy is not None:
        col_entropy = col_entropy.to(device)
    log_pssm = ckpt.get("log_pssm", None)
    if log_pssm is not None:
        log_pssm = log_pssm.to(device)
    elif cfg.get("use_pssm_gate"):
        raise RuntimeError(
            "checkpoint has use_pssm_gate=True but no saved log_pssm"
        )
    r_heads._train_log_pssm = log_pssm
    r_heads._col_entropy = col_entropy
    # Stash tilt / R0 config from checkpoint so generate matches train (CLI can override).
    r_heads._fitness_beta = float(cfg.get("fitness_beta", 0.0))
    r_heads._fitness_score = cfg.get("fitness_score", "log_R0")
    r_heads._fitness_tilt_mode = cfg.get("fitness_tilt_mode", TILT_SITE_LOCAL)
    r_heads._fitness_esm_batch_size = int(cfg.get("fitness_esm_batch_size", 8))
    r_heads._fitness_esm_top_k = cfg.get("fitness_esm_top_k")
    r_heads._ckpt_config = cfg
    return node_enc, tree_enc, r_heads


def get_lm_logits(r0_backend, sequences, max_seq_len, device):
    """Live R0 log-rates via multi-pLM / substitution backend."""
    log_rates = r0_backend.log_mutation_rates(sequences, max_seq_len=max_seq_len)
    return log_rates.to(device)


def tree_to_newick(tree: TreeState) -> str:
#treestate to newich str
    children_map: dict[str, list[str]] = {}
    for p, c in tree.edges:
        children_map.setdefault(p, []).append(c)

    def _fmt(nid: str) -> str:
        bl = tree.branch_lengths.get(
            next(((p, nid) for p, c in tree.edges if c == nid), (None, None)),
            0.0
        )
        children = children_map.get(nid, [])
        if not children:
            return f"{nid}:{bl:.6f}"
        inner = ",".join(_fmt(c) for c in children)
        return f"({inner}){nid}:{bl:.6f}"

    return _fmt(tree.root_id) + ";"


def generate_tree(args):
    seq = args.root_seq.upper()
    nt_chars = set("ACGTU")
    nt_frac = sum(c in nt_chars for c in seq) / max(len(seq), 1)
    if nt_frac > 0.85:
        raise ValueError(
            f"root-seq is a  nucleotide sequence ({nt_frac:.0%} ACGTU). "
            "Translate to amino acids first."
        )
    aa_frac = sum(c in AA_VOCAB for c in seq) / max(len(seq), 1)
    print(f"Root sequence: {len(seq)} aa  ({aa_frac:.0%} standard AA)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    node_enc, tree_enc, rate_heads = load_checkpoint(args.checkpoint, device, args.max_seq_len)

    # ESM-2-8M embeddings for the tree encoder (unchanged); R0 may be a different prior.
    embedder = ESM2Embedder(device=device)

    ckpt_cfg = getattr(rate_heads, "_ckpt_config", {})
    r0_name = getattr(args, "r0_backend", None) or ckpt_cfg.get("r0_backend", "esm2")
    r0_name = normalize_backend_name(r0_name)
    r0_model = getattr(args, "r0_model", None)
    print(f"R0 backend: {r0_name}  model={r0_model or '(default)'}")
    r0_backend = build_r0_backend(r0_name, model_id=r0_model, device=device)

    # t=0: root-only tree
    tree = TreeState.root_only(args.root_seq)
    # Track birth step for each node (used as proxy for calendar time in causal mask)
    node_birth_step: dict[str, int] = {tree.root_id: 0}

    n_steps = args.n_steps
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t = step / n_steps

        if not tree.active_leaves:
            print(f"Step {step}: no active leaves, stopping early")
            break

        node_ids_t    = tree.node_ids
        node_to_idx   = {nid: i for i, nid in enumerate(node_ids_t)}
        active_leaves = list(tree.active_leaves)
        active_idx    = [node_to_idx[v] for v in active_leaves]

        node_times_dict = {
            nid: node_birth_step.get(nid, 0) / n_steps for nid in node_ids_t
        }

        # Tree features for T_t
        struct_t = compute_structural_features(tree, node_to_idx).to(device)
        lap_t    = compute_laplacian_pe(tree, node_to_idx, 8, device=device)
        edge_index_t, _, edge_attr_t = build_edges(tree, node_to_idx)
        edge_index_t  = edge_index_t.to(device)
        branch_lens_t = edge_attr_t.squeeze(-1).to(device)

        # PLM embeddings [N, 320] for current sequences
        seqs_list = [tree.node_seqs[nid] for nid in node_ids_t]
        plm_t = embedder.embed_sequences(seqs_list).to(device)

        # R0 log-rates computed first so mutation head can condition on them
        active_seqs = [tree.node_seqs[v] for v in active_leaves]
        log_R0_mut  = get_lm_logits(r0_backend, active_seqs, args.max_seq_len, device)
        # §4.2: same tilt as training (CLI overrides checkpoint).
        fitness_beta = getattr(args, "fitness_beta", None)
        if fitness_beta is None:
            fitness_beta = getattr(rate_heads, "_fitness_beta", 0.0)
        fitness_score = getattr(args, "fitness_score", None) or getattr(
            rate_heads, "_fitness_score", "log_R0"
        )
        fitness_tilt_mode = getattr(args, "fitness_tilt_mode", None) or getattr(
            rate_heads, "_fitness_tilt_mode", TILT_SITE_LOCAL
        )
        fitness_cache = getattr(args, "_fitness_cache", None)
        if fitness_cache is None:
            fitness_cache = {}
            args._fitness_cache = fitness_cache
        fitness_scorer = getattr(args, "_fitness_scorer", None)
        if (
            fitness_tilt_mode == TILT_FULL_ESM
            and float(fitness_beta) != 0.0
            and fitness_scorer is None
        ):
            fitness_scorer = make_sequence_pll_scorer(
                r0_backend, max_seq_len=args.max_seq_len
            )
            args._fitness_scorer = fitness_scorer
        top_k = getattr(args, "fitness_esm_top_k", None)
        if top_k is None:
            top_k = getattr(rate_heads, "_fitness_esm_top_k", None)
        batch_sz = getattr(args, "fitness_esm_batch_size", None)
        if batch_sz is None:
            batch_sz = getattr(rate_heads, "_fitness_esm_batch_size", 8)
        log_R0_mut = tilt_log_R0_by_fitness(
            log_R0_mut,
            beta=fitness_beta,
            score=fitness_score,
            mode=fitness_tilt_mode,
            sequences=active_seqs if fitness_tilt_mode == TILT_FULL_ESM else None,
            fitness_scorer=fitness_scorer,
            cache=fitness_cache,
            batch_size=int(batch_sz),
            top_k_aas=top_k,
        )

        aa_indices = None
        if getattr(rate_heads, "use_mut_aa_emb", False):
            aa_indices = _build_seq_indices(active_seqs, args.max_seq_len, device)
        col_entropy = getattr(rate_heads, "_col_entropy", None)
        log_pssm = getattr(rate_heads, "_train_log_pssm", None)

        # NodeEncoder -> TreeEncoder -> RateHeads
        with torch.no_grad():
            h_t  = node_enc(plm_t, struct_t, lap_t)
            H_t, _ = tree_enc(h_t, node_ids_t, node_times_dict,
                               edge_index_t, branch_lens_t, t_scalar=t)
            out  = rate_heads(
                H_t, active_idx, log_R0_mut,
                site_entropy=col_entropy,
                aa_indices=aa_indices,
                log_pssm=log_pssm,
            )
        # out["log_R_theta_mut"] = log_R0 + c_θ  [n_active, L, 20]

        # Sample events for each active leaf
        new_node_seqs = dict(tree.node_seqs)
        mrs = getattr(args, "mutation_rate_scale", 1.0)

        for i, leaf_id in enumerate(active_leaves):
            seq     = tree.node_seqs[leaf_id]
            seq_len = min(len(seq), args.max_seq_len)
            log_R_i = out["log_R_theta_mut"][i]
            if args.site_softmax_sample:
                new_node_seqs[leaf_id] = mutate_sequence_site_softmax(
                    log_R_i, seq, seq_len, dt, mrs,
                    site_temperature=args.site_temperature,
                )
            else:
                new_node_seqs[leaf_id] = mutate_sequence_independent(
                    log_R_i, seq, seq_len, dt, mrs,
                )

            # Branch: learned RateHeads (default) or paper Alg. 3 Poisson(λ Δt).
            branching_mode = getattr(args, "branching_mode", "learned")
            if branching_mode == "poisson_ref":
                ref_lam = float(getattr(args, "ref_lambda", 1.0))
                n_ch = sample_poisson_offspring(ref_lam, dt)
                n_ch = min(n_ch, 2)  # keep bifurcate clamp for fair TreeSBM compare
            else:
                lam  = out["branching_rate"][i].item() * args.branch_rate_scale
                p_branch = 1.0 - math.exp(-max(0.0, lam) * dt)
                n_ch = 2 if torch.rand(1).item() < p_branch else 0
            if n_ch > 0:
                child_seqs = [new_node_seqs[leaf_id]] * n_ch
                tree = TreeState(
                    node_ids=tree.node_ids, root_id=tree.root_id,
                    edges=tree.edges, branch_lengths=tree.branch_lengths,
                    node_seqs=new_node_seqs,
                    active_leaves=list(tree.active_leaves),
                )
                tree = tree.branch_node(leaf_id, child_seqs)

                bl_pred = out["branch_length"][i].item()
                new_children = tree.get_children(leaf_id)
                new_bls = {(leaf_id, c): bl_pred for c in new_children}
                tree = TreeState(
                    node_ids=tree.node_ids, root_id=tree.root_id,
                    edges=tree.edges,
                    branch_lengths={**tree.branch_lengths, **new_bls},
                    node_seqs=tree.node_seqs,
                    active_leaves=list(tree.active_leaves),
                )
                new_node_seqs = dict(tree.node_seqs)

                # ESM fitness gate: terminate new children with very low PLL
                for child_id in new_children:
                    node_birth_step.setdefault(child_id, step + 1)
                    child_seq = new_node_seqs[child_id]
                    child_len = min(len(child_seq), args.max_seq_len)
                    aa_idx_c = torch.tensor(
                        [AA_TO_IDX.get(aa, 20) for aa in child_seq[:child_len]],
                        dtype=torch.long, device=device,
                    )
                    valid = aa_idx_c < 20
                    if valid.any():
                        child_pll = (
                            log_R0_mut[i, :child_len]
                            .gather(-1, aa_idx_c.clamp(max=19).unsqueeze(-1))
                            .squeeze(-1)[valid]
                            .mean()
                            .item()
                        )
                        if child_pll < args.pll_threshold:
                            tree = tree.terminate_leaf(child_id)

        # Flush sequence updates for non-branching leaves
        tree = TreeState(
            node_ids=tree.node_ids, root_id=tree.root_id,
            edges=tree.edges, branch_lengths=tree.branch_lengths,
            node_seqs=new_node_seqs,
            active_leaves=list(tree.active_leaves),
        )

        print(
            f"Step {step + 1:03d}/{n_steps}  "
            f"nodes={len(tree.node_ids)}  "
            f"active_leaves={len(tree.active_leaves)}"
        )

    print(f"\nFinal tree: {len(tree.node_ids)} nodes, {len(tree.active_leaves)} leaves")

    nwk = tree_to_newick(tree)
    out_path = Path(args.output)
    out_path.write_text(nwk)
    print(f"Saved to {out_path}")

    # Save all node sequences as FASTA
    fasta_path = out_path.with_suffix(".fasta")
    has_children = {p for p, c in tree.edges}
    with open(fasta_path, "w") as f:
        for nid in tree.node_ids:
            tag = "root" if nid == tree.root_id else ("leaf" if nid not in has_children else "internal")
            f.write(f">{nid}|{tag}\n{tree.node_seqs[nid]}\n")
    print(f"Sequences saved to {fasta_path}")
    r0_backend.close()
    return tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True,
                        help="Path to trained checkpoint (e.g. checkpoints/best.pt)")
    parser.add_argument("--root-seq",      required=True,
                        help="Root amino acid sequence")
    parser.add_argument("--n-steps",       type=int,   default=50)
    parser.add_argument("--output",        default="generated_tree.nwk")
    parser.add_argument("--max-seq-len",   type=int,   default=566)
    parser.add_argument("--pll-threshold",    type=float, default=-100.0,
                        help="Terminate new child if ESM PLL < this (nats/position); -100 disables gate")
    parser.add_argument("--beta",             type=float, default=1.0,
                        help="(legacy, unused) MH acceptance temperature")
    parser.add_argument(
        "--fitness-beta", "--ref-tilt-beta",
        type=float, default=None, dest="fitness_beta",
        help="§4.2 R0 fitness tilt β. Default: checkpoint config, else 0. "
             "Alias: --ref-tilt-beta.",
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
        help="site_local (Option A, default) or full_esm (Option B mutant PLL). "
             "Default: checkpoint config / site_local.",
    )
    parser.add_argument(
        "--fitness-esm-batch-size",
        type=int,
        default=None,
        help="Batch size for full_esm mutant scoring.",
    )
    parser.add_argument(
        "--fitness-esm-top-k",
        type=int,
        default=None,
        help="Only score/tilt top-k untilted AAs per site under full_esm.",
    )
    parser.add_argument(
        "--r0-backend",
        default=None,
        help="Frozen R0 prior for live logits (default: checkpoint config, else esm2). "
             "Frozen R0 backend: esm2 / esm2_650m / esmc / jtt / wag / lg.",
    )
    parser.add_argument(
        "--r0-model",
        default=None,
        help="Optional model id override for --r0-backend.",
    )
    parser.add_argument(
        "--branching-mode",
        choices=["learned", "poisson_ref"],
        default="learned",
        help="Branching dynamics: 'learned' = RateHeads Bernoulli bifurcate (default); "
             "'poisson_ref' = Alg. 3 Poisson(λ Δt) with --ref-lambda (D.2).",
    )
    parser.add_argument(
        "--ref-lambda",
        type=float,
        default=1.0,
        help="Constant Poisson λ when --branching-mode poisson_ref.",
    )
    parser.add_argument("--branch-rate-scale", type=float, default=6.0,
                        help="Multiply model branching rate by this at inference (corrects lam≈1 → lam≈6)")
    parser.add_argument("--mutation-rate-scale", type=float, default=1.0,
                        help="Multiply mutation fire rate by this at inference")
    parser.add_argument("--site-softmax-sample", action="store_true",
                        help="Sample sites from categorical propensity then AA|site "
                             "(antiGen-style; off by default)")
    parser.add_argument("--site-temperature", type=float, default=1.0,
                        help="Temperature on site-propensity logits (--site-softmax-sample)")
    args = parser.parse_args()
    generate_tree(args)


if __name__ == "__main__":
    main()
