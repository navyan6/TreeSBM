#!/usr/bin/env python3
"""
Bridge matching training loop for TreeSBM (Algorithm 1).

Each step:
  1. Load T1 from dataset, compute PLM embeddings once.
  2. Sample t ~ U(0, t_max_clip).
  3. Construct T_t via SampleBridgeState (Algorithm 2).
  4. Rebuild TreeState / structural features / Laplacian PE for T_t.
  5. Run NodeEncoder and TreeEncoder (with time conditioning) to get H_t.
  6. Run RateHeads for active leaves of T_t.
  7. Compute bridge matching loss vs. T1 targets.

Usage:
    python scripts/train.py --data data/train --epochs 100 --lr 1e-4
"""

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import json
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).parent.parent
import sys; sys.path.insert(0, str(ROOT))

from src.dataset import TreeDataset, ConcatGeneTreeDataset, BalancedGeneSampler
from src.tree_state import TreeState
from src.treeencoder.plm_embeddings import ESM2Embedder
from src.treeencoder.node_encoder import NodeEncoder
from src.treeencoder.structural_features import compute_structural_features
from src.treeencoder.laplacian import compute_laplacian_pe
from src.treeencoder.edges import build_edges
from src.networks import TreeEncoder, RateHeads
from src.bridge.fitness_tilt import (
    TILT_FULL_ESM,
    TILT_SITE_LOCAL,
    make_sequence_pll_scorer,
    tilt_log_R0_by_fitness,
)
from src.bridge.sample_bridge_state import sample_bridge_state
from src.bridge.losses import (
    bridge_losses,
    attach_semigroup_loss,
    _build_seq_indices,
    select_mut_hotspots,
)
from src.bridge.site_stats import compute_msa_column_mut_freq
from src.bridge.semigroup import sample_time_triple, semigroup_loss_from_predictor


def _parse_gene_entries(entries: list[str] | None) -> list[tuple[str, str]]:
    """Parse repeated gene_id=path CLI entries."""
    if not entries:
        return []
    out = []
    for raw in entries:
        if "=" not in raw:
            raise SystemExit(f"Bad --gene-*-data entry {raw!r}; expected gene_id=path")
        gid, path = raw.split("=", 1)
        gid, path = gid.strip(), path.strip()
        if not gid or not path:
            raise SystemExit(f"Bad --gene-*-data entry {raw!r}")
        out.append((gid, path))
    return out


def _gene_ids_for_dataset(ds) -> list[str]:
    if isinstance(ds, ConcatGeneTreeDataset):
        return list(ds.gene_ids)
    if isinstance(ds, Subset):
        base = ds.dataset
        return [_gene_ids_for_dataset(base)[i] for i in ds.indices]
    if hasattr(ds, "gene_id_at"):
        return [ds.gene_id_at(i) for i in range(len(ds))]
    return ["default"] * len(ds)


def _subset_by_gene(ds, gene: str):
    """Lightweight view of a dataset restricted to one gene_id."""
    ids = _gene_ids_for_dataset(ds)
    indices = [i for i, g in enumerate(ids) if g == gene]
    return Subset(ds, indices)


def _resolve_col_entropy(col_entropy, batch):
    """Pick per-gene [L] entropy when col_entropy is a gene_id→tensor map."""
    if col_entropy is None:
        return None
    if isinstance(col_entropy, dict):
        gid = batch.get("gene_id", "default")
        if isinstance(gid, (list, tuple)):
            gid = gid[0] if gid else "default"
        if gid in col_entropy:
            return col_entropy[gid]
        if "default" in col_entropy:
            return col_entropy["default"]
        # Fallback: any available gene vector (should be rare).
        return next(iter(col_entropy.values())) if col_entropy else None
    return col_entropy


def _serialize_col_entropy(col_entropy):
    if col_entropy is None:
        return None
    if isinstance(col_entropy, dict):
        return {k: v.detach().cpu() for k, v in col_entropy.items()}
    return col_entropy.detach().cpu()


def compute_site_entropy_from_log_probs(log_R0_mut: torch.Tensor) -> torch.Tensor:
    """Per-position Shannon entropy in nats from [n_active, L, 20] log-probs/logits."""
    log_probs = torch.log_softmax(log_R0_mut, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"


def _train_alignment_aa_counts(dataset, max_seq_len: int):
    """Count AA occurrences per column from TRAIN sequences only (no val/test leakage)."""
    import numpy as np

    tbl = np.full(256, 255, dtype=np.int16)
    for i, aa in enumerate(AA_VOCAB):
        tbl[ord(aa)] = i  # non-AA bytes (X, -, *, ...) stay 255 -> excluded

    counts = np.zeros((max_seq_len, 20), dtype=np.int64)
    for i in range(len(dataset)):
        for seq in dataset[i]["seqs"].values():
            if not seq:
                continue
            # latin-1 is a 1:1 byte map (never drops/shifts a column); protein
            # chars are all <128 so this equals ascii for them.
            arr = tbl[np.frombuffer(seq[:max_seq_len].encode("latin-1"), dtype=np.uint8)]
            valid = arr < 20
            if valid.any():
                pos = np.nonzero(valid)[0]
                np.add.at(counts, (pos, arr[valid]), 1)
    return counts


def compute_empirical_column_entropy(dataset, max_seq_len: int) -> torch.Tensor:
    """
    Per-column Shannon entropy over ALL training-tree sequences, normalized to
    [0,1] by log(20). This is the "which sites historically vary" hotspot signal
    (antigenic sites -- e.g. spike RBD, D614G -- score high), computed ONCE from
    the training alignment and then used identically at train and generation so
    the two never diverge. Columns with no observed residue get 0.

    Uses only `dataset` (the TRAIN split) -- never val/test -- so no leakage.
    """
    import numpy as np

    counts = _train_alignment_aa_counts(dataset, max_seq_len)
    total = counts.sum(axis=1, keepdims=True)
    probs = counts / np.maximum(total, 1)
    safe_log = np.log(np.maximum(probs, 1e-12))   # avoid log(0); masked by probs=0 anyway
    ent = -(probs * safe_log).sum(axis=1)          # nats
    ent = np.clip(ent / math.log(20.0), 0.0, 1.0)
    ent[total[:, 0] == 0] = 0.0
    return torch.tensor(ent, dtype=torch.float32)


def compute_empirical_log_pssm(
    dataset, max_seq_len: int, pseudocount: float = 0.5
) -> torch.Tensor:
    """
    Train-alignment log-PSSM [L, 20] with Laplace pseudocounts.

    Used by the optional static PSSM gate on log R_θ (flag --pssm-gate). Uniform
    log-probs fill empty columns. TRAIN split only.
    """
    import numpy as np

    raw = _train_alignment_aa_counts(dataset, max_seq_len)
    empty = raw.sum(axis=1) == 0
    counts = raw.astype(np.float64) + pseudocount
    probs = counts / counts.sum(axis=1, keepdims=True)
    log_pssm = np.log(np.maximum(probs, 1e-12))
    log_pssm[empty] = -math.log(20.0)
    return torch.tensor(log_pssm, dtype=torch.float32)


def forward_bridge_step(
    batch: dict,
    t: float,
    node_enc: NodeEncoder,
    tree_enc: TreeEncoder,
    rate_heads: RateHeads,
    device: str,
    lap_dim: int = 8,
    max_seq_len: int = 566,
    lambda_top: float = 0.1,
    lambda_br: float = 0.1,
    lambda_stop: float = 0.1,
    lambda_pll: float = 0.01,
    lambda_mut: float = 5.0,
    lambda_cons: float = 1.0,
    lambda_semi: float = 0.0,
    t_max: float = 0.95,
    bridge_c: float = 1.0,
    use_site_entropy: bool = False,
    use_entropy_loss_weighting: bool = False,
    use_entropy_cons_weighting: bool = False,
    entropy_weight_alpha: float = 1.0,
    entropy_weight_alpha_cons: float | None = None,
    entropy_weight_floor: float = 1.0,
    entropy_is_normalized: bool = False,
    mut_normalize: str = "mean",
    col_entropy: torch.Tensor | None = None,
    mut_hotspot_mask: torch.Tensor | None = None,
    mut_hotspot_weight: float = 1.0,
    mut_hotspot_force: bool = False,
    log_pssm: torch.Tensor | None = None,
    embedder: ESM2Embedder | None = None,
    fitness_beta: float = 0.0,
    fitness_score: str = "log_R0",
    fitness_tilt_mode: str = TILT_SITE_LOCAL,
    fitness_scorer=None,
    fitness_cache: dict | None = None,
    fitness_esm_batch_size: int = 8,
    fitness_esm_top_k: int | None = None,
    ablate_terminal_only: bool = False,
    ablate_doob: bool = False,
) -> tuple[dict | None, int]:
    """
    One forward pass of Algorithm 1.
    or
    Returns (losses_dict, n_active_leaves).

    When lambda_semi > 0, also samples 0≤s<r<u≤t_max, builds bridge state at s,
    and adds rate-composition L_semi (§4.5) to the total.

    fitness_beta / fitness_score / fitness_tilt_mode: §4.2 tilt of log_R0 before
    RateHeads so log R_θ = log R0_tilted + c_θ (β=0 disables).
    mode=site_local (default Option A) or full_esm (Option B; needs fitness_scorer).
    """
    node_ids= batch["node_ids"]              # list[str], N
    node_times_t    = batch["node_times"]            # [N] tensor
    seqs            = batch["seqs"]                  # dict[str, str]
    edges           = batch["edges"]                 # list[(parent, child)]
    branch_lengths  = batch["branch_lengths"]        # dict[(str,str), float]
    root_id         = node_ids[batch["root_index"]]

    node_times_dict = {nid: node_times_t[i].item() for i, nid in enumerate(node_ids)}

    if batch.get("plm_embeddings") is not None:
        plm_T1 = batch["plm_embeddings"].to(device)              # [N, 320] cached
    elif embedder is not None:
        sequences = [seqs[nid] for nid in node_ids]
        with torch.no_grad():
            plm_T1 = embedder.embed_sequences(sequences).to(device)
    else:
        raise RuntimeError("No PLM embeddings: run scripts/precompute_plm.py first")
    plm_map = {nid: i for i, nid in enumerate(node_ids)}

    T_t = sample_bridge_state(
        t=t,
        node_ids=node_ids,
        node_times_dict=node_times_dict,
        edges=edges,
        branch_lengths=branch_lengths,
        seqs=seqs,
        root_id=root_id,
    )

    node_ids_t      = T_t["node_ids_t"]
    edges_t         = T_t["edges_t"]
    branch_lengths_t = T_t["branch_lengths_t"]
    seqs_t          = T_t["seqs_t"]
    active_leaves_t = T_t["active_leaves_t"]

    if len(node_ids_t) == 0 or len(active_leaves_t) == 0:
        return None, 0

    tree_t = TreeState(
        node_ids=node_ids_t,
        root_id=root_id,
        edges=edges_t,
        branch_lengths=branch_lengths_t,
        node_seqs=seqs_t,
        active_leaves=active_leaves_t,
    )
    node_to_idx_t = {nid: i for i, nid in enumerate(node_ids_t)}

    #compute necessary features

    struct_t = compute_structural_features(tree_t, node_to_idx_t).to(device)
    lap_t = compute_laplacian_pe(tree_t, node_to_idx_t, lap_dim, device=device)

    plm_t = torch.stack([plm_T1[plm_map[nid]] for nid in node_ids_t]).to(device)  # [N_t, 320]

    
    h_t = node_enc(plm_t.to(device), struct_t.to(device), lap_t.to(device))  # [N_t, 128]

    # edge tensors
    edge_index_t, _, edge_attr_t = build_edges(tree_t, node_to_idx_t)
    edge_index_t = edge_index_t.to(device)
    branch_lens_t = edge_attr_t.squeeze(-1).to(device) 

    # treeencoder + time
    H_t, _ = tree_enc(
        h_t, node_ids_t, node_times_dict, edge_index_t, branch_lens_t, t_scalar=t
    ) 

    # R0 must be computed before rate_heads so the mutation head can condition on it
    active_idx_t = [node_to_idx_t[nid] for nid in active_leaves_t]
    if batch.get("log_ref_mut_rates") is not None:
        log_R0_mut = torch.stack([
            batch["log_ref_mut_rates"][plm_map[nid]] for nid in active_leaves_t
        ]).to(device)                                      # [n_active, 566, 20]
    else:
        log_R0_mut = torch.zeros(len(active_leaves_t), max_seq_len, 20, device=device)

    site_entropy = None
    ent_is_norm = entropy_is_normalized
    if use_site_entropy or use_entropy_loss_weighting or use_entropy_cons_weighting:
        ce = _resolve_col_entropy(col_entropy, batch)
        if ce is not None:
            # Empirical column entropy from the training alignment, already
            # normalized to [0,1]. A [L] vector; RateHeads / bridge_losses
            # broadcast it over active leaves. Used identically here and at
            # generation (loaded from the checkpoint) so the two never diverge.
            # Multi-gene: dict keyed by gene_id — never pool non-homologous columns.
            site_entropy = ce.to(device=log_R0_mut.device, dtype=log_R0_mut.dtype)
            ent_is_norm = True
        else:
            # Fallback: ESM self-entropy H(softmax(log_R0)) per position.
            site_entropy = compute_site_entropy_from_log_probs(log_R0_mut)
            if entropy_is_normalized:
                site_entropy = (site_entropy / math.log(20.0)).clamp(min=0.0, max=1.0)

    # §4.2: tilt R0 before RateHeads / bridge losses.
    # Entropy (above) uses untilted R0 so β does not change the entropy signal.
    # site_local = Option A; full_esm = Option B (needs fitness_scorer + sequences).
    active_seqs_for_tilt = [seqs_t[nid] for nid in active_leaves_t]
    log_R0_mut = tilt_log_R0_by_fitness(
        log_R0_mut,
        beta=fitness_beta,
        score=fitness_score,
        mode=fitness_tilt_mode,
        sequences=active_seqs_for_tilt if fitness_tilt_mode == TILT_FULL_ESM else None,
        fitness_scorer=fitness_scorer,
        cache=fitness_cache,
        batch_size=fitness_esm_batch_size,
        top_k_aas=fitness_esm_top_k,
    )

    active_seqs_t = active_seqs_for_tilt
    aa_indices = None
    if getattr(rate_heads, "needs_aa_indices", False):
        aa_indices = _build_seq_indices(active_seqs_t, max_seq_len, device)

    pssm_t = None
    if getattr(rate_heads, "use_pssm_gate", False):
        if log_pssm is None:
            raise RuntimeError("use_pssm_gate requires log_pssm (train-alignment PSSM)")
        pssm_t = log_pssm.to(device=log_R0_mut.device, dtype=log_R0_mut.dtype)

    out = rate_heads(
        H_t, active_idx_t, log_R0_mut,
        site_entropy=site_entropy,
        aa_indices=aa_indices,
        log_pssm=pssm_t,
    )
            # out["log_R_theta_mut"] = log_R0 + c_θ (optionally PSSM-gated), inside RateHeads

    # §4.2: pass the same tilted R0 used by RateHeads into Doob / PLL losses.
    losses = bridge_losses(
        log_R_theta_mut=out["log_R_theta_mut"],
        log_R_theta_branch=out["branching_rate"],
        branch_length_pred=out["branch_length"],
        stop_prob=out["stop_prob"],
        log_R0_mut=log_R0_mut,
        seqs_t=active_seqs_t,
        active_leaves=active_leaves_t,
        T1_mut_targets=T_t["T1_mut_targets"],
        T1_child_counts=T_t["T1_child_counts"],
        T1_child_bls=T_t["T1_child_bls"],
        t=t,
        max_seq_len=max_seq_len,
        lambda_top=lambda_top,
        lambda_br=lambda_br,
        lambda_stop=lambda_stop,
        lambda_pll=lambda_pll,
        lambda_mut=lambda_mut,
        lambda_cons=lambda_cons,
        bridge_c=bridge_c,
        device=device,
        site_entropy=site_entropy,
        use_entropy_loss_weighting=use_entropy_loss_weighting,
        use_entropy_cons_weighting=use_entropy_cons_weighting,
        entropy_weight_alpha=entropy_weight_alpha,
        entropy_weight_alpha_cons=entropy_weight_alpha_cons,
        entropy_weight_floor=entropy_weight_floor,
        entropy_is_normalized=ent_is_norm,
        mut_normalize=mut_normalize,
        mut_hotspot_mask=mut_hotspot_mask,
        mut_hotspot_weight=mut_hotspot_weight,
        mut_hotspot_force=mut_hotspot_force,
        ablate_terminal_only=ablate_terminal_only,
        ablate_doob=ablate_doob,
    )

    # ── L_semi: rate-composition consistency from an earlier bridge state T_s
    if lambda_semi > 0.0:
        s, r, u = sample_time_triple(t_max=t_max)
        T_s = sample_bridge_state(
            t=s,
            node_ids=node_ids,
            node_times_dict=node_times_dict,
            edges=edges,
            branch_lengths=branch_lengths,
            seqs=seqs,
            root_id=root_id,
        )
        node_ids_s = T_s["node_ids_t"]
        active_s = T_s["active_leaves_t"]
        if len(node_ids_s) > 0 and len(active_s) > 0:
            tree_s = TreeState(
                node_ids=node_ids_s,
                root_id=root_id,
                edges=T_s["edges_t"],
                branch_lengths=T_s["branch_lengths_t"],
                node_seqs=T_s["seqs_t"],
                active_leaves=active_s,
            )
            node_to_idx_s = {nid: i for i, nid in enumerate(node_ids_s)}
            struct_s = compute_structural_features(tree_s, node_to_idx_s).to(device)
            lap_s = compute_laplacian_pe(tree_s, node_to_idx_s, lap_dim, device=device)
            plm_s = torch.stack([plm_T1[plm_map[nid]] for nid in node_ids_s]).to(device)
            h_s = node_enc(plm_s, struct_s, lap_s)
            edge_index_s, _, edge_attr_s = build_edges(tree_s, node_to_idx_s)
            edge_index_s = edge_index_s.to(device)
            branch_lens_s = edge_attr_s.squeeze(-1).to(device)
            active_idx_s = [node_to_idx_s[nid] for nid in active_s]
            if batch.get("log_ref_mut_rates") is not None:
                log_R0_s = torch.stack([
                    batch["log_ref_mut_rates"][plm_map[nid]] for nid in active_s
                ]).to(device)
            else:
                log_R0_s = torch.zeros(len(active_s), max_seq_len, 20, device=device)
            site_ent_s = None
            if use_site_entropy or use_entropy_loss_weighting or use_entropy_cons_weighting:
                ce_s = _resolve_col_entropy(col_entropy, batch)
                if ce_s is not None:
                    site_ent_s = ce_s.to(device=log_R0_s.device, dtype=log_R0_s.dtype)
                else:
                    site_ent_s = compute_site_entropy_from_log_probs(log_R0_s)
                    if entropy_is_normalized:
                        site_ent_s = (site_ent_s / math.log(20.0)).clamp(min=0.0, max=1.0)

            seqs_s = [T_s["seqs_t"][nid] for nid in active_s]
            log_R0_s = tilt_log_R0_by_fitness(
                log_R0_s,
                beta=fitness_beta,
                score=fitness_score,
                mode=fitness_tilt_mode,
                sequences=seqs_s if fitness_tilt_mode == TILT_FULL_ESM else None,
                fitness_scorer=fitness_scorer,
                cache=fitness_cache,
                batch_size=fitness_esm_batch_size,
                top_k_aas=fitness_esm_top_k,
            )

            aa_idx_s = None
            if getattr(rate_heads, "needs_aa_indices", False):
                aa_idx_s = _build_seq_indices(
                    [T_s["seqs_t"][nid] for nid in active_s], max_seq_len, device
                )
            pssm_s = None
            if getattr(rate_heads, "use_pssm_gate", False):
                if log_pssm is None:
                    raise RuntimeError("use_pssm_gate requires log_pssm")
                pssm_s = log_pssm.to(device=log_R0_s.device, dtype=log_R0_s.dtype)

            def _rates_at_time(tau: float):
                # Absolute bridge clock (same t_scalar convention as main step).
                H_d, _ = tree_enc(
                    h_s, node_ids_s, node_times_dict, edge_index_s, branch_lens_s,
                    t_scalar=tau,
                )
                return rate_heads(
                    H_d, active_idx_s, log_R0_s,
                    site_entropy=site_ent_s,
                    aa_indices=aa_idx_s,
                    log_pssm=pssm_s,
                )

            L_semi = semigroup_loss_from_predictor(_rates_at_time, s, r, u)
            losses = attach_semigroup_loss(losses, L_semi, lambda_semi)

    return losses, len(active_leaves_t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="data/train")
    parser.add_argument("--val-data",    default=None,
                        help="Optional separate validation directory (pre-split). "
                             "If set, --test-data is also required.")
    parser.add_argument(
        "--gene-data",
        action="append",
        default=None,
        help="Panviral multi-source train entry gene_id=path (repeatable). "
             "When set, overrides --data for training and enables shared-head "
             "multi-gene loading. Example: --gene-data h3n2_ha=data/h3n2/train",
    )
    parser.add_argument(
        "--gene-val-data",
        action="append",
        default=None,
        help="Optional gene_id=path val entries matching --gene-data.",
    )
    parser.add_argument(
        "--gene-test-data",
        action="append",
        default=None,
        help="Optional gene_id=path test entries matching --gene-data.",
    )
    parser.add_argument(
        "--infer-gene-id",
        action="store_true",
        help="Read gene_id/subtype from each group's meta CSV (pan-flu mixed dirs).",
    )
    parser.add_argument(
        "--balanced-gene-sample",
        action="store_true",
        help="Sample gene_id uniformly each step, then a tree of that gene "
             "(avoids Spike/H3 dominating by tree count).",
    )
    parser.add_argument(
        "--panviral-shared-head",
        action="store_true",
        help="Panviral defaults: no --per-site-pos-emb, no --pssm-gate, no lit "
             "hotspot mask; keep site-entropy + mut-aa-emb; shared RateHeads.",
    )
    parser.add_argument("--test-data",   default=None,
                        help="Pre-split test dir, e.g. data/h3n2/test or data/covid/test.")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--val-frac",    type=float, default=0.1)
    parser.add_argument("--test-frac",   type=float, default=0.1)
    parser.add_argument("--ckpt-dir",    default="checkpoints")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--t-max",       type=float, default=0.95,
                        help="Max t to sample (avoids 1/(1-t) blow-up near t=1)")
    parser.add_argument("--lambda-top",  type=float, default=0.1)
    parser.add_argument("--lambda-br",   type=float, default=0.1)
    parser.add_argument("--lambda-stop", type=float, default=0.1)
    parser.add_argument("--lambda-pll",  type=float, default=0.01)
    parser.add_argument("--lambda-mut",  type=float, default=5.0,
                        help="Upweight loss at mutating positions (T_t_aa != T1_aa) within L_rate")
    parser.add_argument("--lambda-cons", type=float, default=1.0,
                        help="Weight on conserved-position bridge term L_cons inside L_rate "
                             "(L_rate = λ_mut L_mut + λ_cons L_cons). Lower (<1) to reduce "
                             "over-conservation; default 1.0 keeps prior behavior.")
    parser.add_argument("--mut-normalize", choices=["mean", "count"], default="mean",
                        help="How to average L_mut: 'mean' = entropy-weighted mean (default); "
                             "'count' = Σ(w·kl)/n_mut so hotspot weights raise mut mass.")
    parser.add_argument("--lambda-semi", type=float, default=0.0,
                        help="Weight on semigroup rate-composition regularizer L_semi "
                             "(§4.5). 0 disables; try 0.01–0.1 once bridge loss is stable.")
    parser.add_argument("--bridge-c",    type=float, default=1.0,
                        help="Reference resampling rate c in the conditional bridge target "
                             "(kappa = exp(-c(1-t))); larger = sharper terminal pull earlier")
    parser.add_argument(
        "--ablate-terminal-only",
        action="store_true",
        help="Appendix E.1: train with pure terminal CE target (force t→1 in "
             "conditional_bridge_kl; no bridge mixture at t<1). Needs a fresh ckpt.",
    )
    parser.add_argument(
        "--ablate-doob",
        action="store_true",
        help="Appendix E.1: match R0 without Doob h-transform "
             "(KL(softmax(R0)||R_θ); ignore x1). Mutually exclusive with "
             "--ablate-terminal-only. Needs a fresh ckpt.",
    )
    parser.add_argument(
        "--ablate-terminal-consistency",
        action="store_true",
        help="Appendix E.1: drop L_cons (sets --lambda-cons 0). "
             "Needs a fresh ckpt when comparing to full TreeSBM.",
    )
    parser.add_argument(
        "--ablate-mut-head",
        action="store_true",
        help="Appendix E.2: disable the mutation head (log R_θ = log R0; "
             "sets --lambda-mut 0). Branching / BL / stop heads still train. "
             "Needs a fresh ckpt.",
    )
    parser.add_argument(
        "--ablate-stop-head",
        action="store_true",
        help="Appendix E.2: disable the stop/termination head (constant "
             "p_stop=0.5; sets --lambda-stop 0). Needs a fresh ckpt.",
    )
    parser.add_argument(
        "--fitness-beta", "--ref-tilt-beta",
        type=float, default=0.0, dest="fitness_beta",
        help="§4.2 exponential tilt β on R0. "
             "log q_F = log_softmax(log_R0) + β·F; then re-softmax. "
             "0 = disabled (default, backward compatible). "
             "Try 1.0 for paper-faithful fitness weighting. Alias: --ref-tilt-beta.",
    )
    parser.add_argument(
        "--fitness-score",
        choices=["log_R0", "log_softmax"],
        default="log_R0",
        help="Per-AA fitness proxy for site_local tilting: "
             "'log_R0' (default) uses the stored ESM log-rates; "
             "'log_softmax' uses normalized site logprobs. Ignored for full_esm.",
    )
    parser.add_argument(
        "--fitness-tilt-mode",
        choices=[TILT_SITE_LOCAL, TILT_FULL_ESM],
        default=TILT_SITE_LOCAL,
        help="§4.2 tilt mode: 'site_local' (Option A, default, cheap) or "
             "'full_esm' (Option B: score each single-AA mutant with ESM PLL). "
             "full_esm loads a live R0 backend for scoring — expensive "
             "(~N·L·19 ESM forwards/step unless --fitness-esm-top-k is set).",
    )
    parser.add_argument(
        "--fitness-esm-batch-size",
        type=int,
        default=8,
        help="Batch size for full_esm mutant PLL scoring.",
    )
    parser.add_argument(
        "--fitness-esm-top-k",
        type=int,
        default=None,
        help="If set, full_esm only scores/tilts the top-k untilted AAs per site "
             "(big speedup). Remaining AAs keep untilted mass.",
    )
    parser.add_argument(
        "--r0-backend",
        default="esm2",
        help="Which frozen R0 mutation prior cache to load (paper Table 7 / D.1). "
             "Must match precompute --r0-backend. "
             "Default esm2 → legacy group_*_ref_rates.pt. "
             "Also: esm2_650m, esmc, jtt, wag, lg, neutral.",
    )
    parser.add_argument(
        "--ref-rates-tag",
        default=None,
        help="Override R0 cache filename tag (default derived from --r0-backend). "
             "Pass '' to force legacy group_*_ref_rates.pt.",
    )
    parser.add_argument("--per-site-pos-emb", action="store_true",
                        help="Add a learned positional embedding to the mutation head so "
                             "c_theta can act per-site (attacks the recovery ceiling). "
                             "Changes the architecture -> needs a fresh checkpoint.")
    parser.add_argument("--use-site-entropy", action="store_true",
                        help="Inject per-position Shannon entropy into the mutation head.")
    parser.add_argument("--deep-mut-head", action="store_true",
                        help="Widen mutation head (mut_in→128→64→20). Off by default so "
                             "existing checkpoints (incl. covid_v4_mutrec) still load.")
    parser.add_argument("--mut-aa-emb", action="store_true",
                        help="Concat current-AA embedding into RateHeads c_θ input "
                             "(flag-gated; changes mut_in → needs a fresh checkpoint).")
    parser.add_argument("--mut-aa-emb-dim", type=int, default=16,
                        help="Dimension of --mut-aa-emb (default 16).")
    parser.add_argument("--pssm-gate", action="store_true",
                        help="Blend Z(log R_θ) with train-alignment log-PSSM via per-site "
                             "w=σ(γ) (or --pssm-gate-fixed-w). Off by default; new ckpt.")
    parser.add_argument("--pssm-gate-fixed-w", type=float, default=None,
                        help="If set with --pssm-gate, use fixed w∈[0,1] instead of "
                             "learnable per-site gate logits.")
    parser.add_argument("--pssm-pseudocount", type=float, default=0.5,
                        help="Laplace pseudocount for train log-PSSM (--pssm-gate).")
    parser.add_argument("--use-entropy-loss-weighting", action="store_true",
                        help="Weight mutating-position bridge loss (L_mut) by per-position "
                             "entropy: floor + alpha*entropy (mutate freely at hotspots).")
    parser.add_argument("--use-entropy-cons-weighting", action="store_true",
                        help="Weight conserved-position bridge loss (L_cons) by INVERSE "
                             "entropy: floor + alpha*(1-entropy) (penalize mutating cold "
                             "sites hardest -> fights over-mutation / low retention).")
    parser.add_argument("--entropy-weight-alpha", type=float, default=1.0,
                        help="Slope for entropy-based mutation-loss weights.")
    parser.add_argument("--entropy-weight-alpha-cons", type=float, default=None,
                        help="Slope for L_cons entropy weights only. Default: same as "
                             "--entropy-weight-alpha. Lower to ease over-conservation.")
    parser.add_argument("--entropy-weight-floor", type=float, default=1.0,
                        help="Positive baseline added to entropy-based mutation-loss weights.")
    parser.add_argument("--entropy-is-normalized", action="store_true",
                        help="Treat provided entropy values as already normalized to [0,1].")
    parser.add_argument("--entropy-source", choices=["esm_self", "empirical"], default="esm_self",
                        help="Which per-position entropy feeds --use-site-entropy / "
                             "--use-entropy-loss-weighting. 'esm_self': H(softmax(log_R0)) "
                             "per position (free, self-consistent, = ESM uncertainty). "
                             "'empirical': Shannon entropy of each TRAIN-alignment column "
                             "(antigenic hotspots; computed once, saved in the checkpoint, "
                             "and reused at generation so train/inference never diverge).")
    parser.add_argument("--mut-hotspot-topk", type=int, default=None,
                        help="Hard hotspot: upweight the N highest-scoring TRAIN MSA "
                             "columns in L_mut (mutually exclusive with --mut-hotspot-frac). "
                             "Off by default. Score = --mut-hotspot-score.")
    parser.add_argument("--mut-hotspot-frac", type=float, default=None,
                        help="Hard hotspot: upweight the top fraction of columns by "
                             "--mut-hotspot-score in L_mut (e.g. 0.15 ≈ top-192 of L=1280). "
                             "Mutually exclusive with --mut-hotspot-topk.")
    parser.add_argument(
        "--mut-hotspot-score",
        choices=("entropy", "mut_freq"),
        default="entropy",
        help="How to rank MSA columns for hard hotspots (MSA-select → tree-apply). "
             "'entropy': TRAIN column Shannon entropy (requires --entropy-source empirical). "
             "'mut_freq': fraction of TRAIN MSA sequences ≠ column consensus/modal AA "
             "(no entropy-source requirement). Selected indices feed mut_hotspot_mask "
             "on the tree bridge loss.",
    )
    parser.add_argument(
        "--mut-hotspot-mask",
        type=str,
        default=None,
        help="Optional path to a precomputed bool [L] hotspot mask .pt "
             "(e.g. results/covid_mutfreq_vs_lit/mut_hotspot_mask_pmc_lit.pt). "
             "Overrides --mut-hotspot-score / topk / frac ranking when set; "
             "still requires --mut-hotspot-topk or --mut-hotspot-frac OR this flag alone "
             "to enable hotspot weighting (pass --mut-hotspot-weight).",
    )
    parser.add_argument("--mut-hotspot-weight", type=float, default=5.0,
                        help="Extra L_mut multiplier on hotspot columns (default 5.0). "
                             "Stacks with soft floor+alpha*H when --use-entropy-loss-weighting.")
    parser.add_argument("--mut-hotspot-force", action="store_true",
                        help="Also put hotspot columns into L_mut even when aa_t==x1 "
                             "this step (removed from cons_mask). Default off: only "
                             "boost mut_mask ∩ hotspot.")
    parser.add_argument(
        "--no-lit-hotspot-mask", "--no-mut-hotspot-mask",
        action="store_true",
        dest="no_lit_hotspot_mask",
        help="Table 8 train ablation: disable lit/PMC/MSA hotspot mask even if "
             "--mut-hotspot-mask / topk / frac were passed.",
    )
    parser.add_argument("--max-seq-len", type=int,   default=566)
    parser.add_argument("--patience",    type=int,   default=30,
                        help="Early stopping: stop if val loss doesn't improve for this many epochs")
    parser.add_argument("--n-t-samples", type=int,  default=4,
                        help="Number of t values sampled per tree per epoch")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from {ckpt-dir}/best.pt (weights, optimizer, scheduler, "
                             "epoch, best_val, patience_counter) instead of starting fresh. "
                             "Needed for long runs that may hit the SLURM time limit.")
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Load model weights only from this checkpoint (transfer learning). "
             "Does not restore optimizer/scheduler; incompatible layers are skipped.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze NodeEncoder + TreeEncoder (M3 adaptation). Only RateHeads "
             "(and PSSM gate inside RateHeads) receive gradients. Use with "
             "--init-checkpoint for low-data virus finetuning.",
    )
    args = parser.parse_args()

    if args.panviral_shared_head:
        # Shared RateHeads across genes: drop length-tied site index / PSSM gate /
        # lit masks; keep entropy + AA emb + shared mut/topology MLPs.
        if args.per_site_pos_emb:
            print("NOTE: --panviral-shared-head clears --per-site-pos-emb")
        args.per_site_pos_emb = False
        if args.pssm_gate:
            print("NOTE: --panviral-shared-head clears --pssm-gate")
        args.pssm_gate = False
        args.no_lit_hotspot_mask = True
        if not args.use_site_entropy:
            args.use_site_entropy = True
            print("NOTE: --panviral-shared-head enables --use-site-entropy")
        if not args.use_entropy_loss_weighting:
            args.use_entropy_loss_weighting = True
        if not args.use_entropy_cons_weighting:
            args.use_entropy_cons_weighting = True
        # Empirical per-gene column entropy unless user explicitly chose esm_self.
        if args.entropy_source == "esm_self":
            args.entropy_source = "empirical"
            print("NOTE: --panviral-shared-head sets --entropy-source empirical")
        if not args.mut_aa_emb:
            args.mut_aa_emb = True
            print("NOTE: --panviral-shared-head enables --mut-aa-emb")
        if not args.balanced_gene_sample and (
            args.gene_data or args.infer_gene_id
        ):
            args.balanced_gene_sample = True
            print("NOTE: --panviral-shared-head enables --balanced-gene-sample")
        print(
            "Panviral shared-head: pos_emb=OFF pssm_gate=OFF lit_mask=OFF "
            f"entropy={args.entropy_source} mut_aa_emb=ON"
        )

    if args.ablate_terminal_only and args.ablate_doob:
        raise SystemExit(
            "ERROR: --ablate-terminal-only and --ablate-doob are mutually exclusive"
        )
    if args.ablate_terminal_consistency:
        if args.lambda_cons != 0.0:
            print(
                f"--ablate-terminal-consistency: overriding lambda_cons "
                f"{args.lambda_cons} → 0.0"
            )
        args.lambda_cons = 0.0
    if args.ablate_mut_head:
        if args.lambda_mut != 0.0:
            print(f"--ablate-mut-head: overriding lambda_mut {args.lambda_mut} → 0.0")
        args.lambda_mut = 0.0
    if args.ablate_stop_head:
        if args.lambda_stop != 0.0:
            print(f"--ablate-stop-head: overriding lambda_stop {args.lambda_stop} → 0.0")
        args.lambda_stop = 0.0
    if args.ablate_terminal_only or args.ablate_doob or args.ablate_terminal_consistency:
        print(
            "Appendix E.1 train ablation: "
            f"terminal_only={args.ablate_terminal_only}  "
            f"no_doob={args.ablate_doob}  "
            f"lambda_cons={args.lambda_cons}"
        )
    if args.ablate_mut_head or args.ablate_stop_head:
        print(
            "Appendix E.2 train ablation: "
            f"no_mut_head={args.ablate_mut_head}  "
            f"no_stop_head={args.ablate_stop_head}  "
            f"lambda_mut={args.lambda_mut}  lambda_stop={args.lambda_stop}"
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if args.fitness_beta != 0.0:
        print(
            f"§4.2 fitness tilt: β={args.fitness_beta}  mode={args.fitness_tilt_mode}  "
            f"score={args.fitness_score}  (log R_θ = log R0_tilted + c_θ)"
        )
        if args.fitness_tilt_mode == TILT_FULL_ESM:
            print(
                "  WARNING: full_esm scores up to N·L·19 mutants per bridge step "
                "(batched). Prefer --fitness-esm-top-k 5 for trainable cost; "
                "site_local remains the default for existing recipes."
            )

    from src.r0_backends import cache_tag_for_backend, normalize_backend_name, build_r0_backend

    r0_backend = normalize_backend_name(args.r0_backend)
    ref_rates_tag = cache_tag_for_backend(r0_backend, override=args.ref_rates_tag)
    print(f"R0 backend: {r0_backend}  ref_rates_tag={ref_rates_tag!r}")

    # Option B scorer: live R0 backend (same family as cache) for mutant PLL.
    fitness_scorer = None
    fitness_cache: dict = {}
    fitness_r0_live = None
    if args.fitness_tilt_mode == TILT_FULL_ESM and args.fitness_beta != 0.0:
        print(f"Loading live R0 backend for full_esm fitness scoring ({r0_backend}) …")
        fitness_r0_live = build_r0_backend(r0_backend, device=device)
        fitness_scorer = make_sequence_pll_scorer(
            fitness_r0_live, max_seq_len=args.max_seq_len
        )
    # ── data: gene-tagged multi-source OR pre-split dirs OR random subtype split
    gene_train = _parse_gene_entries(args.gene_data)
    gene_val = _parse_gene_entries(args.gene_val_data)
    gene_test = _parse_gene_entries(args.gene_test_data)

    def _mk_ds(path: str, gene_id: str | None = None) -> TreeDataset:
        return TreeDataset(
            path,
            max_seq_len=args.max_seq_len,
            ref_rates_tag=ref_rates_tag,
            gene_id=gene_id,
            infer_gene_id=bool(args.infer_gene_id) and gene_id is None,
        )

    if gene_train:
        print(f"Multi-gene train sources: {gene_train}")
        train_ds = ConcatGeneTreeDataset(
            [_mk_ds(p, gid) for gid, p in gene_train]
        )
        if gene_val:
            val_ds = ConcatGeneTreeDataset(
                [_mk_ds(p, gid) for gid, p in gene_val]
            )
        elif args.val_data:
            val_ds = _mk_ds(args.val_data, gene_id=None)
        else:
            raise SystemExit("Multi-gene train requires --gene-val-data or --val-data")
        if gene_test:
            test_ds = ConcatGeneTreeDataset(
                [_mk_ds(p, gid) for gid, p in gene_test]
            )
        elif args.test_data:
            test_ds = _mk_ds(args.test_data, gene_id=None)
        else:
            raise SystemExit("Multi-gene train requires --gene-test-data or --test-data")
        dataset = train_ds
        Path(args.ckpt_dir).mkdir(exist_ok=True)
        print(f"Total — Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    elif args.val_data and args.test_data:
        print("Pre-split data: loading train/val/test from separate dirs")
        train_ds = _mk_ds(args.data, gene_id=None)
        val_ds = _mk_ds(args.val_data, gene_id=None)
        test_ds = _mk_ds(args.test_data, gene_id=None)
        dataset = train_ds  # used for the PLM-cache probe / export below
        Path(args.ckpt_dir).mkdir(exist_ok=True)
        print(f"Total — Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    else:
        dataset = TreeDataset(
            args.data, max_seq_len=args.max_seq_len, ref_rates_tag=ref_rates_tag,
            infer_gene_id=bool(args.infer_gene_id),
        )

        def subtype_of(group: int) -> str:
            if   1   <= group <= 48:  return "h3n2"
            elif 49  <= group <= 55:  return "swine"
            elif group == 56:         return "avian"
            elif 57  <= group <= 106: return "h1n1_ha"
            elif 107 <= group <= 156: return "h1n1_na"
            elif 157 <= group <= 206: return "h1n1_2015"
            elif 207 <= group <= 238: return "fluB_yam"
            elif 239 <= group <= 284: return "fluB_vic"
            return "unknown"

        bins: dict[str, list[int]] = defaultdict(list)
        for i in range(len(dataset)):
            bins[subtype_of(dataset.groups[i])].append(i)

        rng = random.Random(args.seed)
        train_idx, val_idx, test_idx = [], [], []
        for subtype in sorted(bins):
            indices = bins[subtype]
            rng.shuffle(indices)
            n = len(indices)
            if n < 3:
                train_idx += indices
                print(f"  {subtype}: {n} tree(s) → all train (too small to split)")
                continue
            n_t  = max(1, int(n * args.test_frac))
            n_v  = max(1, int(n * args.val_frac))
            n_tr = n - n_t - n_v
            train_idx += indices[:n_tr]
            val_idx   += indices[n_tr:n_tr + n_v]
            test_idx  += indices[n_tr + n_v:]
            print(f"  {subtype}: {n} trees → {n_tr} train / {n_v} val / {n_t} test")

        train_ds = Subset(dataset, train_idx)
        val_ds   = Subset(dataset, val_idx)
        test_ds  = Subset(dataset, test_idx)
        print(f"Total — Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

        # save split indices so the held-out test set is always recoverable
        split_path = Path(args.ckpt_dir) / "split_indices.json"
        split_path.parent.mkdir(exist_ok=True)
        with open(split_path, "w") as f:
            json.dump({"train": train_ds.indices, "val": val_ds.indices, "test": test_ds.indices}, f)

    # batch_size=1 (trees vary in node count — no collation)
    if args.balanced_gene_sample:
        gene_ids = _gene_ids_for_dataset(train_ds)
        n_unique = len(set(gene_ids))
        print(f"Balanced gene sampler over {n_unique} genes: {sorted(set(gene_ids))}")
        sampler = BalancedGeneSampler(
            gene_ids, num_samples=len(train_ds), seed=args.seed
        )
        train_loader = DataLoader(
            train_ds, batch_size=1, sampler=sampler, collate_fn=lambda x: x[0]
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, collate_fn=lambda x: x[0]
        )
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    # ── models 
    # Only instantiate ESM2 if PLM caches are missing (fallback)
    first_batch = dataset[0]
    if first_batch.get("plm_embeddings") is None:
        print("WARNING: No PLM cache found — run scripts/precompute_plm.py for faster training")
        embedder = ESM2Embedder(device=device)
    else:
        embedder = None
        print("PLM embeddings cached — skipping ESM2 at training time")

    node_enc   = NodeEncoder(d_plm=320, d_struct=3, d_laplacian=8, d_node=128).to(device)
    tree_enc   = TreeEncoder(d_model=128, n_layers=4, n_heads=8, dropout=0.1).to(device)
    rate_heads = RateHeads(
        d_model=128, max_seq_len=args.max_seq_len,
        use_pos_emb=args.per_site_pos_emb,
        use_site_entropy=args.use_site_entropy,
        deep_mut_head=args.deep_mut_head,
        use_mut_aa_emb=args.mut_aa_emb,
        d_aa=args.mut_aa_emb_dim,
        use_pssm_gate=args.pssm_gate,
        pssm_gate_fixed_w=args.pssm_gate_fixed_w,
    ).to(device)
    rate_heads.ablate_mut_head = bool(args.ablate_mut_head)
    rate_heads.ablate_stop_head = bool(args.ablate_stop_head)

    # Hard hotspots: MSA-select (entropy / mut_freq / precomputed mask) → tree-apply.
    if args.no_lit_hotspot_mask:
        if args.mut_hotspot_mask or args.mut_hotspot_topk or args.mut_hotspot_frac:
            print("NOTE: --no-lit-hotspot-mask: clearing mut hotspot mask/topk/frac")
        args.mut_hotspot_mask = None
        args.mut_hotspot_topk = None
        args.mut_hotspot_frac = None
    hotspot_requested = (
        args.mut_hotspot_topk is not None
        or args.mut_hotspot_frac is not None
        or args.mut_hotspot_mask is not None
    )
    if args.mut_hotspot_topk is not None and args.mut_hotspot_frac is not None:
        raise SystemExit("Pass only one of --mut-hotspot-topk / --mut-hotspot-frac")
    if args.mut_hotspot_mask is not None and (
        args.mut_hotspot_topk is not None or args.mut_hotspot_frac is not None
    ):
        print(
            "NOTE: --mut-hotspot-mask set; ignoring --mut-hotspot-topk/--mut-hotspot-frac ranking"
        )
    if (
        hotspot_requested
        and args.mut_hotspot_mask is None
        and args.mut_hotspot_score == "entropy"
        and args.entropy_source != "empirical"
    ):
        raise SystemExit(
            "--mut-hotspot-score entropy requires --entropy-source empirical "
            "(column entropy from the TRAIN alignment)."
        )
    if args.mut_hotspot_force and not hotspot_requested:
        raise SystemExit(
            "--mut-hotspot-force requires --mut-hotspot-topk, --mut-hotspot-frac, "
            "or --mut-hotspot-mask"
        )

    col_entropy = None
    mut_hotspot_mask = None
    need_entropy = args.entropy_source == "empirical" and (
        args.use_site_entropy
        or args.use_entropy_loss_weighting
        or args.use_entropy_cons_weighting
        or (
            hotspot_requested
            and args.mut_hotspot_mask is None
            and args.mut_hotspot_score == "entropy"
        )
    )
    if need_entropy:
        gene_ids = sorted(set(_gene_ids_for_dataset(train_ds)))
        if len(gene_ids) > 1:
            print(
                "Computing per-gene empirical column entropy "
                f"(genes={gene_ids}) — never pool non-homologous columns..."
            )
            col_entropy = {}
            for gid in gene_ids:
                sub = _subset_by_gene(train_ds, gid)
                if len(sub) == 0:
                    continue
                ce = compute_empirical_column_entropy(sub, args.max_seq_len).to(device)
                col_entropy[gid] = ce
                print(
                    f"  [{gid}] col_entropy[{ce.numel()}] mean={ce.mean():.3f} "
                    f"max={ce.max():.3f}"
                )
        else:
            print("Computing empirical column entropy from the training alignment...")
            col_entropy = compute_empirical_column_entropy(train_ds, args.max_seq_len).to(device)
            print(f"  col_entropy: [{col_entropy.numel()}]  mean={col_entropy.mean():.3f}  "
                  f"max={col_entropy.max():.3f}  nonzero={(col_entropy > 0).sum().item()}")

    if hotspot_requested:
        if args.mut_hotspot_mask is not None:
            blob = torch.load(args.mut_hotspot_mask, map_location="cpu", weights_only=False)
            if isinstance(blob, dict) and "mut_hotspot_mask" in blob:
                mut_hotspot_mask = blob["mut_hotspot_mask"].bool().cpu()
                score_name = blob.get("score", "precomputed")
            else:
                mut_hotspot_mask = torch.as_tensor(blob).bool().cpu()
                score_name = "precomputed"
            if mut_hotspot_mask.ndim != 1 or mut_hotspot_mask.numel() != args.max_seq_len:
                raise SystemExit(
                    f"--mut-hotspot-mask length {tuple(mut_hotspot_mask.shape)} "
                    f"!= max_seq_len={args.max_seq_len}"
                )
            print(
                f"  loaded hotspot mask from {args.mut_hotspot_mask} "
                f"(score={score_name})"
            )
        else:
            if args.mut_hotspot_score == "mut_freq":
                print(
                    "Computing TRAIN MSA column mut-freq "
                    "(frac ≠ consensus/modal AA) for hard hotspots..."
                )
                hotspot_scores = compute_msa_column_mut_freq(train_ds, args.max_seq_len)
                print(
                    f"  msa_mut_freq: [{hotspot_scores.numel()}]  "
                    f"mean={hotspot_scores.mean():.4f}  max={hotspot_scores.max():.4f}  "
                    f"nonzero={(hotspot_scores > 0).sum().item()}"
                )
                score_name = "mut_freq"
            else:
                hotspot_scores = col_entropy.detach().cpu()
                score_name = "entropy"
            mut_hotspot_mask = select_mut_hotspots(
                hotspot_scores,
                topk=args.mut_hotspot_topk,
                frac=args.mut_hotspot_frac,
            )
        mut_hotspot_mask = mut_hotspot_mask.to(device)
        n_hot = int(mut_hotspot_mask.sum().item())
        print(
            f"  mut hotspots ({score_name}): n={n_hot}/{mut_hotspot_mask.numel()}  "
            f"weight={args.mut_hotspot_weight}  force={args.mut_hotspot_force}  "
            f"(topk={args.mut_hotspot_topk} frac={args.mut_hotspot_frac} "
            f"mask={args.mut_hotspot_mask})"
        )

    log_pssm = None
    if args.pssm_gate:
        print("Computing empirical train log-PSSM for --pssm-gate...")
        log_pssm = compute_empirical_log_pssm(
            train_ds, args.max_seq_len, pseudocount=args.pssm_pseudocount
        ).to(device)
        print(f"  log_pssm: {tuple(log_pssm.shape)}  mean={log_pssm.mean():.3f}")

    params = (
        list(node_enc.parameters()) +
        list(tree_enc.parameters()) +
        list(rate_heads.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)
    best_val = float("inf")
    patience_counter = 0
    start_epoch = 1

    resume_path = ckpt_dir / "best.pt"
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        node_enc.load_state_dict(ckpt["node_enc"])
        tree_enc.load_state_dict(ckpt["tree_enc"])
        rate_heads.load_state_dict(ckpt["rate_heads"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        best_val = ckpt["val_loss"]
        patience_counter = ckpt.get("patience_counter", 0)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {resume_path}: epoch {ckpt['epoch']}, "
              f"best_val={best_val:.4f}, patience_counter={patience_counter}")
    elif args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_file():
            raise SystemExit(f"ERROR: --init-checkpoint not found: {init_path}")
        ckpt = torch.load(init_path, map_location=device, weights_only=False)

        def _load_tl(module, key: str) -> None:
            state = ckpt.get(key)
            if state is None:
                print(f"  init-checkpoint missing key {key!r} — skip")
                return
            missing, unexpected = module.load_state_dict(state, strict=False)
            if missing:
                print(f"  {key}: missing {len(missing)} keys (expected for seq-len / arch diffs)")
            if unexpected:
                print(f"  {key}: unexpected {len(unexpected)} keys")

        print(f"Transfer init from {init_path} (weights only, fresh optimizer)")
        _load_tl(node_enc, "node_enc")
        _load_tl(tree_enc, "tree_enc")
        _load_tl(rate_heads, "rate_heads")

    if args.freeze_encoder:
        if args.resume:
            raise SystemExit(
                "ERROR: --freeze-encoder is incompatible with --resume "
                "(optimizer would hold frozen params). Use --init-checkpoint instead."
            )
        n_frozen = 0
        for p in list(node_enc.parameters()) + list(tree_enc.parameters()):
            p.requires_grad = False
            n_frozen += p.numel()
        params = [p for p in rate_heads.parameters() if p.requires_grad]
        if not params:
            raise SystemExit("ERROR: --freeze-encoder left no trainable RateHeads params")
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        n_train = sum(p.numel() for p in params)
        print(
            f"--freeze-encoder: froze {n_frozen:,} encoder params; "
            f"training {n_train:,} RateHeads params"
        )
    # ── training loop
    for epoch in range(start_epoch, args.epochs + 1):
        node_enc.train(); tree_enc.train(); rate_heads.train()
        train_loss = 0.0
        n_steps = 0
        loss_breakdown = {"L_rate": 0.0, "L_mut": 0.0, "L_cons": 0.0, "L_top": 0.0, "L_br": 0.0, "L_stop": 0.0, "L_pll": 0.0, "L_semi": 0.0,
                          "mean_mut_entropy": 0.0, "mean_mut_weight": 0.0, "max_mut_weight": 0.0,
                          "L_br_pred_std": 0.0, "L_br_target_std": 0.0}

        for batch in train_loader:
            for _ in range(args.n_t_samples):
                t = random.uniform(0.0, args.t_max)
                optimizer.zero_grad()

                losses, n_active = forward_bridge_step(
                    batch, t, node_enc, tree_enc, rate_heads, device,
                    max_seq_len=args.max_seq_len,
                    lambda_top=args.lambda_top,
                    lambda_br=args.lambda_br,
                    lambda_stop=args.lambda_stop,
                    lambda_pll=args.lambda_pll,
                    lambda_mut=args.lambda_mut,
                    lambda_cons=args.lambda_cons,
                    lambda_semi=args.lambda_semi,
                    t_max=args.t_max,
                    bridge_c=args.bridge_c,
                    use_site_entropy=args.use_site_entropy,
                    use_entropy_loss_weighting=args.use_entropy_loss_weighting,
                    use_entropy_cons_weighting=args.use_entropy_cons_weighting,
                    entropy_weight_alpha=args.entropy_weight_alpha,
                    entropy_weight_alpha_cons=args.entropy_weight_alpha_cons,
                    entropy_weight_floor=args.entropy_weight_floor,
                    entropy_is_normalized=args.entropy_is_normalized,
                    mut_normalize=args.mut_normalize,
                    col_entropy=col_entropy,
                    mut_hotspot_mask=mut_hotspot_mask,
                    mut_hotspot_weight=args.mut_hotspot_weight if hotspot_requested else 1.0,
                    mut_hotspot_force=args.mut_hotspot_force,
                    log_pssm=log_pssm,
                    embedder=embedder,
                    fitness_beta=args.fitness_beta,
                    fitness_score=args.fitness_score,
                    fitness_tilt_mode=args.fitness_tilt_mode,
                    fitness_scorer=fitness_scorer,
                    fitness_cache=fitness_cache,
                    fitness_esm_batch_size=args.fitness_esm_batch_size,
                    fitness_esm_top_k=args.fitness_esm_top_k,
                    ablate_terminal_only=args.ablate_terminal_only,
                    ablate_doob=args.ablate_doob,
                )
                if losses is None or n_active == 0:
                    continue
                if torch.isnan(losses["total"]):
                    continue

                losses["total"].backward()
                for p in params:
                    if p.grad is not None:
                        p.grad.nan_to_num_(0.0, 0.0, 0.0)
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                train_loss += losses["total"].item()
                for k in loss_breakdown:
                    loss_breakdown[k] += losses[k].item()
                n_steps += 1

        if n_steps > 0:
            train_loss /= n_steps
            for k in loss_breakdown:
                loss_breakdown[k] /= n_steps

        # ── validation (fixed t=0.5 for reproducibility) 
        node_enc.eval(); tree_enc.eval(); rate_heads.eval()
        val_loss = 0.0
        n_val_steps = 0
        n_val_nan = 0
        with torch.no_grad():
            for batch in val_loader:
                losses, n_active = forward_bridge_step(
                    batch, t=0.5,
                    node_enc=node_enc, tree_enc=tree_enc, rate_heads=rate_heads,
                    device=device, max_seq_len=args.max_seq_len,
                    lambda_top=args.lambda_top, lambda_br=args.lambda_br,
                    lambda_stop=args.lambda_stop, lambda_pll=args.lambda_pll,
                    lambda_mut=args.lambda_mut, lambda_cons=args.lambda_cons,
                    lambda_semi=args.lambda_semi,
                    t_max=args.t_max, bridge_c=args.bridge_c,
                    use_site_entropy=args.use_site_entropy,
                    use_entropy_loss_weighting=args.use_entropy_loss_weighting,
                    use_entropy_cons_weighting=args.use_entropy_cons_weighting,
                    entropy_weight_alpha=args.entropy_weight_alpha,
                    entropy_weight_alpha_cons=args.entropy_weight_alpha_cons,
                    entropy_weight_floor=args.entropy_weight_floor,
                    entropy_is_normalized=args.entropy_is_normalized,
                    mut_normalize=args.mut_normalize,
                    col_entropy=col_entropy,
                    mut_hotspot_mask=mut_hotspot_mask,
                    mut_hotspot_weight=args.mut_hotspot_weight if hotspot_requested else 1.0,
                    mut_hotspot_force=args.mut_hotspot_force,
                    log_pssm=log_pssm,
                    embedder=embedder,
                    fitness_beta=args.fitness_beta,
                    fitness_score=args.fitness_score,
                    fitness_tilt_mode=args.fitness_tilt_mode,
                    fitness_scorer=fitness_scorer,
                    fitness_cache=fitness_cache,
                    fitness_esm_batch_size=args.fitness_esm_batch_size,
                    fitness_esm_top_k=args.fitness_esm_top_k,
                    ablate_terminal_only=args.ablate_terminal_only,
                    ablate_doob=args.ablate_doob,
                )
                if losses is None or n_active == 0:
                    continue
                total_v = losses["total"].item()
                if not math.isfinite(total_v):
                    n_val_nan += 1
                    continue
                val_loss += total_v
                n_val_steps += 1

        if n_val_steps > 0:
            val_loss /= n_val_steps
        else:
            # Avoid val=nan (never improves → no best.pt). Treat as +inf.
            val_loss = float("inf")
            if epoch == 1:
                print(
                    f"WARNING: no finite val steps "
                    f"(skipped_nan={n_val_nan}); check anc_aa / gap fill",
                    flush=True,
                )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:03d}  "
            f"train={train_loss:.4f} "
            f"(rate={loss_breakdown['L_rate']:.3f} "
            f"mut={loss_breakdown['L_mut']:.3f} "
            f"cons={loss_breakdown['L_cons']:.3f} "
            f"top={loss_breakdown['L_top']:.3f} "
            f"br={loss_breakdown['L_br']:.6f} "
            f"stop={loss_breakdown['L_stop']:.3f} "
            f"pll={loss_breakdown['L_pll']:.3f} "
            f"semi={loss_breakdown['L_semi']:.3f})  "
            f"val={val_loss:.4f}  lr={lr:.2e}  "
            f"[mut_entropy={loss_breakdown['mean_mut_entropy']:.3f} "
            f"mut_w={loss_breakdown['mean_mut_weight']:.3f} "
            f"mut_w_max={loss_breakdown['max_mut_weight']:.3f}]  "
            f"[br_pred_std={loss_breakdown['L_br_pred_std']:.6f} "
            f"br_target_std={loss_breakdown['L_br_target_std']:.6f}]"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "node_enc": node_enc.state_dict(),
                "tree_enc": tree_enc.state_dict(),
                "rate_heads": rate_heads.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "patience_counter": patience_counter,
                "val_loss": val_loss,
                "config": {
                    "use_pos_emb": args.per_site_pos_emb,
                    "use_site_entropy": args.use_site_entropy,
                    "deep_mut_head": args.deep_mut_head,
                    "use_mut_aa_emb": args.mut_aa_emb,
                    "mut_aa_emb_dim": args.mut_aa_emb_dim,
                    "use_pssm_gate": args.pssm_gate,
                    "pssm_gate_fixed_w": args.pssm_gate_fixed_w,
                    "use_entropy_loss_weighting": args.use_entropy_loss_weighting,
                    "use_entropy_cons_weighting": args.use_entropy_cons_weighting,
                    "entropy_weight_alpha": args.entropy_weight_alpha,
                    "entropy_weight_alpha_cons": args.entropy_weight_alpha_cons,
                    "entropy_weight_floor": args.entropy_weight_floor,
                    "entropy_is_normalized": args.entropy_is_normalized,
                    "entropy_source": args.entropy_source,
                    "mut_hotspot_topk": args.mut_hotspot_topk,
                    "mut_hotspot_frac": args.mut_hotspot_frac,
                    "mut_hotspot_score": args.mut_hotspot_score if hotspot_requested else None,
                    "mut_hotspot_mask_path": args.mut_hotspot_mask,
                    "mut_hotspot_weight": args.mut_hotspot_weight if hotspot_requested else 1.0,
                    "mut_hotspot_force": args.mut_hotspot_force,
                    "lambda_mut": args.lambda_mut,
                    "lambda_cons": args.lambda_cons,
                    "mut_normalize": args.mut_normalize,
                    "fitness_beta": args.fitness_beta,
                    "fitness_score": args.fitness_score,
                    "fitness_tilt_mode": args.fitness_tilt_mode,
                    "fitness_esm_batch_size": args.fitness_esm_batch_size,
                    "fitness_esm_top_k": args.fitness_esm_top_k,
                    "r0_backend": r0_backend,
                    "ref_rates_tag": ref_rates_tag,
                    "freeze_encoder": bool(args.freeze_encoder),
                    "ablate_terminal_only": args.ablate_terminal_only,
                    "ablate_doob": args.ablate_doob,
                    "ablate_terminal_consistency": args.ablate_terminal_consistency,
                    "ablate_mut_head": args.ablate_mut_head,
                    "ablate_stop_head": args.ablate_stop_head,
                },
                # empirical column-entropy vector [L] (None for esm_self), so
                # generation reuses the exact same signal training saw.
                "col_entropy": _serialize_col_entropy(col_entropy),
                "mut_hotspot_mask": (
                    mut_hotspot_mask.cpu() if mut_hotspot_mask is not None else None
                ),
                # train log-PSSM [L, 20] for --pssm-gate (None when gate off).
                "log_pssm": log_pssm.cpu() if log_pssm is not None else None,
            }, ckpt_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"\nBest val loss: {best_val:.4f}  -> {ckpt_dir}/best.pt")

    # ── test evaluation on best checkpoint ──
    best_path = ckpt_dir / "best.pt"
    if not best_path.is_file():
        raise SystemExit(
            f"ERROR: {best_path} was never written (val never improved). "
            "Usually val=nan/inf from all-gap internal seqs — see fill_missing_node_seqs."
        )
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    node_enc.load_state_dict(ckpt["node_enc"])
    tree_enc.load_state_dict(ckpt["tree_enc"])
    rate_heads.load_state_dict(ckpt["rate_heads"])
    node_enc.eval(); tree_enc.eval(); rate_heads.eval()
    test_loss = 0.0
    n_test_steps = 0
    with torch.no_grad():
        for batch in test_loader:
            losses, n_active = forward_bridge_step(
                batch, t=0.5,
                node_enc=node_enc, tree_enc=tree_enc, rate_heads=rate_heads,
                device=device, max_seq_len=args.max_seq_len,
                lambda_top=args.lambda_top, lambda_br=args.lambda_br,
                lambda_stop=args.lambda_stop, lambda_pll=args.lambda_pll,
                lambda_mut=args.lambda_mut, lambda_cons=args.lambda_cons,
                lambda_semi=args.lambda_semi,
                t_max=args.t_max, bridge_c=args.bridge_c,
                use_site_entropy=args.use_site_entropy,
                use_entropy_loss_weighting=args.use_entropy_loss_weighting,
                use_entropy_cons_weighting=args.use_entropy_cons_weighting,
                entropy_weight_alpha=args.entropy_weight_alpha,
                entropy_weight_alpha_cons=args.entropy_weight_alpha_cons,
                entropy_weight_floor=args.entropy_weight_floor,
                entropy_is_normalized=args.entropy_is_normalized,
                mut_normalize=args.mut_normalize,
                col_entropy=col_entropy,
                mut_hotspot_mask=mut_hotspot_mask,
                mut_hotspot_weight=args.mut_hotspot_weight if hotspot_requested else 1.0,
                mut_hotspot_force=args.mut_hotspot_force,
                log_pssm=log_pssm,
                embedder=embedder,
                fitness_beta=args.fitness_beta,
                fitness_score=args.fitness_score,
                fitness_tilt_mode=args.fitness_tilt_mode,
                fitness_scorer=fitness_scorer,
                fitness_cache=fitness_cache,
                fitness_esm_batch_size=args.fitness_esm_batch_size,
                fitness_esm_top_k=args.fitness_esm_top_k,
                ablate_terminal_only=args.ablate_terminal_only,
                ablate_doob=args.ablate_doob,
            )
            if losses is None or n_active == 0:
                continue
            test_loss += losses["total"].item()
            n_test_steps += 1
    if n_test_steps > 0:
        test_loss /= n_test_steps
    print(f"Test  loss: {test_loss:.4f}  ({n_test_steps} trees)")

    if fitness_r0_live is not None:
        fitness_r0_live.close()

    # export embeddings
    print("\nExporting embeddings with trained weights")
    export_embeddings(dataset, node_enc, tree_enc, device, Path(args.data))
    print("Done.")


def export_embeddings(
    dataset: TreeDataset,
    node_enc: NodeEncoder,
    tree_enc: TreeEncoder,
    device: str,
    data_dir: Path,
):

    node_enc.eval()
    tree_enc.eval()

    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            g = batch["group"]
            out_path = data_dir / f"group_{g:03d}_trained_emb.pt"

            node_ids      = batch["node_ids"]
            node_times_t  = batch["node_times"]
            edges         = batch["edges"]
            branch_lengths = batch["branch_lengths"]
            plm_T1        = batch["plm_embeddings"].to(device)  # [N, 320]

            node_times_dict = {nid: node_times_t[i].item() for i, nid in enumerate(node_ids)}
            root_id = node_ids[batch["root_index"]]

            has_children = {p for p, c in edges}
            tree_T1 = TreeState(
                node_ids=node_ids, root_id=root_id,
                edges=edges, branch_lengths=branch_lengths,
                node_seqs=batch["seqs"],
                active_leaves=[nid for nid in node_ids if nid not in has_children],
            )
            n2i = {nid: j for j, nid in enumerate(node_ids)}

            struct = compute_structural_features(tree_T1, n2i).to(device)
            lap    = compute_laplacian_pe(tree_T1, n2i, 8, device=device)

            # NodeEncoder: fuse PLM + structural + Laplacian → [N, 128]
            node_emb = node_enc(plm_T1, struct, lap)

            # TreeEncoder at t=1.0 (full tree, no bridge sampling)
            edge_index, _, edge_attr = build_edges(tree_T1, n2i)
            edge_index = edge_index.to(device)
            branch_lens = edge_attr.squeeze(-1).to(device)

            ctx_emb, _ = tree_enc(
                node_emb, node_ids, node_times_dict,
                edge_index, branch_lens, t_scalar=1.0,
            )

            torch.save({
                "node_ids":  node_ids,
                "plm":       plm_T1.cpu(),       # [N, 320] raw ESM2
                "node_emb":  node_emb.cpu(),     # [N, 128] NodeEncoder (PLM+struct+lap)
                "ctx_emb":   ctx_emb.cpu(),      # [N, 128] TreeEncoder contextual
            }, out_path)
            print(f"  group_{g:03d}: plm={tuple(plm_T1.shape)}  "
                  f"node_emb={tuple(node_emb.shape)}  ctx_emb={tuple(ctx_emb.shape)}")


if __name__ == "__main__":
    main()
