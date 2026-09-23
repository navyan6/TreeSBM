#!/usr/bin/env python3
"""Held-out eval with EVE evolutionary-index scoring instead of EVEscape."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, EsmForMaskedLM

from src.dataset import TreeDataset
from src.tree_state import TreeState
from src.treeencoder.plm_embeddings import ESM2Embedder
from src.treeencoder.structural_features import compute_structural_features
from src.treeencoder.laplacian import compute_laplacian_pe
from src.treeencoder.edges import build_edges
from scripts.eval_single_tree import (
    load_models, generate_tree, positional_recovery, get_leaves, seq_identity,
    get_lm_logits, AA_VOCAB,
)

AA_TO_IDX = {a: i for i, a in enumerate(AA_VOCAB)}


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def load_eve_scores(path: str) -> torch.Tensor:
    blob = torch.load(path, map_location="cpu")
    scores = blob["scores"] if isinstance(blob, dict) else blob
    if scores.ndim != 2 or scores.shape[1] != 20:
        raise ValueError(f"Expected eve scores [L, 20], got {tuple(scores.shape)}")
    return scores


def mutation_eve(root_seq: str, leaf_seq: str, eve: torch.Tensor, L: int):
    """Per mutation (root->leaf, leaf_aa != root_aa), look up EVE[pos, leaf_aa].
    Skips positions with zero score (unfilled in the matrix). Returns (scores, n_muts)."""
    scores = []
    n_muts = 0
    for p in range(min(len(root_seq), len(leaf_seq), L)):
        r, g = root_seq[p], leaf_seq[p]
        if g != r and g in AA_TO_IDX:
            n_muts += 1
            s = eve[p, AA_TO_IDX[g]].item()
            if s != 0.0:
                scores.append(s)
    return scores, n_muts


def mutating_sites(root_seq: str, gt_seq: str, L: int) -> list[tuple[int, str]]:
    sites = []
    for p in range(min(len(root_seq), len(gt_seq), L)):
        if root_seq[p] != gt_seq[p] and gt_seq[p] in AA_TO_IDX:
            sites.append((p, gt_seq[p]))
    return sites


def eve_at_sites(sites: list[tuple[int, str]], eve: torch.Tensor) -> list[float]:
    out = []
    for p, aa in sites:
        s = eve[p, AA_TO_IDX[aa]].item()
        if s != 0.0:
            out.append(s)
    return out


def random_aa_baseline(root_seq: str, gt_seq: str, eve: torch.Tensor, L: int,
                       rng: random.Random) -> list[float]:
    """One random non-wt AA per GT mutating site."""
    scores = []
    for p in range(min(len(root_seq), len(gt_seq), L)):
        r = root_seq[p]
        if gt_seq[p] == r or r not in AA_TO_IDX:
            continue
        choices = [a for a in AA_VOCAB if a != r]
        aa = rng.choice(choices)
        s = eve[p, AA_TO_IDX[aa]].item()
        if s != 0.0:
            scores.append(s)
    return scores


def random_aa_mean_baseline(root_seq: str, gt_seq: str, eve: torch.Tensor, L: int) -> list[float]:
    """Mean EVE over all 19 non-wt AAs at each GT mutating site."""
    scores = []
    for p in range(min(len(root_seq), len(gt_seq), L)):
        r = root_seq[p]
        if gt_seq[p] == r or r not in AA_TO_IDX:
            continue
        vals = [eve[p, AA_TO_IDX[a]].item() for a in AA_VOCAB if a != r]
        vals = [v for v in vals if v != 0.0]
        if vals:
            scores.append(sum(vals) / len(vals))
    return scores


def recovered_gt_eve(root_seq: str, gt_seq: str, gen_seq: str, eve: torch.Tensor, L: int) -> list[float]:
    """EVE scores at GT mutating sites where the generated sequence matches GT."""
    scores = []
    for p in range(min(len(root_seq), len(gt_seq), len(gen_seq), L)):
        r, g, m = root_seq[p], gt_seq[p], gen_seq[p]
        if g != r and m == g and g in AA_TO_IDX:
            s = eve[p, AA_TO_IDX[g]].item()
            if s != 0.0:
                scores.append(s)
    return scores


def gt_leaves_of(batch: dict) -> list[str]:
    parents = {p for p, _ in batch["edges"]}
    return [nid for nid in batch["node_ids"] if nid not in parents]


@torch.no_grad()
def root_mutation_log_probs(tree: TreeState, root_id: str, max_seq_len: int,
                            node_enc, tree_enc, rate_heads, embedder,
                            tokenizer, esm_model, aa_token_ids, device,
                            col_entropy=None) -> torch.Tensor | None:
    """Forward pass at t=1 on the generated tree; log-softmax mutation probs at root [L, 20]."""
    try:
        node_ids_t = tree.node_ids
        node_to_idx = {nid: i for i, nid in enumerate(node_ids_t)}
        if root_id not in node_to_idx:
            return None
        root_idx = [node_to_idx[root_id]]
        node_times_dict = {nid: 1.0 for nid in node_ids_t}

        struct_t = compute_structural_features(tree, node_to_idx).to(device)
        lap_t = compute_laplacian_pe(tree, node_to_idx, 8, device=device)
        edge_index_t, _, edge_attr_t = build_edges(tree, node_to_idx)
        edge_index_t = edge_index_t.to(device)
        branch_lens_t = edge_attr_t.squeeze(-1).to(device)
        plm_t = embedder.embed_sequences(
            [tree.node_seqs[nid] for nid in node_ids_t]).to(device)

        root_seq = tree.node_seqs[root_id]
        log_R0_mut = get_lm_logits(tokenizer, esm_model, aa_token_ids,
                                   [root_seq], max_seq_len, device)
        h_t = node_enc(plm_t, struct_t, lap_t)
        H_t, _ = tree_enc(h_t, node_ids_t, node_times_dict,
                          edge_index_t, branch_lens_t, t_scalar=1.0)
        aa_indices = None
        if getattr(rate_heads, "use_mut_aa_emb", False):
            from src.bridge.losses import _build_seq_indices
            aa_indices = _build_seq_indices([root_seq], max_seq_len, device)
        log_pssm = getattr(rate_heads, "_train_log_pssm", None)
        out = rate_heads(
            H_t, root_idx, log_R0_mut,
            site_entropy=col_entropy,
            aa_indices=aa_indices,
            log_pssm=log_pssm,
        )
        return out["log_R_theta_mut"][0].log_softmax(-1).cpu()
    except Exception:
        return None


def rate_eve_correlation(root_seq: str, gt_seq: str, log_probs: torch.Tensor,
                         eve: torch.Tensor, L: int) -> dict | None:
    """Pearson / Spearman between model log-prob for GT AA and EVE at GT mutating sites."""
    eve_vals, rate_vals = [], []
    for p, aa in mutating_sites(root_seq, gt_seq, L):
        e = eve[p, AA_TO_IDX[aa]].item()
        if e == 0.0:
            continue
        eve_vals.append(e)
        rate_vals.append(log_probs[p, AA_TO_IDX[aa]].item())
    if len(eve_vals) < 3:
        return None
    pr = pearsonr(rate_vals, eve_vals)
    sr = spearmanr(rate_vals, eve_vals)
    return {
        "n_sites": len(eve_vals),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-seq-len", type=int, default=1280)
    ap.add_argument("--eve-scores", default=None,
                    help="[L,20] EVE index .pt; enables EVE baseline columns")
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--max-leaves", type=int, default=300)
    ap.add_argument("--branch-rate-scale", type=float, default=6.0)
    ap.add_argument("--mutation-rate-scale", type=float, default=0.3)
    ap.add_argument("--max-trees", type=int, default=20)
    ap.add_argument("--gt-leaves-sampled", type=int, default=30,
                    help="GT leaves per tree to match against gen for recovery")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    node_enc, tree_enc, rate_heads, col_entropy = load_models(
        args.checkpoint, device, args.max_seq_len)
    embedder = ESM2Embedder(device=device)
    mid = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(mid)
    esm_model = EsmForMaskedLM.from_pretrained(mid).to(device).eval()
    for p in esm_model.parameters():
        p.requires_grad = False
    aa_token_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(a) for a in AA_VOCAB], dtype=torch.long)

    eve = None
    if args.eve_scores:
        eve = load_eve_scores(args.eve_scores)
        nz = eve[eve != 0]
        print(f"EVE scores {tuple(eve.shape)}  nonzero={nz.numel()}  "
              f"range=[{nz.min():.4f}, {nz.max():.4f}]" if nz.numel() else
              f"EVE scores {tuple(eve.shape)}  (all zero — check prepare_eve_scores.py)")

    ds = TreeDataset(args.data, max_seq_len=args.max_seq_len)
    n = min(args.max_trees, len(ds)) if args.max_trees else len(ds)
    print(f"Evaluating {n} test trees  (mut_rate={args.mutation_rate_scale} n_steps={args.n_steps})\n")

    recs, rets, idents = [], [], []
    model_eve, gt_eve = [], []
    recovered_eve, random_eve, random_mean_eve = [], [], []
    tree_corrs = []
    per_tree = []
    rng = random.Random(args.seed)

    for i in range(n):
        batch = ds[i]
        root_id = batch["node_ids"][batch["root_index"]]
        root_seq = batch["seqs"][root_id]
        try:
            random.seed(args.seed + i)
            torch.manual_seed(args.seed + i)
            gen = generate_tree(
                root_seq, args.n_steps, args.max_seq_len, args.branch_rate_scale,
                args.max_leaves, args.mutation_rate_scale, node_enc, tree_enc, rate_heads,
                embedder, tokenizer, esm_model, aa_token_ids, device, col_entropy=col_entropy)
        except Exception as e:
            print(f"[{i+1}/{n}] ERROR: {e}")
            continue

        gen_leaves = get_leaves(gen)
        gen_seqs = [gen.node_seqs[g] for g in gen_leaves]
        gt_leaves = gt_leaves_of(batch)
        gt_sample = rng.sample(gt_leaves, min(args.gt_leaves_sampled, len(gt_leaves)))

        t_rec, t_ret, t_id = [], [], []
        tree_rec_eve, tree_gt_eve = [], []
        tree_recovered, tree_random, tree_random_mean = [], [], []
        tree_rate_corrs = []

        log_probs_root = None
        if eve is not None:
            log_probs_root = root_mutation_log_probs(
                gen, root_id, args.max_seq_len, node_enc, tree_enc, rate_heads,
                embedder, tokenizer, esm_model, aa_token_ids, device, col_entropy)

        for gl in gt_sample:
            gt_seq = batch["seqs"][gl]
            best, best_id = None, -1.0
            for gs in gen_seqs:
                idv = seq_identity(gt_seq, gs)
                if idv > best_id:
                    best_id, best = idv, gs
            if best is None:
                continue
            r = positional_recovery(root_seq, gt_seq, best)
            t_rec.append(r["mut_recovery"])
            t_ret.append(r["cons_retention"])
            t_id.append(best_id)

            if eve is not None:
                sites = mutating_sites(root_seq, gt_seq, args.max_seq_len)
                tree_gt_eve += eve_at_sites(sites, eve)
                tree_recovered += recovered_gt_eve(root_seq, gt_seq, best, eve, args.max_seq_len)
                tree_random += random_aa_baseline(root_seq, gt_seq, eve, args.max_seq_len, rng)
                tree_random_mean += random_aa_mean_baseline(root_seq, gt_seq, eve, args.max_seq_len)
                if log_probs_root is not None:
                    c = rate_eve_correlation(root_seq, gt_seq, log_probs_root, eve, args.max_seq_len)
                    if c is not None:
                        tree_rate_corrs.append(c)

        recs.append(_mean(t_rec))
        rets.append(_mean(t_ret))
        idents.append(_mean(t_id))

        if eve is not None:
            for gs in gen_seqs:
                sc, _ = mutation_eve(root_seq, gs, eve, args.max_seq_len)
                tree_rec_eve += sc
            model_eve += tree_rec_eve
            gt_eve += tree_gt_eve
            recovered_eve += tree_recovered
            random_eve += tree_random
            random_mean_eve += tree_random_mean
            if tree_rate_corrs:
                tree_corrs.append({
                    "pearson_r": _mean([c["pearson_r"] for c in tree_rate_corrs]),
                    "spearman_r": _mean([c["spearman_r"] for c in tree_rate_corrs]),
                    "n_pairs": sum(c["n_sites"] for c in tree_rate_corrs),
                })

        per_tree.append({
            "tree": i,
            "gen_leaves": len(gen_leaves),
            "mut_recovery": _mean(t_rec),
            "cons_retention": _mean(t_ret),
            "identity": _mean(t_id),
            "model_eve": _mean(tree_rec_eve) if eve is not None else None,
            "gt_eve": _mean(tree_gt_eve) if eve is not None else None,
            "recovered_gt_eve": _mean(tree_recovered) if eve is not None else None,
            "random_aa_eve": _mean(tree_random) if eve is not None else None,
            "rate_eve_corr": {
                "pearson_r": _mean([c["pearson_r"] for c in tree_rate_corrs]),
                "spearman_r": _mean([c["spearman_r"] for c in tree_rate_corrs]),
                "n_pairs": sum(c["n_sites"] for c in tree_rate_corrs),
            } if tree_rate_corrs else None,
        })
        msg = (f"[{i+1}/{n}] gen_leaves={len(gen_leaves):3d}  "
               f"recovery={_mean(t_rec):.4f}  retention={_mean(t_ret):.4f}  "
               f"identity={_mean(t_id):.4f}")
        if eve is not None:
            msg += f"  model_EVE={_mean(tree_rec_eve):.4f}  recovered_EVE={_mean(tree_recovered):.4f}"
        print(msg)

    print("\n" + "=" * 64)
    print(f"AGGREGATE ({len(recs)} trees)  checkpoint={args.checkpoint}")
    print("=" * 64)
    print(f"  mutation recovery : {_mean(recs):.4f}")
    print(f"  conserved retention: {_mean(rets):.4f}")
    print(f"  best-match identity: {_mean(idents):.4f}")

    summary = {
        "checkpoint": args.checkpoint,
        "n_trees": len(recs),
        "mut_recovery": _mean(recs),
        "cons_retention": _mean(rets),
        "identity": _mean(idents),
    }

    if eve is not None:
        m_ev = _mean(model_eve)
        g_ev = _mean(gt_eve)
        rec_ev = _mean(recovered_eve)
        rnd_ev = _mean(random_eve)
        rnd_mean_ev = _mean(random_mean_eve)
        print(f"\n  EVE scores (higher = more evolutionarily unlikely / deleterious):")
        print(f"    model mutations      : mean EVE = {m_ev:.4f}  ({len(model_eve)} muts)")
        print(f"    real GT mutations    : mean EVE = {g_ev:.4f}  ({len(gt_eve)} muts)")
        print(f"    recovered GT muts    : mean EVE = {rec_ev:.4f}  ({len(recovered_eve)} muts)")
        print(f"    random AA @ mut sites: mean EVE = {rnd_ev:.4f}  ({len(random_eve)} draws)")
        print(f"    random AA (mean 19)  : mean EVE = {rnd_mean_ev:.4f}  ({len(random_mean_eve)} sites)")
        print(f"    recovered - random   : {rec_ev - rnd_ev:+.4f}   (model vs GT: {m_ev - g_ev:+.4f})")

        rate_summary = None
        if tree_corrs:
            rate_summary = {
                "pearson_r": _mean([c["pearson_r"] for c in tree_corrs]),
                "spearman_r": _mean([c["spearman_r"] for c in tree_corrs]),
                "n_trees_with_corr": len(tree_corrs),
                "n_pairs_total": sum(c["n_pairs"] for c in tree_corrs),
            }
            print(f"\n  TreeSBM rate ↔ EVE at GT mutating sites (root log-probs, t=1):")
            print(f"    Pearson r  = {rate_summary['pearson_r']:.4f}")
            print(f"    Spearman r = {rate_summary['spearman_r']:.4f}  "
                  f"({rate_summary['n_pairs_total']} site pairs)")
        else:
            print("\n  TreeSBM rate ↔ EVE correlation: skipped (rates unavailable or <3 sites/tree)")

        summary.update({
            "model_eve": m_ev,
            "gt_eve": g_ev,
            "recovered_gt_eve": rec_ev,
            "random_aa_eve": rnd_ev,
            "random_aa_mean_eve": rnd_mean_ev,
            "recovered_minus_random_eve": rec_ev - rnd_ev,
            "model_minus_gt_eve": m_ev - g_ev,
            "rate_eve_correlation": rate_summary,
        })

    out = args.out or f"checkpoints/eval_eve_{Path(args.checkpoint).parent.name}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"summary": summary, "per_tree": per_tree}, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
