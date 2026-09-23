#!/usr/bin/env python3
"""Leaf-holdout / clade-holdout recovery eval.

Loads held-out FASTAs under ``heldout/`` and scores recovery of withheld leaves.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from Bio import SeqIO
from transformers import AutoTokenizer, EsmForMaskedLM

from src.dataset import TreeDataset
from src.treeencoder.plm_embeddings import ESM2Embedder
from scripts.eval_single_tree import (
    AA_VOCAB, load_models, generate_tree, seq_identity, positional_recovery, get_leaves,
)

SPLITS = ["train", "val", "test"]


def find_heldout_fasta(held_dir: Path, split: str, group_idx: int) -> Path | None:
    """
    Resolve held-out FASTA for a group. Supports both naming schemes:
      h1n1lh_{split}_group_{NNN}_heldout.fasta
      covidch_{split}_group_{NNN}_heldout.fasta
    and any other *_{split}_group_{NNN}_heldout.fasta.
    """
    pattern = f"*_{split}_group_{group_idx:03d}_heldout.fasta"
    matches = sorted(held_dir.glob(pattern))
    if matches:
        return matches[0]
    # legacy exact H1N1 name (same as pattern, kept for clarity)
    legacy = held_dir / f"h1n1lh_{split}_group_{group_idx:03d}_heldout.fasta"
    return legacy if legacy.exists() else None


def load_heldout(held_dir: Path, split: str, group_idx: int) -> list[str]:
    path = find_heldout_fasta(held_dir, split, group_idx)
    if path is None:
        return []
    return [str(rec.seq) for rec in SeqIO.parse(path, "fasta")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/h1n1_leafholdout_v1/best.pt")
    ap.add_argument("--data", default="data/h1n1_leafholdout",
                    help="Base dir with train/val/test/heldout "
                         "(h1n1_leafholdout or covid_cladeholdout)")
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--max-seq-len", type=int, default=566,
                    help="566 for flu HA; 1280 for COVID Spike")
    ap.add_argument("--branch-rate-scale", type=float, default=6.0)
    ap.add_argument("--max-leaves", type=int, default=200)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--max-trees", type=int, default=None,
                    help="Cap trees evaluated per split (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/eval_leaf_holdout_results.json")
    args = ap.parse_args()

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

    base = ROOT / args.data
    held_dir = base / "heldout"

    all_results = []
    for split in args.splits:
        data_dir = base / split
        if not data_dir.exists():
            print(f"skip {split}: {data_dir} not found"); continue
        dataset = TreeDataset(str(data_dir), max_seq_len=args.max_seq_len)
        groups = dataset.groups[:args.max_trees] if args.max_trees else dataset.groups

        for idx, g in enumerate(groups):
            batch = dataset[idx]
            root_id = batch["node_ids"][batch["root_index"]]
            root_seq = batch["seqs"][root_id]

            heldout_seqs = load_heldout(held_dir, split, g)
            if not heldout_seqs:
                print(f"  [{split} {g:03d}] no held-out leaves found, skipping")
                continue

            print(f"[{split} {g:03d}] root_len={len(root_seq)}  "
                  f"train_tree_nodes={len(batch['node_ids'])}  heldout_leaves={len(heldout_seqs)}")

            gen_tree = generate_tree(
                root_seq, args.n_steps, args.max_seq_len, args.branch_rate_scale,
                args.max_leaves, 1.0,
                node_enc, tree_enc, rate_heads, embedder,
                tokenizer, esm_model, aa_token_ids, device, col_entropy=col_entropy,
            )
            gen_leaves = get_leaves(gen_tree)
            if not gen_leaves:
                print(f"  [{split} {g:03d}] generation produced 0 leaves, skipping")
                continue

            mut_recs, cons_rets, best_ids = [], [], []
            for hs in heldout_seqs:
                best_id, best_gen = max(
                    (seq_identity(hs, gen_tree.node_seqs[gn]), gn) for gn in gen_leaves)
                best_ids.append(best_id)
                rec = positional_recovery(root_seq, hs, gen_tree.node_seqs[best_gen])
                if rec["mut_recovery"] == rec["mut_recovery"]:      # not NaN
                    mut_recs.append(rec["mut_recovery"])
                if rec["cons_retention"] == rec["cons_retention"]:  # not NaN
                    cons_rets.append(rec["cons_retention"])

            def _mean(v):
                return sum(v) / len(v) if v else float("nan")

            res = {
                "split": split, "group": g,
                "n_heldout": len(heldout_seqs), "n_gen_leaves": len(gen_leaves),
                "best_match_identity": _mean(best_ids),
                "mut_recovery": _mean(mut_recs),
                "cons_retention": _mean(cons_rets),
            }
            all_results.append(res)
            print(f"  best_match_identity={res['best_match_identity']:.4f}  "
                  f"mut_recovery={res['mut_recovery']:.4f}  "
                  f"cons_retention={res['cons_retention']:.4f}")

    if all_results:
        def agg(key):
            vals = [r[key] for r in all_results if r[key] == r[key]]
            return sum(vals) / len(vals) if vals else float("nan")

        print(f"\n{'='*60}")
        print(f"LEAF-HOLDOUT RECOVERY SUMMARY  ({len(all_results)} trees)")
        print(f"{'='*60}")
        print(f"Best-match identity (real held-out leaf -> nearest gen leaf): {agg('best_match_identity'):.4f}")
        print(f"Mutating-site recovery (root->true differs, model matched):  {agg('mut_recovery'):.4f}")
        print(f"Conserved-site retention (root->true same, model kept):      {agg('cons_retention'):.4f}")
    else:
        print("\nNo trees evaluated.")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
