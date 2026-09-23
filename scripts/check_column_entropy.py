#!/usr/bin/env python3
"""Compare column Shannon entropy on mutating vs conserved alignment sites."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import TreeDataset


def _load_compute_empirical_column_entropy():
    """Import the exact train.py implementation (no duplication drift)."""
    path = ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("_treesbm_train_entropy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_empirical_column_entropy


def mutating_mask_root_descendant(dataset: TreeDataset, max_seq_len: int) -> np.ndarray:
    """True where any train tree has root ≠ some descendant at that column."""
    mut = np.zeros(max_seq_len, dtype=bool)
    for i in range(len(dataset)):
        batch = dataset[i]
        node_ids = batch["node_ids"]
        root_id = node_ids[batch["root_index"]]
        root_seq = batch["seqs"][root_id][:max_seq_len]
        L = min(len(root_seq), max_seq_len)
        for nid in node_ids:
            if nid == root_id:
                continue
            seq = batch["seqs"][nid][:L]
            for j in range(min(L, len(seq))):
                a, b = root_seq[j], seq[j]
                if a in ("-", "X", "*") or b in ("-", "X", "*"):
                    continue
                if a != b:
                    mut[j] = True
    return mut


def mutating_mask_mut_freq(dataset: TreeDataset, max_seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Parent→child change counts / edge counts per column; mutating iff count > 0."""
    changes = np.zeros(max_seq_len, dtype=np.int64)
    edges = np.zeros(max_seq_len, dtype=np.int64)
    for i in range(len(dataset)):
        batch = dataset[i]
        seqs = batch["seqs"]
        for parent, child in batch["edges"]:
            p = seqs[parent][:max_seq_len]
            c = seqs[child][:max_seq_len]
            L = min(len(p), len(c), max_seq_len)
            for j in range(L):
                a, b = p[j], c[j]
                if a in ("-", "X", "*") or b in ("-", "X", "*"):
                    continue
                edges[j] += 1
                if a != b:
                    changes[j] += 1
    freq = changes / np.maximum(edges, 1)
    return changes > 0, freq


def ascii_hist(values: np.ndarray, bins: int = 20, width: int = 40) -> str:
    if values.size == 0:
        return "(empty)"
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    peak = max(int(counts.max()), 1)
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(width * c / peak))
        lines.append(f"  [{edges[i]:.2f},{edges[i+1]:.2f}) {c:5d} {bar}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Train split directory (TreeDataset layout)")
    ap.add_argument("--max-seq-len", type=int, default=566)
    ap.add_argument(
        "--label-mode",
        choices=("root_descendant", "mut_freq", "mut_freq_median"),
        default="root_descendant",
        help="How to split mutating vs conserved sites. "
             "mut_freq_median = high vs low empirical parent→child mut frequency (median split).",
    )
    ap.add_argument("--margin", type=float, default=0.0, help="Require mean_mut > mean_cons + margin")
    ap.add_argument("--out-dir", default=None, help="Optional directory for JSON/CSV outputs")
    ap.add_argument("--hist-bins", type=int, default=20)
    ap.add_argument(
        "--also-freq-median",
        action="store_true",
        help="Also report high vs low mut-frequency median split (recommended when nearly all sites ever mutate)",
    )
    args = ap.parse_args()

    compute_empirical_column_entropy = _load_compute_empirical_column_entropy()

    dataset = TreeDataset(args.data, max_seq_len=args.max_seq_len)
    if len(dataset) == 0:
        print(f"ERROR: no trees found under {args.data}")
        return 2

    col_entropy = compute_empirical_column_entropy(dataset, args.max_seq_len).numpy()
    _, mut_freq = mutating_mask_mut_freq(dataset, args.max_seq_len)

    max_L = 0
    for i in range(len(dataset)):
        for seq in dataset[i]["seqs"].values():
            max_L = max(max_L, min(len(seq), args.max_seq_len))
    active = np.zeros(args.max_seq_len, dtype=bool)
    active[:max_L] = True

    def summarize(mut_mask: np.ndarray, label: str) -> dict:
        mut_mask = mut_mask & active
        cons_mask = (~mut_mask) & active
        ent_mut = col_entropy[mut_mask]
        ent_cons = col_entropy[cons_mask]
        mean_mut = float(ent_mut.mean()) if ent_mut.size else float("nan")
        mean_cons = float(ent_cons.mean()) if ent_cons.size else float("nan")
        std_mut = float(ent_mut.std()) if ent_mut.size else float("nan")
        std_cons = float(ent_cons.std()) if ent_cons.size else float("nan")
        median_mut = float(np.median(ent_mut)) if ent_mut.size else float("nan")
        median_cons = float(np.median(ent_cons)) if ent_cons.size else float("nan")
        passed = (
            ent_mut.size > 0
            and ent_cons.size > 0
            and mean_mut > mean_cons + args.margin
        )
        print("-" * 60)
        print(f"Split: {label}")
        print(f"n_mutating / high:  {int(mut_mask.sum())}")
        print(f"n_conserved / low:  {int(cons_mask.sum())}")
        print(f"mean entropy (mut/high):  {mean_mut:.6f}  ± {std_mut:.6f}  (median {median_mut:.6f})")
        print(f"mean entropy (cons/low):  {mean_cons:.6f}  ± {std_cons:.6f}  (median {median_cons:.6f})")
        print(f"difference:               {mean_mut - mean_cons:.6f}")
        print(f"pass (>{args.margin} margin): {'PASS' if passed else 'FAIL'}")
        print()
        print("Histogram — mut/high:")
        print(ascii_hist(ent_mut, bins=args.hist_bins))
        print()
        print("Histogram — cons/low:")
        print(ascii_hist(ent_cons, bins=args.hist_bins))
        if int(cons_mask.sum()) < 5:
            print("\nWARNING: very few conserved/low sites; consider --label-mode mut_freq_median")
        return {
            "label": label,
            "n_mutating": int(mut_mask.sum()),
            "n_conserved": int(cons_mask.sum()),
            "mean_entropy_mutating": mean_mut,
            "std_entropy_mutating": std_mut,
            "median_entropy_mutating": median_mut,
            "mean_entropy_conserved": mean_cons,
            "std_entropy_conserved": std_cons,
            "median_entropy_conserved": median_cons,
            "diff_mut_minus_cons": mean_mut - mean_cons,
            "passed": passed,
            "mut_mask": mut_mask,
        }

    print("=" * 60)
    print("Column entropy mut vs conserved sanity check")
    print("=" * 60)
    print(f"data:            {args.data}")
    print(f"n_trees:         {len(dataset)}")
    print(f"max_seq_len:     {args.max_seq_len} (active cols: {int(active.sum())})")
    print(f"entropy source:  train.py::compute_empirical_column_entropy (log20-normalized)")
    print()

    if args.label_mode == "root_descendant":
        primary_mask = mutating_mask_root_descendant(dataset, args.max_seq_len)
        primary_label = "root≠descendant anywhere (mutating) vs always conserved"
    elif args.label_mode == "mut_freq":
        primary_mask = mut_freq > 0
        primary_label = "parent→child mut_freq > 0 vs == 0"
    else:
        med = float(np.median(mut_freq[active]))
        primary_mask = mut_freq > med
        primary_label = f"mut_freq > median ({med:.6g}) vs ≤ median"

    primary = summarize(primary_mask, primary_label)
    splits = [primary]

    # Auto secondary when binary ever-mutated split is nearly one-sided.
    want_freq = args.also_freq_median or (
        args.label_mode != "mut_freq_median"
        and (primary["n_conserved"] < 5 or primary["n_mutating"] < 5)
    )
    if want_freq:
        med = float(np.median(mut_freq[active]))
        secondary = summarize(
            mut_freq > med,
            f"secondary: mut_freq > median ({med:.6g}) vs ≤ median",
        )
        splits.append(secondary)

    # Primary pass/fail drives exit code; secondary is informational unless primary mode is median.
    passed = primary["passed"]
    if args.label_mode != "mut_freq_median" and want_freq and len(splits) > 1:
        # Require primary OR (if primary conserved set tiny) secondary.
        if primary["n_conserved"] < 5 or primary["n_mutating"] < 5:
            passed = splits[1]["passed"]
            print(f"\nNOTE: primary split nearly one-sided; exit code uses secondary median split → "
                  f"{'PASS' if passed else 'FAIL'}")

    print()
    print(f"OVERALL RESULT: {'PASS' if passed else 'FAIL'}")

    payload = {
        "data": args.data,
        "n_trees": len(dataset),
        "max_seq_len": args.max_seq_len,
        "active_cols": int(active.sum()),
        "label_mode": args.label_mode,
        "margin": args.margin,
        "passed": passed,
        "splits": [{k: v for k, v in s.items() if k != "mut_mask"} for s in splits],
    }

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "column_entropy_sanity.json").write_text(json.dumps(payload, indent=2))
        mut_mask = primary["mut_mask"]
        with open(out / "column_entropy_by_site.csv", "w") as f:
            f.write("position,entropy,label,mut_freq\n")
            for j in range(args.max_seq_len):
                if not active[j]:
                    continue
                lab = "mutating" if mut_mask[j] else "conserved"
                f.write(f"{j},{col_entropy[j]:.8f},{lab},{float(mut_freq[j]):.8f}\n")
        print(f"\nWrote {out / 'column_entropy_sanity.json'}")
        print(f"Wrote {out / 'column_entropy_by_site.csv'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
