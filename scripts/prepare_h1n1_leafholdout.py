#!/usr/bin/env python3
"""Prepare an H1N1 HA random leaf-holdout split.

Holds out a fraction of leaves per tree for recovery eval.
Output: ``data/h1n1_leafholdout``.
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from Bio import SeqIO

from scripts.prepare_h1n1_geo import parse_header, parse_date, RAW_DIR

LOC_MIN = 100
GROUP_SIZE = 300
MIN_GROUP = 50
MAX_SPAN_YEARS = 4
HOLDOUT_FRAC = 0.15
MIN_HELDOUT = 2


def chunk_unit(records, group_size, min_chunk, max_span_years):
    """records: (acc, augur_date_str, sort_key, seq), single unit, date-ordered.
    Yields date-contiguous chunks, dropping any chunk smaller than min_chunk."""
    chunk = []
    for rec in records:
        year = rec[2][0]
        if chunk and (len(chunk) >= group_size or (year - chunk[0][2][0]) > max_span_years):
            if len(chunk) >= min_chunk:
                yield chunk
            chunk = []
        chunk.append(rec)
    if len(chunk) >= min_chunk:
        yield chunk


def write_group(fasta_path: Path, csv_path: Path, records):
    with open(fasta_path, "w") as ff, open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["name", "date"])
        for acc, date_str, _, seq in records:
            ff.write(f">{acc},{date_str}\n{seq}\n")
            w.writerow([acc, date_str])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", default="data/h1n1_leafholdout")
    ap.add_argument("--loc-min", type=int, default=LOC_MIN)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--min-group", type=int, default=MIN_GROUP,
                    help="Minimum leaves remaining for TREE-BUILDING after holdout is removed")
    ap.add_argument("--max-span-years", type=int, default=MAX_SPAN_YEARS)
    ap.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC,
                    help="Fraction of each tree's leaves pulled out before tree-building")
    ap.add_argument("--min-heldout", type=int, default=MIN_HELDOUT)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw_dir = ROOT / RAW_DIR
    base = ROOT / args.out_base
    raw_files = sorted(raw_dir.glob("*_h1n1.fasta"))
    if not raw_files:
        print(f"No *_h1n1.fasta in {raw_dir}"); return

    # ── pass 1: load all records, count fine-location frequencies (same as prepare_h1n1_geo)
    records = []
    loc_count: dict[str, int] = {}
    seen: set[str] = set()
    dup = bad = 0
    for rf in raw_files:
        for rec in SeqIO.parse(rf, "fasta"):
            try:
                acc, loc, date, country = parse_header(rec.description)
            except (IndexError, ValueError):
                bad += 1; continue
            if acc in seen:
                dup += 1; continue
            seen.add(acc)
            records.append((acc, loc, country, date, str(rec.seq)))
            loc_count[loc] = loc_count.get(loc, 0) + 1
    print(f"loaded {len(records)} seqs ({dup} dup, {bad} malformed) from {len(raw_files)} files")

    # ── pass 2: assign each seq to a grouping unit -- this is ONLY for building
    # coherent trees (fine loc if it has enough seqs, else fall back to country);
    # it has nothing to do with the train/val/test split below.
    unit_records: dict[str, list] = {}
    dropped_date = 0
    for acc, loc, country, date_raw, seq in records:
        unit = loc if loc_count[loc] >= args.loc_min else country
        try:
            date_str, sort_key = parse_date(date_raw)
        except (ValueError, IndexError):
            dropped_date += 1; continue
        unit_records.setdefault(unit, []).append((acc, date_str, sort_key, seq))
    print(f"{len(unit_records)} grouping units; {dropped_date} unparseable-date seqs dropped")

    # min chunk size so that AFTER holdout removal, tree-building side still has
    # >= min_group leaves (round-up of min_group / (1 - holdout_frac))
    min_chunk = max(
        args.min_group + args.min_heldout,
        math.ceil(args.min_group / (1 - args.holdout_frac)),
    )

    # ── build all candidate trees across all units, ignoring geography entirely
    all_chunks = []
    for unit, recs in unit_records.items():
        recs = sorted(recs, key=lambda r: r[2])
        for chunk in chunk_unit(recs, args.group_size, min_chunk, args.max_span_years):
            all_chunks.append(chunk)
    print(f"{len(all_chunks)} candidate trees built from {len(unit_records)} units "
          f"(min {min_chunk} leaves/tree so >= {args.min_group} remain after holdout)")

    # ── RANDOM (non-geographic) train/val/test split at the tree level
    rng.shuffle(all_chunks)
    n = len(all_chunks)
    n_test = max(1, int(n * args.test_frac))
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val - n_test
    split_chunks = {
        "train": all_chunks[:n_train],
        "val": all_chunks[n_train:n_train + n_val],
        "test": all_chunks[n_train + n_val:],
    }

    prefix_map = {"train": "h1n1lhtrain", "val": "h1n1lhval", "test": "h1n1lhtest"}
    held_dir = base / "heldout"
    held_dir.mkdir(parents=True, exist_ok=True)

    for split, chunks in split_chunks.items():
        out_dir = base / split
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = prefix_map[split]
        n_kept = n_held = 0
        for i, chunk in enumerate(chunks, start=1):
            recs = list(chunk)
            rng.shuffle(recs)
            n_hold = max(args.min_heldout, round(len(recs) * args.holdout_frac))
            heldout, keep = recs[:n_hold], recs[n_hold:]
            write_group(out_dir / f"{prefix}_group_{i:03d}.fasta",
                        out_dir / f"{prefix}_group_{i:03d}.csv", keep)
            write_group(held_dir / f"h1n1lh_{split}_group_{i:03d}_heldout.fasta",
                        held_dir / f"h1n1lh_{split}_group_{i:03d}_heldout.csv", heldout)
            n_kept += len(keep)
            n_held += len(heldout)
        print(f"=== {split}: {len(chunks)} trees -> {out_dir} "
              f"({n_kept} tree-building leaves, {n_held} held-out leaves in {held_dir}) ===")

    print("\nNext:")
    print("  scripts/run_all_groups.py --data-dir data/h1n1_leafholdout/{split} --prefix h1n1lh{split}")
    print("  scripts/precompute_plm.py / precompute_ref_rates.py --data data/h1n1_leafholdout/{split}")
    print("  scripts/train.py --data data/h1n1_leafholdout/train --val-data data/h1n1_leafholdout/val "
          "--test-data data/h1n1_leafholdout/test --ckpt-dir checkpoints/h1n1_leafholdout_v1")
    print("  scripts/eval_leaf_holdout.py --data data/h1n1_leafholdout "
          "--checkpoint checkpoints/h1n1_leafholdout_v1/best.pt")


if __name__ == "__main__":
    main()
