#!/usr/bin/env python3
"""Prepare a temporally split H3N2 HA dataset.

Default cutoffs: train ≤2022, val=2023, test=2024. Writes groups under
``--out-base`` (default ``data/h3n2``) plus ``SPLIT_PROTOCOL.json``.

Example::

    python scripts/prepare_h3n2_temporal.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from Bio import SeqIO
from scripts.split_fasta_by_date import split_fasta_by_date

# Source windows relative to repo root. Each entry: (path, year_lo, year_hi inclusive).
# year_lo/hi describe collection years present in that file (used for cutoff assignment).
SOURCE_WINDOWS: list[tuple[str, int, int]] = [
    ("data/train/h3n2_train/1974_2013_h3n2_HA.fasta", 1974, 2013),
    ("data/train/h3n2_train/2014_2016_h3n2_ha.fasta", 2014, 2016),
    ("data/train/h3n2_train/2016_2017_h3n2_ha.fasta", 2016, 2017),
    ("data/train/h3n2_train/2017_2018_h3n2_ha.fasta", 2017, 2018),
    ("data/train/h3n2_train/2018_2019_h3n2_ha.fasta", 2018, 2019),
    ("data/train/h3n2_train/2019_2020_h3n2_ha.fasta", 2019, 2020),
    ("data/train/h3n2_train/2020_2022_h3n2_ha.fasta", 2020, 2022),
    ("data/train/h3n2_train/2022_h3n2_ha.fasta", 2022, 2022),
    ("data/train/h3n2_train/2023_h3n2_ha.fasta", 2023, 2023),
    ("data/validate/h3n2_val/2023_h3n2_ha.fasta", 2023, 2023),
    ("data/test/h3n2_ha_test/h3n2_ha_early_2024.fasta", 2024, 2024),
    ("data/test/h3n2_ha_test/h3n2_ha_2024.fasta", 2024, 2024),
]


def _id(rec) -> str:
    # headers: >EPI_ISL_XXXXXX,YYYY-MM-DD
    return rec.description.split(",", 1)[0].strip()


def _year(rec) -> int | None:
    parts = rec.description.split(",", 1)
    if len(parts) < 2:
        return None
    try:
        return int(parts[1].strip()[:4])
    except ValueError:
        return None


def assign_split(
    year: int,
    train_end: int,
    val_year: int | None,
    test_start: int,
    test_end: int,
) -> str | None:
    """Map a collection year to train/val/test, or None if outside all bands."""
    if year <= train_end:
        return "train"
    if val_year is not None and year == val_year:
        return "val"
    if test_start <= year <= test_end:
        return "test"
    return None


def load_by_cutoffs(
    train_end: int,
    val_year: int | None,
    test_start: int,
    test_end: int,
    min_year: int | None,
) -> tuple[dict[str, list], dict]:
    """
    Load all source windows, filter by per-record year, assign to splits.
    Dedup by EPI_ISL id with priority train < val < test (earlier split keeps id).
    """
    pools: dict[str, list] = {"train": [], "val": [], "test": []}
    seen: set[str] = set()
    window_map: list[dict] = []
    stats = {s: {"kept": 0, "dup": 0, "out_of_band": 0} for s in pools}

    for wp, y_lo, y_hi in SOURCE_WINDOWS:
        path = ROOT / wp
        entry = {
            "path": wp,
            "file_year_lo": y_lo,
            "file_year_hi": y_hi,
            "exists": path.exists(),
            "assigned_years": {},
        }
        if not path.exists():
            print(f"  WARNING: missing {wp}")
            window_map.append(entry)
            continue

        year_counts: dict[str, int] = {}
        for rec in SeqIO.parse(path, "fasta"):
            y = _year(rec)
            if y is None:
                continue
            if min_year is not None and y < min_year:
                continue
            split = assign_split(y, train_end, val_year, test_start, test_end)
            if split is None:
                stats["train"]["out_of_band"] += 1  # account once
                continue
            rid = _id(rec)
            if rid in seen:
                stats[split]["dup"] += 1
                continue
            seen.add(rid)
            pools[split].append(rec)
            stats[split]["kept"] += 1
            year_counts[f"{split}:{y}"] = year_counts.get(f"{split}:{y}", 0) + 1
        entry["assigned_years"] = year_counts
        window_map.append(entry)
        print(f"  {wp}: {year_counts or '(no seqs in cutoff bands)'}")

    for split, s in stats.items():
        print(f"[{split}] kept={s['kept']} dups_skipped={s['dup']}")
    return pools, {"windows": window_map, "stats": stats}


def write_pool(records, out_fasta: Path):
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fasta, "w") as f:
        for rec in records:
            f.write(f">{rec.description}\n{str(rec.seq)}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Temporal H3N2 HA split for flu-season forecasting evals."
    )
    ap.add_argument("--out-base", default="data/h3n2",
                    help="Output root (use data/h3n2_temporal_forecast for alternate cutoffs)")
    ap.add_argument("--group-size", type=int, default=400)
    ap.add_argument("--min-year", type=int, default=2014,
                    help="Drop sequences before this year (default 2014; set 1974 to include older)")
    ap.add_argument("--train-end-year", type=int, default=2022,
                    help="Train includes collection years <= this (default 2022)")
    ap.add_argument("--val-year", type=int, default=2023,
                    help="Validation year (exact). Pass --no-val to disable.")
    ap.add_argument("--no-val", action="store_true",
                    help="Disable val band (years between train and test are dropped)")
    ap.add_argument("--test-start-year", type=int, default=2024)
    ap.add_argument("--test-end-year", type=int, default=2024,
                    help="Inclusive test end year (use 2025 for multi-season horizon)")
    ap.add_argument("--skip-group-split", action="store_true",
                    help="Only write pooled FASTAs; skip date grouping")
    args = ap.parse_args()

    val_year = None if args.no_val else args.val_year
    if val_year is not None and not (args.train_end_year < val_year < args.test_start_year):
        # Allow val == test_start - 1 typically; warn on overlap/inversion
        if val_year <= args.train_end_year or val_year >= args.test_start_year:
            print(
                f"WARNING: val_year={val_year} is not strictly between "
                f"train_end={args.train_end_year} and test_start={args.test_start_year}"
            )
    if args.test_start_year > args.test_end_year:
        ap.error("--test-start-year must be <= --test-end-year")

    base = ROOT / args.out_base
    print(
        f"Cutoffs: train≤{args.train_end_year} | "
        f"val={val_year} | test={args.test_start_year}–{args.test_end_year} | "
        f"min_year={args.min_year}"
    )
    print("Season mapping: each date-contiguous group of "
          f"{args.group_size} seqs is a tree; groups stay inside one year-band "
          "(train / val / test). Flu 'next season' ≈ test year = train_end+2 "
          "with val = train_end+1; 'next few seasons' widens --test-end-year.")

    pools, meta = load_by_cutoffs(
        args.train_end_year, val_year, args.test_start_year, args.test_end_year,
        args.min_year,
    )

    protocol = {
        "dataset": "h3n2",
        "split_type": "temporal",
        "out_base": args.out_base,
        "train_end_year": args.train_end_year,
        "val_year": val_year,
        "test_start_year": args.test_start_year,
        "test_end_year": args.test_end_year,
        "min_year": args.min_year,
        "group_size": args.group_size,
        "group_to_season": (
            "Groups are date-ordered chunks within a year-band. "
            "Northern-hemisphere flu seasons roughly span Oct(Y-1)–Sep(Y); "
            "cutoff years here are collection-year cutoffs, not season labels."
        ),
        "windows": meta["windows"],
        "counts": {s: len(pools[s]) for s in ("train", "val", "test")},
    }
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "SPLIT_PROTOCOL.json", "w") as f:
        json.dump(protocol, f, indent=2)
    print(f"Wrote {base / 'SPLIT_PROTOCOL.json'}")

    splits = {
        "train": ("h3n2train", pools["train"]),
        "val": ("h3n2val", pools["val"]),
        "test": ("h3n2test", pools["test"]),
    }
    for split, (prefix, recs) in splits.items():
        if not recs:
            print(f"\n=== {split}: EMPTY (check cutoffs / source windows) ===")
            continue
        out_dir = base / split
        out_dir.mkdir(parents=True, exist_ok=True)
        # clear stale group files
        for old in out_dir.glob(f"{prefix}_group_*"):
            old.unlink()
        pool = out_dir / f"{prefix}.fasta"
        write_pool(recs, pool)
        if args.skip_group_split:
            print(f"\n=== {split}: {len(recs)} seqs -> {pool} (no grouping) ===")
            continue
        print(f"\n=== splitting {split}: {len(recs)} seqs -> groups of {args.group_size} ===")
        split_fasta_by_date(str(pool), args.group_size, str(out_dir))

    print("\nDone. Next: run_all_groups.py --data-dir on each non-empty split dir.")
    print("Next: run_all_groups → precompute_plm/ref_rates → train.py")
    print(f"  # or edit prefixes/dirs for {args.out_base}")


if __name__ == "__main__":
    main()
