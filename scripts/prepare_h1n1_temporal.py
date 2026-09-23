#!/usr/bin/env python3
"""Prepare a temporally split H1N1 HA dataset.

Default cutoffs: train ≤2023, val=2024, test=2025. Output: ``data/h1n1_temporal``.

Example::

    python scripts/prepare_h1n1_temporal.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from Bio import SeqIO

from scripts.prepare_h1n1_geo import (
    RAW_DIR,
    normalize_location,
    parse_date,
    parse_header,
)

LOC_MIN = 100
GROUP_SIZE = 300
MIN_GROUP = 50
MAX_SPAN_YEARS = 4


def assign_split(
    year: int,
    train_end: int,
    val_year: int | None,
    test_start: int,
    test_end: int,
) -> str | None:
    if year <= train_end:
        return "train"
    if val_year is not None and year == val_year:
        return "val"
    if test_start <= year <= test_end:
        return "test"
    return None


def _emit(chunk, out_dir: Path, prefix: str, g: int, min_group: int) -> int:
    if len(chunk) < min_group:
        return g
    with open(out_dir / f"{prefix}_group_{g:03d}.fasta", "w") as ff, \
         open(out_dir / f"{prefix}_group_{g:03d}.csv", "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["name", "date"])
        for acc, date_str, _, seq in chunk:
            ff.write(f">{acc},{date_str}\n{seq}\n")
            w.writerow([acc, date_str])
    return g + 1


def write_groups(records, out_dir: Path, prefix: str, start_group: int,
                 group_size: int, min_group: int, max_span_years: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    g = start_group
    chunk = []
    for rec in records:
        year = rec[2][0]
        if chunk and (len(chunk) >= group_size or (year - chunk[0][2][0]) > max_span_years):
            g = _emit(chunk, out_dir, prefix, g, min_group)
            chunk = []
        chunk.append(rec)
    g = _emit(chunk, out_dir, prefix, g, min_group)
    return g


def main():
    ap = argparse.ArgumentParser(
        description="Temporal H1N1 HA split for flu-season forecasting."
    )
    ap.add_argument("--out-base", default="data/h1n1_temporal")
    ap.add_argument("--loc-min", type=int, default=LOC_MIN)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--min-group", type=int, default=MIN_GROUP)
    ap.add_argument("--max-span-years", type=int, default=MAX_SPAN_YEARS)
    ap.add_argument("--min-year", type=int, default=2009,
                    help="Drop pre-pandemic / sparse years (default 2009)")
    ap.add_argument("--train-end-year", type=int, default=2023)
    ap.add_argument("--val-year", type=int, default=2024)
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--test-start-year", type=int, default=2025)
    ap.add_argument("--test-end-year", type=int, default=2025)
    args = ap.parse_args()

    val_year = None if args.no_val else args.val_year
    if args.test_start_year > args.test_end_year:
        ap.error("--test-start-year must be <= --test-end-year")

    raw_dir = ROOT / RAW_DIR
    base = ROOT / args.out_base
    raw_files = sorted(raw_dir.glob("*_h1n1.fasta"))
    if not raw_files:
        print(f"No *_h1n1.fasta in {raw_dir}"); return

    print(
        f"Cutoffs: train≤{args.train_end_year} | val={val_year} | "
        f"test={args.test_start_year}–{args.test_end_year} | min_year={args.min_year}"
    )

    # pass 1: load + year-split
    by_split: dict[str, list] = {"train": [], "val": [], "test": []}
    seen: set[str] = set()
    dup = bad = dropped_year = 0
    year_hist: dict[str, dict[int, int]] = {s: {} for s in by_split}

    for rf in raw_files:
        for rec in SeqIO.parse(rf, "fasta"):
            try:
                acc, loc, date_raw, country = parse_header(rec.description)
            except (IndexError, ValueError):
                bad += 1
                continue
            if acc in seen:
                dup += 1
                continue
            try:
                date_str, sort_key = parse_date(date_raw)
            except (ValueError, IndexError):
                bad += 1
                continue
            y = sort_key[0]
            if y < args.min_year:
                dropped_year += 1
                continue
            split = assign_split(
                y, args.train_end_year, val_year,
                args.test_start_year, args.test_end_year,
            )
            if split is None:
                dropped_year += 1
                continue
            seen.add(acc)
            by_split[split].append((acc, loc, country, date_str, sort_key, str(rec.seq)))
            year_hist[split][y] = year_hist[split].get(y, 0) + 1

    print(f"loaded {sum(len(v) for v in by_split.values())} seqs "
          f"({dup} dup, {bad} bad, {dropped_year} year-filtered) from {len(raw_files)} files")
    for split in ("train", "val", "test"):
        print(f"  [{split}] {len(by_split[split])} seqs years={dict(sorted(year_hist[split].items()))}")

    prefix_map = {"train": "h1n1ttrain", "val": "h1n1tval", "test": "h1n1ttest"}
    g_counter = {"train": 1, "val": 1, "test": 1}
    group_meta = []

    for split in ("train", "val", "test"):
        out_dir = base / split
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = prefix_map[split]
        for old in out_dir.glob(f"{prefix}_group_*"):
            old.unlink()

        # location counts within this split only
        loc_count: dict[str, int] = {}
        for acc, loc, country, date_str, sort_key, seq in by_split[split]:
            loc_count[loc] = loc_count.get(loc, 0) + 1

        unit_records: dict[str, list] = {}
        for acc, loc, country, date_str, sort_key, seq in by_split[split]:
            unit = loc if loc_count[loc] >= args.loc_min else country
            unit_records.setdefault(unit, []).append((acc, date_str, sort_key, seq))

        for unit in sorted(unit_records, key=lambda u: (-len(unit_records[u]), u)):
            recs = sorted(unit_records[unit], key=lambda r: r[2])
            g_before = g_counter[split]
            g_counter[split] = write_groups(
                recs, out_dir, prefix, g_counter[split],
                args.group_size, args.min_group, args.max_span_years,
            )
            if g_counter[split] > g_before:
                years = sorted({r[2][0] for r in recs})
                group_meta.append({
                    "split": split, "unit": unit,
                    "groups": list(range(g_before, g_counter[split])),
                    "n_seqs": len(recs), "years": years,
                })

        n_groups = g_counter[split] - 1
        print(f"=== {split}: {n_groups} groups -> {out_dir} ===")

    protocol = {
        "dataset": "h1n1",
        "split_type": "temporal",
        "out_base": args.out_base,
        "train_end_year": args.train_end_year,
        "val_year": val_year,
        "test_start_year": args.test_start_year,
        "test_end_year": args.test_end_year,
        "min_year": args.min_year,
        "group_size": args.group_size,
        "max_span_years": args.max_span_years,
        "group_to_season": (
            "Each group is a date-contiguous location/country chunk within one "
            "year-band. With max_span_years=4 a tree may span a few seasons but "
            "never crosses the train/val/test year cutoffs. "
            "Next-season forecast: train≤2023, val=2024, test=2025."
        ),
        "year_hist": {s: dict(sorted(year_hist[s].items())) for s in year_hist},
        "counts": {s: len(by_split[s]) for s in by_split},
        "n_groups": {s: g_counter[s] - 1 for s in g_counter},
        "units": group_meta,
    }
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "SPLIT_PROTOCOL.json", "w") as f:
        json.dump(protocol, f, indent=2)
    print(f"Wrote {base / 'SPLIT_PROTOCOL.json'}")
    print("\nNext: run_all_groups.py --data-dir data/h1n1_temporal/{split} "
          "--prefix h1n1t{split}")
    print("Next: run_all_groups → precompute_plm/ref_rates → train.py")


if __name__ == "__main__":
    main()
