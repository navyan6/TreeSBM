#!/usr/bin/env python3
"""Prepare a temporally split COVID Spike dataset.

Requires Spike FASTAs from ``covid_extract_spike.py``. Default cutoffs:
train ≤2022, val=2023, test=2024–2025. Output: ``data/covid_temporal``.
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

from scripts.prepare_covid_geo import (
    RAW_DIR,
    MIN_GROUP,
    load_spike_records,
    parse_date,
)

GROUP_SIZE = 300


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


def write_groups(records, out_dir: Path, prefix: str, group_size: int,
                 start_group: int, min_group: int) -> int:
    """records: (acc, augur_date_str, sort_key, seq), already date-ordered."""
    out_dir.mkdir(parents=True, exist_ok=True)
    g = start_group
    for i in range(0, len(records), group_size):
        chunk = records[i:i + group_size]
        if len(chunk) < min_group:
            print(f"    dropping remainder of {len(chunk)} seqs (< min_group={min_group})")
            continue
        with open(out_dir / f"{prefix}_group_{g:03d}.fasta", "w") as f:
            for acc, date_str, _, seq in chunk:
                f.write(f">{acc},{date_str}\n{seq}\n")
        with open(out_dir / f"{prefix}_group_{g:03d}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "date"])
            for acc, date_str, _, seq in chunk:
                w.writerow([acc, date_str])
        g += 1
    return g


def main():
    ap = argparse.ArgumentParser(
        description="Temporal COVID Spike split for time-cutoff forecasting."
    )
    ap.add_argument("--out-base", default="data/covid_temporal")
    ap.add_argument("--raw-dir", default=RAW_DIR,
                    help="Dir with *_covid_seqs.fasta + matching *_spike.fasta")
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--min-group", type=int, default=MIN_GROUP)
    ap.add_argument("--train-end-year", type=int, default=2022)
    ap.add_argument("--val-year", type=int, default=2023)
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--test-start-year", type=int, default=2024)
    ap.add_argument("--test-end-year", type=int, default=2025)
    args = ap.parse_args()

    val_year = None if args.no_val else args.val_year
    if args.test_start_year > args.test_end_year:
        ap.error("--test-start-year must be <= --test-end-year")

    base = ROOT / args.out_base
    raw_dir = ROOT / args.raw_dir
    raw_sources = sorted(raw_dir.glob("*_covid_seqs.fasta"))
    if not raw_sources:
        print(f"No *_covid_seqs.fasta found in {raw_dir}"); return

    print(
        f"Cutoffs: train≤{args.train_end_year} | val={val_year} | "
        f"test={args.test_start_year}–{args.test_end_year}"
    )

    # by_split[split][country] -> list of (acc, date_str, sort_key, seq)
    by_split_country: dict[str, dict[str, list]] = {
        "train": {}, "val": {}, "test": {},
    }
    seen_acc: set[str] = set()
    year_hist: dict[str, dict[int, int]] = {s: {} for s in by_split_country}
    n_missing_spike = 0

    for raw_path in raw_sources:
        spike_path = raw_path.with_name(
            raw_path.stem.replace("_covid_seqs", "") + "_spike.fasta"
        )
        if not spike_path.exists():
            print(f"WARNING: missing {spike_path.name} -- run "
                  f"covid_extract_spike.py / slurm_covid_extract.sh first. Skipping.")
            n_missing_spike += 1
            continue
        recs = load_spike_records(raw_path, spike_path)
        dup = bad_date = dropped = 0
        for acc, date_raw, country, seq in recs:
            if acc in seen_acc:
                dup += 1
                continue
            try:
                date_str, sort_key = parse_date(date_raw)
            except (ValueError, IndexError):
                bad_date += 1
                continue
            y = sort_key[0]
            split = assign_split(
                y, args.train_end_year, val_year,
                args.test_start_year, args.test_end_year,
            )
            if split is None:
                dropped += 1
                continue
            seen_acc.add(acc)
            by_split_country[split].setdefault(country, []).append(
                (acc, date_str, sort_key, seq)
            )
            year_hist[split][y] = year_hist[split].get(y, 0) + 1
        print(f"{raw_path.name}: {len(recs)} spike "
              f"(dup={dup}, bad_date={bad_date}, year_drop={dropped})")

    if n_missing_spike == len(raw_sources):
        print("No spike FASTAs available — cannot build temporal split.")
        return

    prefix_map = {
        "train": "covidttrain", "val": "covidtval", "test": "covidttest",
    }
    counts = {}
    n_groups = {}

    for split, prefix in prefix_map.items():
        out_dir = base / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob(f"{prefix}_group_*"):
            old.unlink()
        g, total = 1, 0
        for country, recs in sorted(by_split_country[split].items()):
            recs.sort(key=lambda r: r[2])
            g_before = g
            g = write_groups(recs, out_dir, prefix, args.group_size, g, args.min_group)
            total += len(recs)
            if g > g_before:
                print(f"  [{split}] {country}: {len(recs)} seqs -> "
                      f"groups {g_before:03d}-{g - 1:03d}")
        counts[split] = total
        n_groups[split] = g - 1
        print(f"=== {split}: {total} seqs, {g - 1} groups -> {out_dir} ===\n")

    protocol = {
        "dataset": "covid",
        "split_type": "temporal",
        "out_base": args.out_base,
        "train_end_year": args.train_end_year,
        "val_year": val_year,
        "test_start_year": args.test_start_year,
        "test_end_year": args.test_end_year,
        "group_size": args.group_size,
        "group_to_season": (
            "Groups are single-country date-ordered chunks within a year-band. "
            "COVID has no flu 'season' — cutoffs are calendar years. "
            "Default train≤2022 / val2023 / test2024–2025 forecasts across "
            "later pandemic years; tighten test-end for single-year holdout."
        ),
        "year_hist": {s: dict(sorted(year_hist[s].items())) for s in year_hist},
        "counts": counts,
        "n_groups": n_groups,
    }
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "SPLIT_PROTOCOL.json", "w") as f:
        json.dump(protocol, f, indent=2)
    print(f"Wrote {base / 'SPLIT_PROTOCOL.json'}")
    print("Next: run_all_groups.py --data-dir data/covid_temporal/{split} "
          "--prefix covidt{split}")
    print("Next: run_all_groups → precompute_plm/ref_rates → train.py")


if __name__ == "__main__":
    main()
