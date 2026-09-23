#!/usr/bin/env python3
"""Prepare an epidemic-aware COVID Spike split.

Builds trees by country × lineage with temporal cutoffs.
Output: ``data/covid_epidemic``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.epidemic_split_common import (
    assign_temporal_split,
    chunk_records,
    slug_label,
    write_fasta_csv_groups,
    write_split_protocol,
)
from scripts.prepare_covid_cladeholdout import load_clade_tsv
from scripts.prepare_covid_geo import RAW_DIR, MIN_GROUP, load_spike_records, parse_date

GROUP_SIZE = 300
PREFIXES = {"train": "covidetrain", "val": "covideval", "test": "covidetest"}


def _clade_tsv_paths(raw_dir: Path, extra: list[str]) -> list[Path]:
    paths = [Path(p) for p in extra]
    globs = [
        "data/covid/clade/*.tsv",
        "data/covid/nextclade/*.tsv",
        "nextclade/*.tsv",
    ]
    for pat in globs:
        base = ROOT if pat.startswith("data/") else raw_dir
        paths.extend(sorted(base.glob(pat)))
    # dedupe
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _epidemic_key(country: str, year: int, clade_map: dict[str, str], acc: str) -> str:
    clade = clade_map.get(acc, "")
    if clade:
        return slug_label(f"{country}_{clade}")
    return slug_label(f"{country}_y{year}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", default="data/covid_epidemic")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--min-group", type=int, default=MIN_GROUP)
    ap.add_argument("--train-end-year", type=int, default=2022)
    ap.add_argument("--val-year", type=int, default=2023)
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--test-start-year", type=int, default=2024)
    ap.add_argument("--test-end-year", type=int, default=2025)
    ap.add_argument("--clade-tsv", action="append", default=[], help="Nextclade TSV (repeatable)")
    args = ap.parse_args()

    val_year = None if args.no_val else args.val_year
    base = ROOT / args.out_base
    raw_dir = ROOT / args.raw_dir
    raw_sources = sorted(raw_dir.glob("*_covid_seqs.fasta"))
    if not raw_sources:
        print(f"No *_covid_seqs.fasta in {raw_dir}")
        sys.exit(1)

    clade_paths = _clade_tsv_paths(raw_dir, args.clade_tsv)
    clade_map = load_clade_tsv(clade_paths)
    print(f"Clade labels: {len(clade_map)} accessions from {len(clade_paths)} TSV path(s)")

    # split -> epidemic_key -> records
    pools: dict[str, dict[str, list]] = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }
    seen: set[str] = set()
    year_hist: dict[str, dict[int, int]] = {s: {} for s in pools}
    epidemic_hist: dict[str, dict[str, int]] = {s: {} for s in pools}

    for raw_path in raw_sources:
        spike_path = raw_path.with_name(
            raw_path.stem.replace("_covid_seqs", "") + "_spike.fasta"
        )
        if not spike_path.exists():
            print(f"WARNING: missing {spike_path.name}")
            continue
        for acc, date_raw, country, seq in load_spike_records(raw_path, spike_path):
            if acc in seen:
                continue
            try:
                date_str, sort_key = parse_date(date_raw)
            except (ValueError, IndexError):
                continue
            y = sort_key[0]
            split = assign_temporal_split(
                y, args.train_end_year, val_year, args.test_start_year, args.test_end_year
            )
            if split is None:
                continue
            seen.add(acc)
            ekey = _epidemic_key(country, y, clade_map, acc)
            row = (acc, date_str, sort_key, seq, country, ekey)
            pools[split][ekey].append(row)
            year_hist[split][y] = year_hist[split].get(y, 0) + 1
            epidemic_hist[split][ekey] = epidemic_hist[split].get(ekey, 0) + 1

    counts = {}
    n_groups = {}
    for split, prefix in PREFIXES.items():
        out_dir = base / split
        for old in out_dir.glob(f"{prefix}_group_*"):
            old.unlink()
        g, total = 1, 0
        for ekey in sorted(pools[split].keys()):
            chunks = chunk_records(
                pools[split][ekey], args.group_size, args.min_group, sort_key_idx=2
            )
            if not chunks:
                continue
            g_end, n = write_fasta_csv_groups(
                chunks,
                out_dir,
                prefix,
                g,
                extra_csv_cols=["country", "epidemic_key"],
                extra_row_fn=lambda r: [r[0], r[1], r[4], r[5]],
            )
            print(f"  [{split}] {ekey}: {len(pools[split][ekey])} seqs -> "
                  f"groups {g:03d}-{g_end - 1:03d}")
            g = g_end
            total += n
        counts[split] = total
        n_groups[split] = g - 1
        print(f"=== {split}: {total} seqs, {g - 1} groups ===\n")

    clade_frac = sum(1 for a in seen if a in clade_map) / max(len(seen), 1)
    protocol = {
        "dataset": "covid",
        "split_type": "epidemic",
        "out_base": args.out_base,
        "train_end_year": args.train_end_year,
        "val_year": val_year,
        "test_start_year": args.test_start_year,
        "test_end_year": args.test_end_year,
        "group_size": args.group_size,
        "tree_semantics": "one tree per (country, Nextclade clade); "
        "fallback (country, year) without clade TSV",
        "clade_labeled_frac": round(clade_frac, 3),
        "year_hist": {s: dict(sorted(year_hist[s].items())) for s in year_hist},
        "top_epidemics": {
            s: dict(sorted(epidemic_hist[s].items(), key=lambda x: -x[1])[:30])
            for s in epidemic_hist
        },
        "counts": counts,
        "n_groups": n_groups,
        "checkpoint_target": "checkpoints/covid_v6_epidemic_mutrec",
    }
    write_split_protocol(base, protocol)
    print(f"Wrote {base / 'SPLIT_PROTOCOL.json'}")
    if clade_frac < 0.3:
        print("WARNING: <30% clade labels — run prepare_covid_cladeholdout.py --write-clade-tsv")


if __name__ == "__main__":
    main()
