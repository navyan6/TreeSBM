#!/usr/bin/env python3
"""Build pan-flu forecast holdout splits.

Calendar-year and Northern-Hemisphere season tests with subtype rotation.
Writes matrix dirs and ``SPLIT_PROTOCOL.json``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.epidemic_split_common import (
    chunk_records,
    nh_flu_season,
    parse_flu_header,
    write_fasta_csv_groups,
    write_split_protocol,
)
from scripts.prepare_panflu_pool import load_subtype

TRAIN_POOLS = {
    "h3n2": ("h3n2",),
    "h3n2_h1n1": ("h3n2", "h1n1"),
    "panflu": ("h3n2", "h1n1", "flub"),
}
PREFIX = {"train": "pflutrain", "val": "pfluval", "test": "pflutest"}
CKPT_NAMES = {
    ("h3n2", "h3n2", "calendar"): "h3n2_only_forecast_2020_cal",
    ("h3n2", "h3n2", "season"): "h3n2_only_forecast_2020_season",
    ("h3n2_h1n1", "h3n2", "calendar"): "dual_forecast_h3n2_cal",
    ("h3n2_h1n1", "h1n1", "calendar"): "dual_forecast_h1n1_cal",
    ("panflu", "h3n2", "calendar"): "panflu_forecast_h3n2_cal",
    ("panflu", "h3n2", "season"): "panflu_forecast_h3n2_season",
    ("panflu", "h1n1", "calendar"): "panflu_forecast_h1n1_cal",
    ("panflu", "flub", "calendar"): "panflu_forecast_flub_cal",
}


def _parse_date_fields(date_str: str) -> tuple[int | None, int | None, str | None]:
    _, ds, y, m, season = parse_flu_header(f"x,{date_str}")
    return y, m, season


def assign_record(
    subtype: str,
    date_str: str,
    test_subtype: str,
    test_mode: str,
    test_year: int,
    test_season: str,
    train_end_year: int,
) -> str | None:
    y, m, season = _parse_date_fields(date_str)
    if y is None:
        return None
    if season is None:
        season = nh_flu_season(y, m or 7)

    if test_mode == "calendar":
        if subtype == test_subtype and y == test_year:
            return "test"
        # Val = final pre-holdout year of non-test subtypes (was previously
        # unreachable because y <= train_end_year always hit train first).
        if y < train_end_year:
            return "train"
        if y == train_end_year:
            if subtype != test_subtype:
                return "val"
            return "train"
        return None

    if subtype == test_subtype and season == test_season:
        return "test"
    if season < test_season:
        return "train"
    return None


def load_pool_rows(
    train_pool: str,
    h3n2_pool: Path,
    h1n1_pool: Path,
    panflu_pool: Path | None,
    min_year: int,
) -> list[tuple]:
    rows = []
    for st in TRAIN_POOLS[train_pool]:
        if st == "flub" and panflu_pool and (panflu_pool / "inventory/flub_ha.fasta").is_file():
            from Bio import SeqIO
            for rec in SeqIO.parse(panflu_pool / "inventory/flub_ha.fasta", "fasta"):
                parts = (rec.description or rec.id).split(",")
                rid = parts[0].strip().lstrip(">")
                date_str = parts[1].strip() if len(parts) > 1 else "2000-01-01"
                rows.append((rid, date_str, date_str, str(rec.seq), "flub"))
            continue
        src = h3n2_pool if st == "h3n2" else h1n1_pool
        for rid, date_str, seq, subtype in load_subtype(st, src, min_year):
            rows.append((rid, date_str, date_str, seq, subtype))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pool", choices=sorted(TRAIN_POOLS.keys()), default="h3n2")
    ap.add_argument("--test-subtype", choices=("h3n2", "h1n1", "flub"), default="h3n2")
    ap.add_argument("--test-mode", choices=("calendar", "season"), default="calendar")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--test-season", default="2019-2020")
    ap.add_argument("--train-end-year", type=int, default=2019)
    ap.add_argument("--group-size", type=int, default=400)
    ap.add_argument("--min-group", type=int, default=40)
    ap.add_argument("--min-year", type=int, default=2009)
    ap.add_argument("--h3n2-pool", default="data/h3n2")
    ap.add_argument("--h1n1-pool", default="data/h1n1")
    ap.add_argument("--panflu-pool", default="data/panflu_pool")
    ap.add_argument(
        "--out-base",
        default="",
        help="Default: data/panflu_forecast_{subtype}_{cal|season}",
    )
    ap.add_argument(
        "--alias-out",
        default="",
        help="Also write protocol field alias e.g. data/h3n2_forecast_2020_cal",
    )
    args = ap.parse_args()

    mode_suffix = "cal" if args.test_mode == "calendar" else "season"
    out_base = args.out_base or f"data/panflu_forecast_{args.test_subtype}_{mode_suffix}"
    base = ROOT / out_base

    rows = load_pool_rows(
        args.train_pool,
        ROOT / args.h3n2_pool,
        ROOT / args.h1n1_pool,
        ROOT / args.panflu_pool,
        args.min_year,
    )
    if not rows:
        print("No sequences loaded — check pools / run prepare_panflu_pool.py")
        sys.exit(1)

    pools: dict[str, dict[str, list]] = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }
    for rid, date_str, sort_key, seq, subtype in rows:
        split = assign_record(
            subtype,
            date_str,
            args.test_subtype,
            args.test_mode,
            args.test_year,
            args.test_season,
            args.train_end_year,
        )
        if split is None:
            continue
        _, _, y, m, season = parse_flu_header(f"{rid},{date_str}")
        if not season and y:
            season = nh_flu_season(y, m or 7)
        gkey = f"{subtype}_{season or 'unknown'}"
        pools[split][gkey].append((rid, date_str, sort_key, seq, subtype, season or ""))

    counts = {}
    n_groups = {}
    for split, prefix in PREFIX.items():
        out_dir = base / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob(f"{prefix}_group_*"):
            old.unlink()
        g, total = 1, 0
        for gkey in sorted(pools[split].keys()):
            chunks = chunk_records(
                pools[split][gkey], args.group_size, args.min_group, sort_key_idx=2
            )
            if not chunks:
                continue
            g_end, n = write_fasta_csv_groups(
                chunks,
                out_dir,
                prefix,
                g,
                extra_csv_cols=["subtype", "season"],
                extra_row_fn=lambda r: [r[0], r[1], r[4], r[5]],
            )
            g = g_end
            total += n
        counts[split] = total
        n_groups[split] = max(0, g - 1)
        print(f"=== {split}: {total} seqs, {n_groups[split]} groups ===")

    ckpt_key = (args.train_pool, args.test_subtype, args.test_mode)
    protocol = {
        "dataset": "panflu_forecast",
        "split_type": "forecast_holdout",
        "out_base": out_base,
        "train_pool": args.train_pool,
        "train_subtypes": list(TRAIN_POOLS[args.train_pool]),
        "test_subtype": args.test_subtype,
        "test_mode": args.test_mode,
        "test_year": args.test_year if args.test_mode == "calendar" else None,
        "test_season": args.test_season if args.test_mode == "season" else None,
        "train_end_year": args.train_end_year,
        "group_size": args.group_size,
        "tree_semantics": "one tree per (subtype, NH flu season); cross-strain diversity in train",
        "counts": counts,
        "n_groups": n_groups,
        "checkpoint_target": CKPT_NAMES.get(ckpt_key, f"panflu_holdout_{args.test_subtype}"),
        "matrix_row": {
            "B0": ("h3n2", "h3n2", "calendar"),
            "B1": ("h3n2", "h3n2", "season"),
            "G1": ("h3n2_h1n1", "h3n2", "calendar"),
            "G2": ("h3n2_h1n1", "h1n1", "calendar"),
            "G3": ("panflu", "h3n2", "calendar"),
            "G4": ("panflu", "h3n2", "season"),
            "G5": ("panflu", "h1n1", "calendar"),
            "G6": ("panflu", "flub", "calendar"),
        },
    }
    if args.alias_out:
        protocol["alias_out"] = args.alias_out
    write_split_protocol(base, protocol)
    print(f"Wrote {base / 'SPLIT_PROTOCOL.json'}")
    print(f"Target ckpt: {protocol['checkpoint_target']}")


if __name__ == "__main__":
    main()
