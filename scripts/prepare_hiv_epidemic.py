#!/usr/bin/env python3
"""Prepare an epidemic-aware HIV-1 Env split.

Builds trees by geographic unit × collection year with temporal cutoffs.
Output: ``data/hiv_epidemic``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.epidemic_split_common import (  # noqa: E402
    chunk_records,
    slug_label,
    write_fasta_csv_groups,
    write_split_protocol,
)
from scripts.hiv_env_utils import load_eligible_env_records  # noqa: E402
from scripts.prepare_hiv_splits import (  # noqa: E402
    LOC_MIN,
    RAW_GLOB,
    assign_temporal_split,
    resolve_unit,
)

GROUP_SIZE = 300
MIN_GROUP = 50
PREFIXES = {"train": "hivetrain", "val": "hiveval", "test": "hivetest"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-glob", default=RAW_GLOB)
    ap.add_argument("--out-base", default="data/hiv_epidemic")
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--min-group", type=int, default=MIN_GROUP)
    ap.add_argument("--loc-min", type=int, default=LOC_MIN)
    ap.add_argument("--train-end-year", type=int, default=2014)
    ap.add_argument("--val-start-year", type=int, default=2015)
    ap.add_argument("--val-end-year", type=int, default=2015)
    ap.add_argument("--test-start-year", type=int, default=2016)
    ap.add_argument("--no-genomes", action="store_true")
    args = ap.parse_args()

    raw_files = sorted(ROOT.glob(args.raw_glob))
    if not raw_files:
        print(f"ERROR: no files matching {args.raw_glob}", file=sys.stderr)
        sys.exit(1)

    records, stats = load_eligible_env_records(
        raw_files, include_genomes=not args.no_genomes
    )
    print("Eligibility:", json.dumps(stats, indent=2))
    print(f"Kept Env CDS: {len(records)}")

    loc_count: dict[str, int] = Counter()
    for r in records:
        loc_count[r.unit] += 1

    pools: dict[str, dict[str, list]] = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }
    year_hist: dict[str, Counter] = {s: Counter() for s in pools}
    epidemic_hist: dict[str, Counter] = {s: Counter() for s in pools}
    skipped = 0

    for r in records:
        y = int(r.sort_key[0])
        split = assign_temporal_split(
            y,
            args.train_end_year,
            args.val_start_year,
            args.val_end_year,
            args.test_start_year,
        )
        if split is None:
            skipped += 1
            continue
        unit = resolve_unit(r, loc_count, args.loc_min)
        ekey = slug_label(f"{unit}_y{y}")
        # (acc, date_str, sort_key, seq, unit, year, country)
        pools[split][ekey].append(
            (r.acc, r.date_str, r.sort_key, r.nt, unit, y, r.country)
        )
        year_hist[split][str(y)] += 1
        epidemic_hist[split][ekey] += 1

    base = ROOT / args.out_base
    for split in pools:
        out_dir = base / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob(f"{PREFIXES[split]}_group_*"):
            old.unlink()

    n_groups = {}
    n_seqs = {}
    for split in ("train", "val", "test"):
        group_lists = []
        for ekey in sorted(pools[split], key=lambda k: (-len(pools[split][k]), k)):
            group_lists.extend(
                chunk_records(
                    pools[split][ekey],
                    args.group_size,
                    args.min_group,
                    sort_key_idx=2,
                )
            )
        g_next, n_seq = write_fasta_csv_groups(
            group_lists,
            base / split,
            PREFIXES[split],
            start_group=1,
            extra_csv_cols=["unit", "year", "country"],
        )
        n_groups[split] = g_next - 1
        n_seqs[split] = n_seq
        print(
            f"  [{split}] {n_seqs[split]} seqs -> {n_groups[split]} groups "
            f"({len(pools[split])} epidemic keys)"
        )

    write_split_protocol(
        base,
        {
            "dataset": "hiv_env",
            "split_type": "epidemic",
            "out_base": str(args.out_base),
            "train_end_year": args.train_end_year,
            "val_start_year": args.val_start_year,
            "val_end_year": args.val_end_year,
            "test_start_year": args.test_start_year,
            "group_size": args.group_size,
            "min_group": args.min_group,
            "tree_semantics": "one tree per (geo unit, collection year); "
            "oversized keys date-chunked",
            "n_groups": n_groups,
            "n_seqs": n_seqs,
            "year_hist": {s: dict(year_hist[s]) for s in year_hist},
            "top_epidemics": {
                s: dict(epidemic_hist[s].most_common(30)) for s in epidemic_hist
            },
            "skipped_out_of_band": skipped,
            "eligibility": stats,
            "checkpoint_target": "checkpoints/hiv_epidemic_v1",
        },
    )
    print(f"Wrote {base}/SPLIT_PROTOCOL.json (skipped out-of-band={skipped})")


if __name__ == "__main__":
    main()
