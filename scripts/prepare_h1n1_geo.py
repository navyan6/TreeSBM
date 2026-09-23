#!/usr/bin/env python3
"""Prepare a geographically split H1N1 HA dataset.

Output: ``data/h1n1``.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from Bio import SeqIO

RAW_DIR = "data/h1n1/train"      # raw *_h1n1.fasta regional dumps live here
LOC_MIN = 100                    # a fine location needs >= this to be its own unit
GROUP_SIZE = 300                 # target tree size (aim 100-500)
MIN_GROUP = 50                   # drop a chunk/remainder smaller than this


def normalize_location(field2: str) -> str:
    """Unify location strings: 'USA: California state' / 'USA: California' -> same."""
    s = field2.strip()
    s = re.sub(r"\s+state\b", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_date(raw: str) -> tuple[str, tuple[int, int, int]]:
    """'2020' / '2020-03' / '2020-03-15' -> (augur-style date str w/ XX, sort key)."""
    parts = raw.split("-")
    y = int(parts[0])
    if len(parts) == 1:
        return f"{y:04d}-XX-XX", (y, 7, 15)
    if len(parts) == 2:
        m = int(parts[1])
        return f"{y:04d}-{m:02d}-XX", (y, m, 15)
    m, d = int(parts[1]), int(parts[2])
    return f"{y:04d}-{m:02d}-{d:02d}", (y, m, d)


def parse_header(desc: str) -> tuple[str, str, str, str]:
    """>ACC |description|LOCATION|DATE|COUNTRY|LENGTH -> (acc, location, date, country)."""
    parts = desc.split("|")
    acc = parts[0].strip().lstrip(">").split()[0]
    location = normalize_location(parts[2])
    date = parts[3].strip()
    country = parts[4].strip()
    return acc, location, date, country


def greedy_geo_split(unit_counts: dict[str, int]) -> dict[str, str]:
    """
    Assign whole units to train/val/test targeting 80/10/10 by seq count.
    Largest-first into whichever split is furthest below its target (deterministic).
    """
    total = sum(unit_counts.values())
    target = {"train": 0.80 * total, "val": 0.10 * total, "test": 0.10 * total}
    fill = {"train": 0.0, "val": 0.0, "test": 0.0}
    assign = {}
    for unit, c in sorted(unit_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        split = max(("train", "val", "test"), key=lambda s: target[s] - fill[s])
        assign[unit] = split
        fill[split] += c
    return assign


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
    """
    records: (acc, augur_date_str, sort_key, seq), single unit, date-ordered.
    Chunk into date-contiguous trees, closing a tree when it hits group_size OR
    would exceed max_span_years -- so each tree is a coherent time window (the
    "Kenya 2022-2024" shape), not a century-spanning mix of distinct lineages.
    Chunks smaller than min_group (sparse tails) are dropped.
    """
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", default="data/h1n1")
    ap.add_argument("--loc-min", type=int, default=LOC_MIN)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--min-group", type=int, default=MIN_GROUP)
    ap.add_argument("--max-span-years", type=int, default=4,
                    help="Close a tree once its date range would exceed this many "
                         "years, so trees stay temporally coherent (H1N1 mixes "
                         "distinct pandemic lineages across decades).")
    args = ap.parse_args()
    loc_min, group_size, min_group = args.loc_min, args.group_size, args.min_group
    max_span_years = args.max_span_years

    raw_dir = ROOT / RAW_DIR
    base = ROOT / args.out_base
    raw_files = sorted(raw_dir.glob("*_h1n1.fasta"))
    if not raw_files:
        print(f"No *_h1n1.fasta in {raw_dir}"); return

    # ── pass 1: load all records, count fine-location frequencies
    records = []          # (acc, location, country, date_raw, seq)
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

    # ── pass 2: assign each seq to a group unit (fine loc if big enough, else country)
    unit_records: dict[str, list] = {}
    unit_counts: dict[str, int] = {}
    dropped_date = 0
    for acc, loc, country, date_raw, seq in records:
        unit = loc if loc_count[loc] >= loc_min else country
        try:
            date_str, sort_key = parse_date(date_raw)
        except (ValueError, IndexError):
            dropped_date += 1; continue
        unit_records.setdefault(unit, []).append((acc, date_str, sort_key, seq))
        unit_counts[unit] = unit_counts.get(unit, 0) + 1
    print(f"{len(unit_counts)} group units "
          f"({sum(1 for k in unit_counts if k in loc_count and loc_count[k] >= loc_min)} fine-location, "
          f"rest country-pooled); {dropped_date} unparseable-date seqs dropped")

    # ── geographic split over whole units
    assign = greedy_geo_split(unit_counts)
    total = sum(unit_counts.values())
    for split in ("train", "val", "test"):
        s_units = [u for u, sp in assign.items() if sp == split]
        s_seqs = sum(unit_counts[u] for u in s_units)
        print(f"  [{split}] {len(s_units)} units, {s_seqs} seqs ({100*s_seqs/total:.1f}%)")

    # ── write groups per split (clear stale group files from a prior run first)
    prefix_map = {"train": "h1n1train", "val": "h1n1val", "test": "h1n1test"}
    for split, prefix in prefix_map.items():
        for old in (base / split).glob(f"{prefix}_group_*"):
            old.unlink()
    g_counter = {"train": 1, "val": 1, "test": 1}
    for unit in sorted(unit_counts, key=lambda u: (-unit_counts[u], u)):
        split = assign[unit]
        recs = sorted(unit_records[unit], key=lambda r: r[2])
        out_dir = base / split
        g_counter[split] = write_groups(recs, out_dir, prefix_map[split], g_counter[split],
                                        group_size, min_group, max_span_years)

    for split in ("train", "val", "test"):
        n_groups = g_counter[split] - 1
        print(f"=== {split}: {n_groups} groups written to {base / split} ===")
    print("\nNext: run_all_groups.py --data-dir data/h1n1/{split} --prefix h1n1{split}")


if __name__ == "__main__":
    main()
