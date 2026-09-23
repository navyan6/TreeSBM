#!/usr/bin/env python3
"""Merge H3N2 + H1N1 (+ optional Flu B) HA pools for multi-subtype training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Bio import SeqIO

from scripts.epidemic_split_common import parse_flu_header, write_split_protocol
from scripts.prepare_h3n2_temporal import SOURCE_WINDOWS as H3_WINDOWS, _id, _year
from scripts.prepare_h1n1_geo import RAW_DIR as H1_RAW

SUBTYPES = ("h3n2", "h1n1", "flub")
POOL_FASTA = {
    "h3n2": ("data/h3n2", {"train": "h3n2train", "val": "h3n2val", "test": "h3n2test"}),
    "h1n1": ("data/h1n1", {"train": "h1n1train", "val": "h1n1val", "test": "h1n1test"}),
    "flub": ("data/flub", {"train": "flubtrain", "val": "flubval", "test": "flubtest"}),
}


def _load_h3n2_from_windows(min_year: int | None) -> list[tuple]:
    rows = []
    seen: set[str] = set()
    for wp, _, _ in H3_WINDOWS:
        path = ROOT / wp
        if not path.is_file():
            continue
        for rec in SeqIO.parse(path, "fasta"):
            y = _year(rec)
            if y is None or (min_year and y < min_year):
                continue
            rid = _id(rec)
            if rid in seen:
                continue
            seen.add(rid)
            _, date_str, _, _, _ = parse_flu_header(rec.description or f"{rid},{y:04d}-01-01")
            rows.append((rid, date_str, str(rec.seq), "h3n2"))
    return rows


def _load_from_pools(subtype: str, pool_base: Path) -> list[tuple]:
    _, names = POOL_FASTA[subtype]
    rows = []
    seen: set[str] = set()
    for split, prefix in names.items():
        pool = pool_base / split / f"{prefix}.fasta"
        if not pool.is_file():
            continue
        for rec in SeqIO.parse(pool, "fasta"):
            rid = rec.id.split(",")[0].strip()
            if rid in seen:
                continue
            seen.add(rid)
            date_str = rec.description.strip() if rec.description else "2000-01-01"
            if "," in rec.id:
                date_str = rec.id.split(",", 1)[1].strip()
            rows.append((rid, date_str, str(rec.seq), subtype))
    return rows


def _load_h1n1_raw(min_year: int) -> list[tuple]:
    rows = []
    seen: set[str] = set()
    raw_dir = ROOT / H1_RAW
    for fp in sorted(raw_dir.glob("*_h1n1.fasta")):
        for rec in SeqIO.parse(fp, "fasta"):
            rid = rec.id.split()[0].split(",")[0]
            if rid in seen:
                continue
            seen.add(rid)
            parts = (rec.description or "").split("|")
            date_str = parts[3].strip() if len(parts) > 3 else "2000-01-01"
            try:
                y = int(date_str[:4])
            except ValueError:
                y = 2000
            if y < min_year:
                continue
            rows.append((rid, date_str, str(rec.seq), "h1n1"))
    return rows


def load_subtype(subtype: str, pool_base: Path | None, min_year: int) -> list[tuple]:
    if pool_base and pool_base.is_dir():
        rows = _load_from_pools(subtype, pool_base)
        if rows:
            return rows
    if subtype == "h3n2":
        return _load_h3n2_from_windows(min_year)
    if subtype == "h1n1":
        return _load_h1n1_raw(min_year)
    flub_pool = ROOT / "data/flub/train/flubtrain.fasta"
    if flub_pool.is_file():
        return _load_from_pools("flub", ROOT / "data/flub")
    print(f"WARNING: no Flu B source — create {flub_pool} before pan-flu runs")
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", default="data/panflu_pool")
    ap.add_argument("--min-year", type=int, default=2009)
    ap.add_argument("--h3n2-pool", default="data/h3n2")
    ap.add_argument("--h1n1-pool", default="data/h1n1")
    ap.add_argument("--include-flub", action="store_true")
    args = ap.parse_args()

    base = ROOT / args.out_base
    inv_dir = base / "inventory"
    inv_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    season_hist: dict[str, Counter] = {}
    all_rows: list[tuple] = []

    sources = {
        "h3n2": ROOT / args.h3n2_pool,
        "h1n1": ROOT / args.h1n1_pool,
    }
    subtypes = list(SUBTYPES[:2])
    if args.include_flub:
        subtypes.append("flub")

    for st in subtypes:
        rows = load_subtype(st, sources.get(st), args.min_year)
        counts[st] = len(rows)
        season_hist[st] = Counter()
        out_fa = inv_dir / f"{st}_ha.fasta"
        with open(out_fa, "w") as ff:
            for rid, date_str, seq, subtype in rows:
                ff.write(f">{rid},{date_str},subtype={subtype}\n{seq}\n")
                _, _, _, _, season = parse_flu_header(f"{rid},{date_str}")
                if season:
                    season_hist[st][season] += 1
                all_rows.append((rid, date_str, seq, subtype))
        print(f"{st}: {len(rows)} seqs -> {out_fa}")

    master = base / "panflu_master.fasta"
    with open(master, "w") as ff:
        for rid, date_str, seq, subtype in all_rows:
            ff.write(f">{rid},{date_str},subtype={subtype}\n{seq}\n")

    protocol = {
        "dataset": "panflu_pool",
        "split_type": "inventory",
        "out_base": args.out_base,
        "subtypes": subtypes,
        "counts_by_subtype": counts,
        "total_seqs": sum(counts.values()),
        "season_hist_by_subtype": {k: dict(v) for k, v in season_hist.items()},
        "master_fasta": str(master.relative_to(ROOT)),
        "note": "Use prepare_panflu_forecast.py for train/val/test holdouts.",
    }
    write_split_protocol(base, protocol)
    print(f"Wrote {base / 'SPLIT_PROTOCOL.json'} total={sum(counts.values())}")


if __name__ == "__main__":
    main()
