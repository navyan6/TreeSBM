#!/usr/bin/env python3
"""Prepare HIV-1 Env temporal and geographic splits.

Outputs: ``data/hiv_temporal``, ``data/hiv_geo``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.hiv_env_utils import (  # noqa: E402
    greedy_geo_split,
    load_eligible_env_records,
    write_date_contiguous_groups,
)

RAW_GLOB = "data/hiv/*.fasta"
GROUP_SIZE = 300
MIN_GROUP = 200  # user target band 200–500
MAX_SPAN_YEARS = 6
LOC_MIN = 100  # fine location needs >= this to stay its own unit


def inventory_raw(raw_files: list[Path]) -> dict:
    """Lightweight raw FASTA inventory (n, pipe fields, length buckets)."""
    from collections import Counter

    files = []
    for fp in raw_files:
        n = 0
        lens = []
        pipes = Counter()
        with open(fp) as f:
            seq = []
            for line in f:
                if line.startswith(">"):
                    if seq:
                        n += 1
                        lens.append(sum(len(x) for x in seq))
                    pipes[line.count("|")] += 1
                    seq = []
                else:
                    seq.append(line.strip())
            if seq:
                n += 1
                lens.append(sum(len(x) for x in seq))
        entry = {
            "path": str(fp.relative_to(ROOT)),
            "n_seq": n,
            "nt_len_min": min(lens) if lens else None,
            "nt_len_median": sorted(lens)[len(lens) // 2] if lens else None,
            "nt_len_max": max(lens) if lens else None,
            "pipe_field_counts": dict(pipes),
            "header_layout": (
                ">ACC |description|organism|LOCATION|COUNTRY|DATE|LENGTH"
            ),
        }
        files.append(entry)
    return {
        "n_files": len(files),
        "n_seq_total": sum(e["n_seq"] for e in files),
        "files": files,
    }


def assign_temporal_split(
    year: int,
    train_end: int,
    val_start: int,
    val_end: int,
    test_start: int,
) -> str | None:
    if year <= train_end:
        return "train"
    if val_start <= year <= val_end:
        return "val"
    if year >= test_start:
        return "test"
    return None


def resolve_unit(rec, loc_count: dict[str, int], loc_min: int) -> str:
    """Fine location if frequent enough, else country (H1N1-style)."""
    # rec.unit already state-or-country; also allow falling back further
    if rec.has_state:
        # count by the state unit string
        if loc_count.get(rec.unit, 0) >= loc_min:
            return rec.unit
        return rec.country
    return rec.country


def build_groups_for_split(
    unit_to_recs: dict[str, list],
    out_base: Path,
    split: str,
    prefix: str,
    group_size: int,
    min_group: int,
    max_span_years: int,
) -> list[dict]:
    out_dir = out_base / split
    for old in out_dir.glob(f"{prefix}_group_*"):
        old.unlink()
    g = 1
    all_summ = []
    for unit in sorted(unit_to_recs, key=lambda u: (-len(unit_to_recs[u]), u)):
        recs = sorted(unit_to_recs[unit], key=lambda r: r.sort_key)
        tuples = [(r.acc, r.date_str, r.sort_key, r.nt) for r in recs]
        g, summ = write_date_contiguous_groups(
            tuples,
            out_dir,
            prefix,
            g,
            group_size,
            min_group,
            max_span_years,
            meta_extra={"unit": unit, "split": split},
        )
        all_summ.extend(summ)
    return all_summ


def year_histogram(records) -> dict[str, int]:
    c = Counter(r.sort_key[0] for r in records)
    return {str(y): c[y] for y in sorted(c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-glob", default=RAW_GLOB)
    ap.add_argument("--temporal-out", default="data/hiv_temporal")
    ap.add_argument("--geo-out", default="data/hiv_geo")
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--min-group", type=int, default=MIN_GROUP)
    ap.add_argument("--max-span-years", type=int, default=MAX_SPAN_YEARS)
    ap.add_argument("--loc-min", type=int, default=LOC_MIN)
    ap.add_argument("--train-end-year", type=int, default=2014)
    ap.add_argument("--val-start-year", type=int, default=2015)
    ap.add_argument("--val-end-year", type=int, default=2015)
    ap.add_argument("--test-start-year", type=int, default=2016)
    ap.add_argument("--no-genomes", action="store_true")
    ap.add_argument("--skip-temporal", action="store_true")
    ap.add_argument("--skip-geo", action="store_true")
    ap.add_argument("--inventory-only", action="store_true")
    args = ap.parse_args()

    raw_files = sorted(ROOT.glob(args.raw_glob))
    if not raw_files:
        print(f"ERROR: no files matching {args.raw_glob}", file=sys.stderr)
        sys.exit(1)

    inv = inventory_raw(raw_files)
    print(f"Raw inventory: {inv['n_files']} files, {inv['n_seq_total']} seqs")
    for e in inv["files"]:
        print(
            f"  {e['path']}: n={e['n_seq']}  "
            f"NT[{e['nt_len_min']},{e['nt_len_median']},{e['nt_len_max']}]"
        )

    records, stats = load_eligible_env_records(
        raw_files, include_genomes=not args.no_genomes
    )
    print("Eligibility:", json.dumps(stats, indent=2))
    print(f"Kept Env CDS: {len(records)}")
    yhist = year_histogram(records)
    print("Year histogram (kept):")
    tot = sum(yhist.values()) or 1
    cum = 0
    for y, c in yhist.items():
        cum += c
        print(f"  {y}: {c:5d}  cum={100 * cum / tot:.1f}%")

    inv_path = ROOT / "data" / "hiv_inventory.json"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.write_text(
        json.dumps(
            {
                "raw": inv,
                "eligibility": stats,
                "year_histogram_kept": yhist,
                "n_kept": len(records),
                "frame_orf_policy": (
                    "Near-complete Env NT [2200,2800]; require ATG + Env "
                    "signal-like N-term (MRV*/MKV*/…); genomes scanned for "
                    "Env ORF; empty/range dates dropped (not imputed)."
                ),
                "hxb2_v_regions": {
                    "V1": [131, 157],
                    "V2": [158, 196],
                    "V3": [296, 331],
                    "V4": [385, 418],
                    "V5": [461, 471],
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {inv_path}")

    if args.inventory_only:
        return

    if stats["empty_date"] > 0.25 * stats["n_raw"]:
        print(
            f"NOTE: {stats['empty_date']} empty-date seqs dropped "
            f"({100 * stats['empty_date'] / max(stats['n_raw'], 1):.1f}% of raw). "
            "Not imputed."
        )
    if stats["range_or_bad_date"] > 1000:
        print(
            f"NOTE: {stats['range_or_bad_date']} range/bad dates dropped "
            "(e.g. 2015/2018) — not mid-pointed."
        )
    if stats["extract_fail"] > 5000:
        print(
            f"WARN: {stats['extract_fail']} env/genome seqs failed frame QC — "
            "see hiv_env_utils.extract_env_cds; not forcing weak ORFs."
        )

    # ── Temporal split ──────────────────────────────────────────────
    if not args.skip_temporal:
        tbase = ROOT / args.temporal_out
        # First assign split by year, then unit within split, then group.
        # Critical: splits are by TREE; we build trees only within one
        # (split × unit) so a tree never crosses splits.
        split_unit_recs: dict[str, dict[str, list]] = {
            "train": {},
            "val": {},
            "test": {},
        }
        # Unit frequency within each temporal band (for loc_min fallback)
        band_loc_count: dict[str, Counter] = {
            "train": Counter(),
            "val": Counter(),
            "test": Counter(),
        }
        skipped_gap = 0
        for r in records:
            sp = assign_temporal_split(
                r.sort_key[0],
                args.train_end_year,
                args.val_start_year,
                args.val_end_year,
                args.test_start_year,
            )
            if sp is None:
                skipped_gap += 1
                continue
            band_loc_count[sp][r.unit] += 1

        for r in records:
            sp = assign_temporal_split(
                r.sort_key[0],
                args.train_end_year,
                args.val_start_year,
                args.val_end_year,
                args.test_start_year,
            )
            if sp is None:
                continue
            unit = r.unit
            if r.has_state and band_loc_count[sp][r.unit] < args.loc_min:
                unit = r.country
            split_unit_recs[sp].setdefault(unit, []).append(r)

        all_groups = {}
        for split in ("train", "val", "test"):
            prefix = f"hivtemporal{split}"
            summ = build_groups_for_split(
                split_unit_recs[split],
                tbase,
                split,
                prefix,
                args.group_size,
                args.min_group,
                args.max_span_years,
            )
            all_groups[split] = summ
            n_leaves = sum(g["n_leaves"] for g in summ)
            print(
                f"[temporal {split}] {len(summ)} trees, {n_leaves} leaves "
                f"from {len(split_unit_recs[split])} units"
            )

        n_trees = {s: len(all_groups[s]) for s in all_groups}
        tot_trees = sum(n_trees.values()) or 1
        protocol = {
            "split_type": "temporal",
            "gene": "HIV-1 Env (nucleotide CDS → AA after ASR)",
            "raw_glob": args.raw_glob,
            "cutoffs": {
                "train_end_year": args.train_end_year,
                "val_start_year": args.val_start_year,
                "val_end_year": args.val_end_year,
                "test_start_year": args.test_start_year,
                "note": (
                    "Adjusted from Env-eligible year histogram for ~80/10/10 "
                    "by trees; default train≤2021/val2022-23/test≥2024 leaves "
                    "almost no late trees on this corpus."
                ),
            },
            "grouping": {
                "group_size": args.group_size,
                "min_group": args.min_group,
                "max_span_years": args.max_span_years,
                "loc_min": args.loc_min,
                "rule": "date-contiguous within one geoloc unit; no region mix",
            },
            "eligibility": stats,
            "year_histogram_kept": yhist,
            "n_trees": n_trees,
            "tree_frac": {s: n_trees[s] / tot_trees for s in n_trees},
            "n_leaves": {
                s: sum(g["n_leaves"] for g in all_groups[s]) for s in all_groups
            },
            "groups": all_groups,
            "skipped_year_gap": skipped_gap,
            "frame_orf_policy": (
                "Env ATG + signal-like N-term; HXB2-length window; "
                "empty/range dates dropped"
            ),
        }
        (tbase / "SPLIT_PROTOCOL.json").write_text(json.dumps(protocol, indent=2))
        # Size report
        report = []
        for split in ("train", "val", "test"):
            units = sorted({g["unit"] for g in all_groups[split]})
            report.append(
                {
                    "split": split,
                    "n_trees": n_trees[split],
                    "n_leaves": protocol["n_leaves"][split],
                    "tree_pct": round(100 * n_trees[split] / tot_trees, 1),
                    "n_units": len(units),
                    "year_min": min((g["year_min"] for g in all_groups[split]), default=None),
                    "year_max": max((g["year_max"] for g in all_groups[split]), default=None),
                    "units_sample": units[:20],
                }
            )
        (tbase / "SIZE_REPORT.json").write_text(json.dumps(report, indent=2))
        print(f"Wrote {tbase / 'SPLIT_PROTOCOL.json'}")
        print("Temporal size report:", json.dumps(report, indent=2))

    # ── Geographic split ────────────────────────────────────────────
    if not args.skip_geo:
        gbase = ROOT / args.geo_out
        loc_count = Counter(r.unit for r in records)
        unit_recs: dict[str, list] = {}
        unit_counts: dict[str, int] = {}
        for r in records:
            unit = r.unit if loc_count[r.unit] >= args.loc_min else r.country
            unit_recs.setdefault(unit, []).append(r)
            unit_counts[unit] = unit_counts.get(unit, 0) + 1

        assign = greedy_geo_split(unit_counts)
        total = sum(unit_counts.values())
        for split in ("train", "val", "test"):
            s_units = [u for u, sp in assign.items() if sp == split]
            s_seqs = sum(unit_counts[u] for u in s_units)
            print(
                f"[geo units→{split}] {len(s_units)} units, "
                f"{s_seqs} seqs ({100 * s_seqs / total:.1f}%)"
            )

        split_unit_recs = {"train": {}, "val": {}, "test": {}}
        for unit, recs in unit_recs.items():
            split_unit_recs[assign[unit]][unit] = recs

        all_groups = {}
        for split in ("train", "val", "test"):
            prefix = f"hivgeo{split}"
            summ = build_groups_for_split(
                split_unit_recs[split],
                gbase,
                split,
                prefix,
                args.group_size,
                args.min_group,
                args.max_span_years,
            )
            all_groups[split] = summ
            n_leaves = sum(g["n_leaves"] for g in summ)
            print(
                f"[geo {split}] {len(summ)} trees, {n_leaves} leaves "
                f"from {len(split_unit_recs[split])} units"
            )

        n_trees = {s: len(all_groups[s]) for s in all_groups}
        tot_trees = sum(n_trees.values()) or 1
        protocol = {
            "split_type": "geographic",
            "gene": "HIV-1 Env (nucleotide CDS → AA after ASR)",
            "raw_glob": args.raw_glob,
            "allocation": "greedy largest-first 80/10/10 by seq count over units",
            "unit_assignment": {
                u: assign[u] for u in sorted(assign, key=lambda x: (-unit_counts[x], x))
            },
            "unit_counts": unit_counts,
            "grouping": {
                "group_size": args.group_size,
                "min_group": args.min_group,
                "max_span_years": args.max_span_years,
                "loc_min": args.loc_min,
                "rule": "date-contiguous within one geoloc unit; no region mix",
            },
            "eligibility": stats,
            "n_trees": n_trees,
            "tree_frac": {s: n_trees[s] / tot_trees for s in n_trees},
            "n_leaves": {
                s: sum(g["n_leaves"] for g in all_groups[s]) for s in all_groups
            },
            "groups": all_groups,
            "frame_orf_policy": (
                "Env ATG + signal-like N-term; HXB2-length window; "
                "empty/range dates dropped"
            ),
        }
        (gbase / "SPLIT_PROTOCOL.json").write_text(json.dumps(protocol, indent=2))
        report = []
        for split in ("train", "val", "test"):
            units = sorted({g["unit"] for g in all_groups[split]})
            report.append(
                {
                    "split": split,
                    "n_trees": n_trees[split],
                    "n_leaves": protocol["n_leaves"][split],
                    "tree_pct": round(100 * n_trees[split] / tot_trees, 1),
                    "n_units": len(units),
                    "units": units,
                    "year_min": min((g["year_min"] for g in all_groups[split]), default=None),
                    "year_max": max((g["year_max"] for g in all_groups[split]), default=None),
                }
            )
        (gbase / "SIZE_REPORT.json").write_text(json.dumps(report, indent=2))
        print(f"Wrote {gbase / 'SPLIT_PROTOCOL.json'}")
        print("Geo size report:", json.dumps(report, indent=2))

    print("\nNext: run_all_groups → precompute → train.py")


if __name__ == "__main__":
    main()
