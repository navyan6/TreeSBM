#!/usr/bin/env python3
"""Prepare a COVID Spike clade-holdout split for leaf recovery.

Holds out one clade per tree; writes reduced trees under train/val/test and
held-out leaves under ``heldout/``. Output: ``data/covid_cladeholdout``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
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
HOLDOUT_FRAC_TARGET = 0.15  # preferred holdout size when choosing among clades
MIN_HELDOUT = 2


def load_clade_tsv(paths: list[Path]) -> dict[str, str]:
    """Load seqName -> clade from one or more Nextclade-style TSVs."""
    mapping: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            print(f"  WARNING: clade TSV missing: {path}")
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if not reader.fieldnames:
                continue
            # Nextclade uses seqName; some exports use strain / name
            name_key = next(
                (k for k in ("seqName", "strain", "name", "seq_name")
                 if k in reader.fieldnames),
                None,
            )
            clade_key = next(
                (k for k in ("clade", "clade_nextstrain", "Nextclade_pango",
                             "pango", "pangolin_lineage")
                 if k in reader.fieldnames),
                None,
            )
            if name_key is None or clade_key is None:
                print(f"  WARNING: {path.name} missing seqName/clade columns "
                      f"(have {reader.fieldnames[:8]}...)")
                continue
            n = 0
            for row in reader:
                name = (row.get(name_key) or "").strip().split()[0]
                clade = (row.get(clade_key) or "").strip()
                if not name or not clade or clade in (".", "?", "unassigned"):
                    continue
                mapping[name] = clade
                n += 1
            print(f"  loaded {n} clade labels from {path}")
    return mapping


def write_clade_tsv_via_nextclade(
    raw_fasta: Path, dataset_dir: Path, out_tsv: Path, jobs: int,
) -> Path | None:
    """Run nextclade for clade assignment only (output TSV)."""
    if out_tsv.exists() and out_tsv.stat().st_size > 0:
        return out_tsv
    if not dataset_dir.exists():
        print(f"  nextclade dataset missing at {dataset_dir}")
        return None
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nextclade", "run",
        f"--input-dataset={dataset_dir}",
        f"--output-tsv={out_tsv}",
        f"--jobs={jobs}",
        str(raw_fasta),
    ]
    print(f"  [nextclade] {raw_fasta.name} -> {out_tsv.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  nextclade failed: {result.stderr[-1500:]}")
        return None
    return out_tsv if out_tsv.exists() else None


def chunk_country(records, group_size, min_chunk):
    """Yield fixed-size date-ordered chunks (drop short remainders)."""
    for i in range(0, len(records), group_size):
        chunk = records[i:i + group_size]
        if len(chunk) >= min_chunk:
            yield chunk


def choose_holdout_clade(
    labels: list[str],
    min_heldout: int,
    min_keep: int,
    holdout_clade: str | None,
    target_frac: float,
) -> tuple[str | None, set[int]]:
    """
    Pick which clade indices to hold out.
    labels[i] is clade string for record i (may be '' if unknown).
    Returns (clade_name, index_set) or (None, empty) if impossible.
    """
    n = len(labels)
    by_clade: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        if lab:
            by_clade.setdefault(lab, []).append(i)

    if holdout_clade:
        idxs = by_clade.get(holdout_clade, [])
        if len(idxs) >= min_heldout and (n - len(idxs)) >= min_keep:
            return holdout_clade, set(idxs)
        return None, set()

    # Prefer clades whose size is near target_frac, that leave enough keep leaves.
    candidates = []
    for clade, idxs in by_clade.items():
        n_hold, n_keep = len(idxs), n - len(idxs)
        if n_hold < min_heldout or n_keep < min_keep:
            continue
        # score: closeness to target fraction, prefer smaller-than-majority
        frac = n_hold / n
        if frac >= 0.85:  # don't hold out near-entire tree
            continue
        score = abs(frac - target_frac)
        candidates.append((score, -n_hold, clade, idxs))
    if not candidates:
        return None, set()
    candidates.sort()
    _, _, clade, idxs = candidates[0]
    return clade, set(idxs)


def choose_holdout_year_proxy(
    years: list[int], min_heldout: int, min_keep: int, target_frac: float,
) -> tuple[str | None, set[int]]:
    """Fallback: hold out minority calendar year inside the chunk."""
    labels = [str(y) for y in years]
    name, idxs = choose_holdout_clade(
        labels, min_heldout, min_keep, None, target_frac,
    )
    if name is None:
        return None, set()
    return f"year:{name}", idxs


def write_group(fasta_path: Path, csv_path: Path, records):
    with open(fasta_path, "w") as ff, open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["name", "date"])
        for acc, date_str, _, seq in records:
            ff.write(f">{acc},{date_str}\n{seq}\n")
            w.writerow([acc, date_str])


def main():
    ap = argparse.ArgumentParser(
        description="COVID Spike clade-holdout split for leaf recovery evals."
    )
    ap.add_argument("--out-base", default="data/covid_cladeholdout")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--min-group", type=int, default=MIN_GROUP,
                    help="Minimum leaves remaining for TREE-BUILDING after holdout")
    ap.add_argument("--min-heldout", type=int, default=MIN_HELDOUT)
    ap.add_argument("--holdout-frac-target", type=float, default=HOLDOUT_FRAC_TARGET,
                    help="Prefer holding out a clade near this fraction of the tree")
    ap.add_argument("--holdout-clade", default=None,
                    help="Force this Nextclade clade name when present in a tree")
    ap.add_argument("--mode", choices=("nextclade", "majority-year"), default="nextclade",
                    help="Clade source: nextclade TSV labels, or year-block proxy")
    ap.add_argument("--clade-tsv", nargs="*", default=None,
                    help="Nextclade TSV path(s). Default: auto-discover in raw-dir")
    ap.add_argument("--write-clade-tsv", action="store_true",
                    help="Run nextclade to produce *_clades.tsv before splitting")
    ap.add_argument("--nextclade-dataset", default="data/covid/nextclade_dataset")
    ap.add_argument("--nextclade-jobs", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-trees", type=int, default=None,
                    help="Cap total candidate trees (smoke / dry-run)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw_dir = ROOT / args.raw_dir
    base = ROOT / args.out_base
    raw_sources = sorted(raw_dir.glob("*_covid_seqs.fasta"))
    if not raw_sources:
        print(f"No *_covid_seqs.fasta in {raw_dir}"); return

    # ── clade labels ──────────────────────────────────────────────────────
    clade_map: dict[str, str] = {}
    if args.mode == "nextclade":
        if args.write_clade_tsv:
            dataset_dir = ROOT / args.nextclade_dataset
            for raw_path in raw_sources:
                out_tsv = raw_path.with_name(
                    raw_path.stem.replace("_covid_seqs", "") + "_clades.tsv"
                )
                write_clade_tsv_via_nextclade(
                    raw_path, dataset_dir, out_tsv, args.nextclade_jobs,
                )
        if args.clade_tsv:
            tsv_paths = [ROOT / p for p in args.clade_tsv]
        else:
            tsv_paths = sorted(raw_dir.glob("*_clades.tsv"))
            tsv_paths += sorted(raw_dir.glob("*_nextclade.tsv"))
            # also accept nextclade default names next to aligned genomes
            tsv_paths += sorted(raw_dir.glob("*_covid_seqs_aligned*.tsv"))
        # dedup
        seen_p = set()
        uniq = []
        for p in tsv_paths:
            if p.resolve() not in seen_p:
                seen_p.add(p.resolve())
                uniq.append(p)
        print(f"Loading clade labels from {len(uniq)} TSV(s)...")
        clade_map = load_clade_tsv(uniq)
        if not clade_map:
            print(
                "WARNING: no clade labels found. Falling back to --mode majority-year. "
                "On cluster: python scripts/prepare_covid_cladeholdout.py --write-clade-tsv"
            )
            args.mode = "majority-year"

    # ── load spike records by country ─────────────────────────────────────
    by_country: dict[str, list] = {}
    seen_acc: set[str] = set()
    n_spike = 0
    n_missing_spike = 0
    for raw_path in raw_sources:
        spike_path = raw_path.with_name(
            raw_path.stem.replace("_covid_seqs", "") + "_spike.fasta"
        )
        if not spike_path.exists():
            print(f"WARNING: missing {spike_path.name} — run spike extract first.")
            n_missing_spike += 1
            continue
        recs = load_spike_records(raw_path, spike_path)
        for acc, date_raw, country, seq in recs:
            if acc in seen_acc:
                continue
            try:
                date_str, sort_key = parse_date(date_raw)
            except (ValueError, IndexError):
                continue
            seen_acc.add(acc)
            by_country.setdefault(country, []).append(
                (acc, date_str, sort_key, seq)
            )
            n_spike += 1
        print(f"{raw_path.name}: +spike -> running total {n_spike}")

    if n_missing_spike == len(raw_sources) or n_spike == 0:
        print("No spike FASTAs available — cannot build clade-holdout split.")
        return

    min_chunk = max(
        args.min_group + args.min_heldout,
        math.ceil(args.min_group / max(1e-6, 1 - args.holdout_frac_target)),
    )

    # ── build candidate trees ─────────────────────────────────────────────
    all_chunks = []  # list of list[(acc, date_str, sort_key, seq)]
    for country, recs in by_country.items():
        recs = sorted(recs, key=lambda r: r[2])
        for chunk in chunk_country(recs, args.group_size, min_chunk):
            all_chunks.append((country, chunk))
    rng.shuffle(all_chunks)
    if args.max_trees is not None:
        all_chunks = all_chunks[:args.max_trees]
    print(f"{len(all_chunks)} candidate trees "
          f"(min {min_chunk} leaves so >= {args.min_group} remain after holdout)")

    # ── random tree-level train/val/test ──────────────────────────────────
    n = len(all_chunks)
    if n < 3:
        print(f"Need >=3 candidate trees, got {n}"); return
    n_test = max(1, int(n * args.test_frac))
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val - n_test
    split_chunks = {
        "train": all_chunks[:n_train],
        "val": all_chunks[n_train:n_train + n_val],
        "test": all_chunks[n_train + n_val:],
    }

    prefix_map = {
        "train": "covidchtrain", "val": "covidchval", "test": "covidchtest",
    }
    held_dir = base / "heldout"
    held_dir.mkdir(parents=True, exist_ok=True)
    for old in held_dir.glob("covidch_*_heldout.*"):
        old.unlink()

    protocol_trees = []
    skipped_no_clade = 0

    for split, chunks in split_chunks.items():
        out_dir = base / split
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = prefix_map[split]
        for old in out_dir.glob(f"{prefix}_group_*"):
            old.unlink()

        kept_trees = 0
        n_kept = n_held = 0
        gi = 1
        for country, chunk in chunks:
            recs = list(chunk)
            if args.mode == "nextclade":
                labels = [clade_map.get(acc, "") for acc, _, _, _ in recs]
                clade_name, hold_idxs = choose_holdout_clade(
                    labels, args.min_heldout, args.min_group,
                    args.holdout_clade, args.holdout_frac_target,
                )
            else:
                years = [sk[0] for _, _, sk, _ in recs]
                clade_name, hold_idxs = choose_holdout_year_proxy(
                    years, args.min_heldout, args.min_group,
                    args.holdout_frac_target,
                )

            if not hold_idxs:
                skipped_no_clade += 1
                continue

            heldout = [recs[i] for i in sorted(hold_idxs)]
            keep = [recs[i] for i in range(len(recs)) if i not in hold_idxs]
            if len(keep) < args.min_group or len(heldout) < args.min_heldout:
                skipped_no_clade += 1
                continue

            write_group(
                out_dir / f"{prefix}_group_{gi:03d}.fasta",
                out_dir / f"{prefix}_group_{gi:03d}.csv",
                keep,
            )
            write_group(
                held_dir / f"covidch_{split}_group_{gi:03d}_heldout.fasta",
                held_dir / f"covidch_{split}_group_{gi:03d}_heldout.csv",
                heldout,
            )
            # also write a small sidecar with clade name
            with open(held_dir / f"covidch_{split}_group_{gi:03d}_heldout_meta.json", "w") as f:
                json.dump({
                    "split": split, "group": gi, "country": country,
                    "holdout_clade": clade_name,
                    "n_heldout": len(heldout), "n_keep": len(keep),
                    "mode": args.mode,
                }, f, indent=2)

            protocol_trees.append({
                "split": split, "group": gi, "country": country,
                "holdout_clade": clade_name,
                "n_heldout": len(heldout), "n_keep": len(keep),
            })
            n_kept += len(keep)
            n_held += len(heldout)
            kept_trees += 1
            gi += 1

        print(f"=== {split}: {kept_trees} trees -> {out_dir} "
              f"({n_kept} tree leaves, {n_held} held-out) ===")

    protocol = {
        "dataset": "covid",
        "split_type": "clade_holdout",
        "out_base": args.out_base,
        "mode": args.mode,
        "holdout_clade_forced": args.holdout_clade,
        "holdout_frac_target": args.holdout_frac_target,
        "group_size": args.group_size,
        "min_group": args.min_group,
        "n_clade_labels": len(clade_map),
        "skipped_no_viable_clade": skipped_no_clade,
        "protocol": (
            "Hold out one Nextclade clade (or year-proxy) per tree before "
            "tree-building. Train on reduced trees; eval recovers held-out "
            "clade leaves via eval_leaf_holdout.py (same metrics as H1N1 LH)."
        ),
        "trees": protocol_trees,
        "heldout_name_pattern": "covidch_{split}_group_{NNN}_heldout.fasta",
    }
    with open(base / "SPLIT_PROTOCOL.json", "w") as f:
        json.dump(protocol, f, indent=2)
    print(f"Wrote {base / 'SPLIT_PROTOCOL.json'} "
          f"({len(protocol_trees)} trees, skipped={skipped_no_clade})")
    print("\nNext:")
    print("  scripts/run_all_groups.py --data-dir data/covid_cladeholdout/{split} "
          "--prefix covidch{split}")
    print("Next: run_all_groups → precompute_plm/ref_rates → train.py")
    print("  scripts/eval_leaf_holdout.py --data data/covid_cladeholdout "
          "--checkpoint checkpoints/covid_cladeholdout_v1/best.pt --max-seq-len 1280")


if __name__ == "__main__":
    main()
