#!/usr/bin/env python3
"""
Carve a non-empty val split for panflu_forecast_* dirs that were prepared with
empty val (dead calendar branch in older prepare_panflu_forecast.py).

Moves complete train groups (rooted + anc_aa + bl) into val, preferring later
seasons and balancing subtypes when pflutrain_group_*.csv tags are present.

Usage:
  python scripts/fix_panflu_val_split.py --data data/panflu_forecast_h3n2_cal
  python scripts/fix_panflu_val_split.py --data data/panflu_forecast_h3n2_cal --frac 0.1 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


GROUP_SUFFIXES = (
    "_rooted.nwk",
    "_anc_aa.fasta",
    "_bl.json",
    "_anc_nt.fasta",
    "_plm.pt",
    "_ref_rates.pt",
    "_meta.csv",
    "_clean.fasta",
    "_aligned.fasta",
    "_tree.nwk",
    "_aligned_mutations.json",
    "_trained_emb.pt",
)


def _subtype_season(train_dir: Path, g: int) -> tuple[str, str]:
    for pattern in (f"*group_{g:03d}.csv", f"group_{g:03d}_meta.csv"):
        for csv_path in sorted(train_dir.glob(pattern)):
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            st = (rows[0].get("subtype") or rows[0].get("gene_id") or "unknown").strip().lower()
            season = (rows[0].get("season") or rows[0].get("date") or "").strip()
            return st or "unknown", season
    return "unknown", ""


def _complete_groups(train_dir: Path) -> list[int]:
    groups = []
    for nwk in sorted(train_dir.glob("group_*_rooted.nwk")):
        g = int(nwk.stem.split("_")[1])
        if (train_dir / f"group_{g:03d}_anc_aa.fasta").exists() and (
            train_dir / f"group_{g:03d}_bl.json"
        ).exists():
            groups.append(g)
    return groups


def _move_group(src: Path, dst: Path, g: int, dry_run: bool) -> int:
    moved = 0
    for suf in GROUP_SUFFIXES:
        sp = src / f"group_{g:03d}{suf}"
        if not sp.exists():
            continue
        dp = dst / sp.name
        if dry_run:
            print(f"  DRY would move {sp.name}")
        else:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(str(sp), str(dp))
        moved += 1
    # Also move prep-side pflutrain/pfluval/pflutest companions if present.
    for pref in ("pflutrain", "pfluval", "pflutest"):
        for ext in (".csv", ".fasta"):
            sp = src / f"{pref}_group_{g:03d}{ext}"
            if not sp.exists():
                continue
            dp = dst / sp.name
            if dry_run:
                print(f"  DRY would move {sp.name}")
            else:
                shutil.move(str(sp), str(dp))
            moved += 1
    return moved


def pick_val_groups(
    train_dir: Path, groups: list[int], frac: float, min_per_subtype: int
) -> list[int]:
    by_st: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for g in groups:
        st, season = _subtype_season(train_dir, g)
        by_st[st].append((season, g))

    chosen: list[int] = []
    for st, items in sorted(by_st.items()):
        items_sorted = sorted(items)  # season ascending → take from the end
        n = max(min_per_subtype, int(round(len(items_sorted) * frac)))
        n = min(n, max(0, len(items_sorted) - 1))  # keep ≥1 train tree per subtype if possible
        if len(items_sorted) == 1:
            n = 0  # don't empty a singleton subtype from train
        take = [g for _, g in items_sorted[-n:]] if n else []
        print(f"  subtype={st}: train={len(items_sorted)} → move {len(take)} to val "
              f"(seasons {[s for s,_ in items_sorted[-n:]] if n else []})")
        chosen.extend(take)
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="e.g. data/panflu_forecast_h3n2_cal")
    ap.add_argument("--frac", type=float, default=0.10, help="Fraction of each subtype → val")
    ap.add_argument("--min-per-subtype", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Carve even if val already non-empty")
    args = ap.parse_args()

    base = Path(args.data)
    train_dir = base / "train"
    val_dir = base / "val"
    if not train_dir.is_dir():
        raise SystemExit(f"Missing {train_dir}")

    n_val = len(list(val_dir.glob("group_*_rooted.nwk"))) if val_dir.is_dir() else 0
    if n_val > 0 and not args.force:
        print(f"Val already has {n_val} rooted trees — pass --force to carve more.")
        return

    groups = _complete_groups(train_dir)
    print(f"Complete train trees: {len(groups)}")
    chosen = pick_val_groups(train_dir, groups, args.frac, args.min_per_subtype)
    if not chosen:
        raise SystemExit("No groups selected for val — check train contents / frac")

    print(f"Moving {len(chosen)} groups → {val_dir}")
    for g in chosen:
        _move_group(train_dir, val_dir, g, dry_run=args.dry_run)

    # Update SPLIT_PROTOCOL counts if present.
    proto_path = base / "SPLIT_PROTOCOL.json"
    if proto_path.exists() and not args.dry_run:
        proto = json.loads(proto_path.read_text())
        n_train = len(_complete_groups(train_dir))
        n_val = len(_complete_groups(val_dir))
        proto.setdefault("n_groups", {})
        proto["n_groups"]["train"] = n_train
        proto["n_groups"]["val"] = n_val
        proto["val_carved_from_train"] = {
            "n_moved": len(chosen),
            "groups": chosen,
            "frac": args.frac,
        }
        proto_path.write_text(json.dumps(proto, indent=2) + "\n")
        print(f"Updated {proto_path}: train={n_train} val={n_val}")

    print("Done." if not args.dry_run else "Dry-run done (no moves).")


if __name__ == "__main__":
    main()
