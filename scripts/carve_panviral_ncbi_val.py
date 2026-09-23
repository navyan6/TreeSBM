#!/usr/bin/env python3
"""
Carve temporal val from NCBI panviral train splits (train/test already exist).

For each virus under data/panviral/<slug>/ with train trees, move the latest
``--frac`` of complete train groups (by max tip year) into val/. Does not
touch test/. Skips viruses that already have a non-empty val unless --force.

Usage:
  python scripts/carve_panviral_ncbi_val.py
  python scripts/carve_panviral_ncbi_val.py --frac 0.1 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
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
)


def _max_year(meta: Path) -> int:
    years = []
    with open(meta) as f:
        for row in csv.DictReader(f):
            d = (row.get("date") or row.get("Date") or "").strip()
            m = re.match(r"(\d{4})", d)
            if m:
                years.append(int(m.group(1)))
    return max(years) if years else -1


def _complete(train: Path) -> list[tuple[int, int]]:
    out = []
    for nwk in sorted(train.glob("group_*_rooted.nwk")):
        g = int(nwk.stem.split("_")[1])
        if not (train / f"group_{g:03d}_anc_aa.fasta").exists():
            continue
        if not (train / f"group_{g:03d}_bl.json").exists():
            continue
        meta = train / f"group_{g:03d}_meta.csv"
        if not meta.exists():
            continue
        out.append((g, _max_year(meta)))
    return out


def _move_group(src: Path, dst: Path, g: int, dry: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for suf in GROUP_SUFFIXES:
        sp = src / f"group_{g:03d}{suf}"
        if not sp.exists():
            continue
        dp = dst / sp.name
        if dry:
            continue
        if dp.exists() or dp.is_symlink():
            dp.unlink()
        shutil.move(str(sp), str(dp))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/panviral"))
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    summary = {}
    for manifest in sorted(args.root.glob("*/manifest.json")):
        slug = manifest.parent.name
        train = manifest.parent / "train"
        val = manifest.parent / "val"
        if not train.is_dir():
            continue
        existing = list(val.glob("group_*_rooted.nwk")) if val.is_dir() else []
        if existing and not args.force:
            summary[slug] = {"skipped": True, "val_existing": len(existing)}
            continue
        groups = _complete(train)
        # Need ≥2 complete trees to keep ≥1 in train after carving.
        if len(groups) < 2:
            summary[slug] = {"skipped": True, "reason": "too_few_train", "n": len(groups)}
            continue
        groups.sort(key=lambda t: (t[1], t[0]))
        n_val = max(1, int(round(len(groups) * args.frac)))
        n_val = min(n_val, len(groups) - 1)
        take = [g for g, _ in groups[-n_val:]]
        print(f"{slug}: train={len(groups)} → val {len(take)} (years "
              f"{[y for _, y in groups[-n_val:]]})")
        for g in take:
            _move_group(train, val, g, args.dry_run)
        summary[slug] = {
            "train_before": len(groups),
            "val_moved": len(take),
            "val_years": [y for _, y in groups[-n_val:]],
        }

    out = args.root / "VAL_CARVE.json"
    if not args.dry_run:
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {out}")
    n_carved = sum(1 for v in summary.values() if v.get("val_moved"))
    print(f"carved val for {n_carved} viruses")


if __name__ == "__main__":
    main()
