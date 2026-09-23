#!/usr/bin/env python3
"""Write data/panviral/plm_worklist.tsv: one train/val/test dir per line needing caches."""

from __future__ import annotations

import argparse
from pathlib import Path


def needs_cache(d: Path) -> bool:
    nwks = list(d.glob("group_*_rooted.nwk"))
    if not nwks:
        return False
    for nwk in nwks:
        g = nwk.name.replace("_rooted.nwk", "")
        if not (d / f"{g}_anc_aa.fasta").exists():
            continue
        if not (d / f"{g}_plm.pt").exists():
            return True
        if not (d / f"{g}_ref_rates.pt").exists():
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/panviral"))
    ap.add_argument("--out", type=Path, default=Path("data/panviral/plm_worklist.tsv"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--train-val-only", action="store_true",
                    help="Skip test (shorten wall-clock before pan-total train)")
    args = ap.parse_args()

    splits = ["train", "val"] if args.train_val_only else list(args.splits)
    rows = []
    for manifest in sorted(args.root.glob("*/manifest.json")):
        for sp in splits:
            d = manifest.parent / sp
            if d.is_dir() and needs_cache(d):
                rows.append(str(d))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"wrote {len(rows)} dirs -> {args.out}")


if __name__ == "__main__":
    main()
