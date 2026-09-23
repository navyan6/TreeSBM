#!/usr/bin/env python3
"""
Build pan-total gene lists for TreeSBM shared-head training.

Writes:
  data/panviral_total/genes_train.tsv   gene_id\\tpath
  data/panviral_total/genes_val.tsv
  data/panviral_total/genes_test.tsv
  data/panviral_total/GENE_MANIFEST.json

Includes:
  - panviral_glyco/{h3n2_ha,h1n1_ha,spike,env}
  - data/panviral/<slug> excluding flu/CoV/HIV overlap with glyco

Only dirs with ≥1 complete tree (rooted+anc_aa+plm+ref_rates) are listed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

OVERLAP = re.compile(
    r"influenza|orthomyxo|corona|sars|mers|lentivirus_hum|lentivirus_sim",
    re.I,
)

GLYCO = ("h3n2_ha", "h1n1_ha", "spike", "env")


def _complete_count(d: Path) -> int:
    n = 0
    if not d.is_dir():
        return 0
    for nwk in d.glob("group_*_rooted.nwk"):
        stem = nwk.name.replace("_rooted.nwk", "")
        if (
            (d / f"{stem}_anc_aa.fasta").exists()
            and (d / f"{stem}_plm.pt").exists()
            and (d / f"{stem}_ref_rates.pt").exists()
        ):
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glyco-root", type=Path, default=Path("data/panviral_glyco"))
    ap.add_argument("--ncbi-root", type=Path, default=Path("data/panviral"))
    ap.add_argument("--out-root", type=Path, default=Path("data/panviral_total"))
    ap.add_argument("--keep-overlap", action="store_true",
                    help="Keep NCBI flu/CoV/HIV slugs (default: drop)")
    args = ap.parse_args()

    rows = {"train": [], "val": [], "test": []}
    meta = {"glyco": {}, "ncbi": {}, "dropped_overlap": []}

    for gene in GLYCO:
        for sp in rows:
            d = args.glyco_root / gene / sp
            n = _complete_count(d)
            if n:
                rows[sp].append((gene, str(d), n))
        meta["glyco"][gene] = {
            sp: _complete_count(args.glyco_root / gene / sp) for sp in rows
        }

    for manifest in sorted(args.ncbi_root.glob("*/manifest.json")):
        slug = manifest.parent.name
        if not args.keep_overlap and OVERLAP.search(slug):
            meta["dropped_overlap"].append(slug)
            continue
        entry = {}
        for sp in rows:
            d = manifest.parent / sp
            n = _complete_count(d)
            entry[sp] = n
            if n:
                rows[sp].append((slug, str(d), n))
        if entry.get("train", 0) > 0:
            meta["ncbi"][slug] = entry

    args.out_root.mkdir(parents=True, exist_ok=True)
    for sp, items in rows.items():
        out = args.out_root / f"genes_{sp}.tsv"
        with out.open("w") as fh:
            for gid, path, n in items:
                fh.write(f"{gid}\t{path}\t{n}\n")
        print(f"{sp}: {len(items)} gene dirs → {out}")

    meta["counts"] = {sp: len(rows[sp]) for sp in rows}
    meta["train_trees"] = sum(n for _, _, n in rows["train"])
    meta["val_trees"] = sum(n for _, _, n in rows["val"])
    meta["test_trees"] = sum(n for _, _, n in rows["test"])
    (args.out_root / "GENE_MANIFEST.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["counts"], indent=2))
    print(f"trees train/val/test = {meta['train_trees']}/{meta['val_trees']}/{meta['test_trees']}")
    print(f"dropped overlap slugs: {len(meta['dropped_overlap'])}")


if __name__ == "__main__":
    main()
