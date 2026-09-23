#!/usr/bin/env python3
"""Resplit one-protein tree corpora with the panviral temporal holdout rule."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "panviral"))
from fetch_virus_dataset import choose_cutoff  # noqa: E402

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

DEFAULT_SOURCES = {
    "h3n2_ha": "data/h3n2_epidemic",
    "h1n1_ha": "data/h1n1_epidemic",
    "spike": "data/covid_epidemic",
    "env": "data/hiv_temporal",
}


@dataclass
class GroupUnit:
    gene_id: str
    src_dir: Path
    group_id: int
    tip_years: list[int]
    n_tips: int

    @property
    def max_year(self) -> int:
        return max(self.tip_years)

    @property
    def min_year(self) -> int:
        return min(self.tip_years)


def _years_from_meta(meta: Path) -> list[int]:
    years: list[int] = []
    with open(meta) as f:
        for row in csv.DictReader(f):
            d = (row.get("date") or row.get("Date") or "").strip()
            m = re.match(r"(\d{4})", d)
            if m:
                years.append(int(m.group(1)))
    return years


def _find_meta(split_dir: Path, g: int) -> Path | None:
    for pat in (f"group_{g:03d}_meta.csv", f"*group_{g:03d}.csv"):
        hits = sorted(split_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def collect_groups(gene_id: str, base: Path) -> list[GroupUnit]:
    units: list[GroupUnit] = []
    for split in ("train", "val", "test"):
        d = base / split
        if not d.is_dir():
            continue
        for nwk in sorted(d.glob("group_*_rooted.nwk")):
            g = int(nwk.stem.split("_")[1])
            aa = d / f"group_{g:03d}_anc_aa.fasta"
            bl = d / f"group_{g:03d}_bl.json"
            if not aa.exists() or not bl.exists():
                continue
            meta = _find_meta(d, g)
            if meta is None:
                continue
            years = _years_from_meta(meta)
            if not years:
                continue
            units.append(
                GroupUnit(
                    gene_id=gene_id,
                    src_dir=d,
                    group_id=g,
                    tip_years=years,
                    n_tips=len(years),
                )
            )
    return units


def assign_splits(
    units: list[GroupUnit],
    holdout_frac: float,
    val_frac: float,
    window: int,
    fixed_cutoff: int | None,
) -> tuple[int, dict[str, list[GroupUnit]]]:
    # Tip-weighted records for the same choose_cutoff as panviral fetch.
    records = [{"year": y} for u in units for y in u.tip_years]
    cutoff = choose_cutoff(records, fixed_cutoff, holdout_frac, window)

    train_u: list[GroupUnit] = []
    test_u: list[GroupUnit] = []
    for u in units:
        if u.min_year > cutoff:
            test_u.append(u)
        elif u.max_year <= cutoff:
            train_u.append(u)
        else:
            # Straddles cutoff — keep holdout honest.
            test_u.append(u)

    train_u.sort(key=lambda u: (u.max_year, u.group_id))
    n_val = int(round(len(train_u) * val_frac))
    n_val = min(max(n_val, 1 if len(train_u) >= 10 else 0), max(0, len(train_u) - 1))
    if n_val:
        val_u = train_u[-n_val:]
        train_u = train_u[:-n_val]
    else:
        val_u = []

    return cutoff, {"train": train_u, "val": val_u, "test": test_u}


def _link_group(unit: GroupUnit, dst_dir: Path, new_id: int, dry_run: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for suf in GROUP_SUFFIXES:
        src = unit.src_dir / f"group_{unit.group_id:03d}{suf}"
        if not src.exists():
            continue
        dst = dst_dir / f"group_{new_id:03d}{suf}"
        if dry_run:
            continue
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=Path("data/panviral_glyco"))
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of pre-cutoff trees carved to val (latest years)")
    ap.add_argument("--window", type=int, default=1,
                    help="year bin for cutoff snap; 1 = no snap (groups are not "
                         "uniform 2y bins in the old epidemic corpora)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="GENE=PATH",
        help="override/extra gene source, e.g. spike=data/covid_epidemic",
    )
    args = ap.parse_args()

    sources = dict(DEFAULT_SOURCES)
    for item in args.source:
        if "=" not in item:
            raise SystemExit(f"--source needs gene=path, got {item!r}")
        gene, path = item.split("=", 1)
        sources[gene.strip()] = path.strip()

    protocol: dict = {
        "split_type": "temporal_per_virus_holdout",
        "holdout_frac": args.holdout_frac,
        "val_frac": args.val_frac,
        "window": args.window,
        "note": "Whole trees only; one protein per tree. Cutoff via panviral "
                "choose_cutoff on tip years. Straddling groups → test.",
        "genes": {},
    }

    print(f"out_root={args.out_root}  holdout={args.holdout_frac}  val={args.val_frac}")
    for gene_id, rel in sources.items():
        base = Path(rel)
        if not base.is_dir():
            print(f"SKIP {gene_id}: missing {base}")
            continue
        units = collect_groups(gene_id, base)
        if not units:
            print(f"SKIP {gene_id}: no complete groups in {base}")
            continue
        cutoff, splits = assign_splits(
            units, args.holdout_frac, args.val_frac, args.window, None
        )
        print(
            f"{gene_id}: source={base}  groups={len(units)}  cutoff≤{cutoff}  "
            f"train={len(splits['train'])} val={len(splits['val'])} "
            f"test={len(splits['test'])}  "
            f"year_span={min(u.min_year for u in units)}-{max(u.max_year for u in units)}"
        )
        gene_proto = {
            "source": str(base),
            "train_cutoff": cutoff,
            "n_groups_in": len(units),
            "counts": {k: len(v) for k, v in splits.items()},
            "year_range": [min(u.min_year for u in units), max(u.max_year for u in units)],
            "mapping": {},
        }
        for split_name, group_list in splits.items():
            dst = args.out_root / gene_id / split_name
            if not args.dry_run:
                if dst.exists():
                    for p in dst.iterdir():
                        if p.is_symlink() or p.is_file():
                            p.unlink()
                dst.mkdir(parents=True, exist_ok=True)
            for new_id, unit in enumerate(group_list, start=1):
                _link_group(unit, dst, new_id, args.dry_run)
                gene_proto["mapping"][f"{split_name}/group_{new_id:03d}"] = {
                    "src": f"{unit.src_dir}/group_{unit.group_id:03d}",
                    "min_year": unit.min_year,
                    "max_year": unit.max_year,
                    "n_tips": unit.n_tips,
                }
        protocol["genes"][gene_id] = gene_proto

    if not args.dry_run:
        args.out_root.mkdir(parents=True, exist_ok=True)
        out = args.out_root / "SPLIT_PROTOCOL.json"
        out.write_text(json.dumps(protocol, indent=2) + "\n")
        print(f"wrote {out}")

        # Convenience train recipe snippet
        lines = ["# panviral glyco multi-gene (temporal per virus)", "python scripts/train.py \\"]
        for gene in protocol["genes"]:
            lines.append(f"  --gene-data {gene}={args.out_root}/{gene}/train \\")
            lines.append(f"  --gene-val-data {gene}={args.out_root}/{gene}/val \\")
            lines.append(f"  --gene-test-data {gene}={args.out_root}/{gene}/test \\")
        lines.append("  --balanced-gene-sample --panviral-shared-head \\")
        lines.append("  --ckpt-dir checkpoints/panviral_glyco_v1")
        recipe = args.out_root / "TRAIN_CMD.sh"
        recipe.write_text("\n".join(lines) + "\n")
        print(f"wrote {recipe}")
    else:
        print(json.dumps({g: protocol["genes"][g]["counts"] | {"cutoff": protocol["genes"][g]["train_cutoff"]}
                          for g in protocol["genes"]}, indent=2))


if __name__ == "__main__":
    main()
