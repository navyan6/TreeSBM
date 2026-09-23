#!/usr/bin/env python3
"""Rebuild ``SPLIT_PROTOCOL.json`` from existing group CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.epidemic_split_common import nh_flu_season

GROUP_RE = re.compile(r".*_group_(\d+)\.csv$", re.I)


def _scan_split(split_dir: Path) -> tuple[int, int, Counter, Counter]:
    n_groups = 0
    n_seqs = 0
    years: Counter = Counter()
    seasons: Counter = Counter()
    if not split_dir.is_dir():
        return n_groups, n_seqs, years, seasons
    for csv_path in sorted(split_dir.glob("*_group_*.csv")):
        if not GROUP_RE.match(csv_path.name):
            continue
        n_groups += 1
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_seqs += 1
                date = (row.get("date") or "").strip()
                if len(date) >= 4 and date[:4].isdigit():
                    y = int(date[:4])
                    years[y] += 1
                    if len(date) >= 7 and date[5:7].isdigit():
                        seasons[nh_flu_season(y, int(date[5:7]))] += 1
                    else:
                        seasons[nh_flu_season(y, 7)] += 1
                if row.get("season"):
                    seasons[row["season"]] += 1
    return n_groups, n_seqs, years, seasons


def infer_defaults(data_dir: Path) -> dict:
    name = data_dir.name
    if name in ("h3n2", "h3n2_epidemic", "h3n2_temporal_forecast"):
        return {
            "dataset": "h3n2",
            "split_type": "epidemic" if "epidemic" in name else "temporal",
            "train_end_year": 2022,
            "val_year": 2023,
            "test_start_year": 2024,
            "test_end_year": 2024,
        }
    if name in ("h1n1", "h1n1_epidemic"):
        return {
            "dataset": "h1n1",
            "split_type": "epidemic" if "epidemic" in name else "geographic",
        }
    if name == "covid" or name == "covid_epidemic":
        return {
            "dataset": "covid",
            "split_type": "epidemic" if "epidemic" in name else "geographic",
            "train_end_year": 2022,
            "val_year": 2023,
            "test_start_year": 2024,
            "test_end_year": 2025,
        }
    if name == "hiv_geo":
        return {"dataset": "hiv", "split_type": "geographic", "gene": "HIV-1 Env"}
    if name == "hiv_temporal":
        return {"dataset": "hiv", "split_type": "temporal", "gene": "HIV-1 Env"}
    if name == "filo_l":
        return {"dataset": "filo_l", "split_type": "outbreak", "track": "A"}
    return {"dataset": name, "split_type": "unknown"}


def reconstruct(data_dir: Path, force: bool = False) -> dict | None:
    proto_path = data_dir / "SPLIT_PROTOCOL.json"
    if proto_path.is_file() and not force:
        return json.loads(proto_path.read_text())

    counts: dict[str, int] = {}
    n_groups: dict[str, int] = {}
    year_hist: dict[str, dict] = {}
    season_hist: dict[str, dict] = {}

    for split in ("train", "val", "test"):
        ng, ns, years, seasons = _scan_split(data_dir / split)
        counts[split] = ns
        n_groups[split] = ng
        year_hist[split] = dict(sorted(years.items()))
        season_hist[split] = dict(sorted(seasons.items()))

    if sum(counts.values()) == 0 and sum(n_groups.values()) == 0:
        print(f"SKIP {data_dir}: no group CSVs")
        return None

    protocol = {
        **infer_defaults(data_dir),
        "out_base": str(data_dir.relative_to(ROOT)) if data_dir.is_relative_to(ROOT) else str(data_dir),
        "reconstructed_from_csv": True,
        "note": "Rebuilt from group CSVs; original prep metadata may be incomplete.",
        "counts": counts,
        "n_groups": n_groups,
        "year_hist": year_hist,
        "season_hist": season_hist,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    proto_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Wrote {proto_path}  counts={counts}  n_groups={n_groups}")
    return protocol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "dirs",
        nargs="*",
        default=[
            "data/h3n2",
            "data/h1n1",
            "data/covid",
            "data/hiv_geo",
            "data/hiv_temporal",
            "data/covid_epidemic",
            "data/h3n2_epidemic",
            "data/h1n1_epidemic",
            "data/filo_l",
        ],
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for d in args.dirs:
        reconstruct(ROOT / d, force=args.force)


if __name__ == "__main__":
    main()
