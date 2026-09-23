#!/usr/bin/env python3
"""Synthetic smoke demo for absolute-e coverage and clade recall.

No real data or GPU required; useful for checking metric plumbing offline.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tree_state import TreeState
from benchmarks.coverage_curves import (
    score_pool, emerging_clade_fingerprints, curve_fields, k_grid,
)


def _make_gt() -> tuple[TreeState, str, list[str]]:
    root = "AAAAAAAAAA"
    # Two emerging clades sharing fingerprints at pos 0 and pos 1
    edges = [
        ("root", "iA"), ("iA", "A1"), ("iA", "A2"),
        ("root", "iB"), ("iB", "B1"), ("iB", "B2"),
    ]
    seqs = {
        "root": root,
        "iA": "CAAAAAAAAA",
        "A1": "CDAAAAAAAA",
        "A2": "CEAAAAAAAA",
        "iB": "AGAAAAAAAA",
        "B1": "AGHAAAAAAA",
        "B2": "AGIAAAAAAA",
    }
    bl = {e: 0.1 for e in edges}
    gt = TreeState(
        node_ids=list(seqs), root_id="root", edges=edges,
        branch_lengths=bl, node_seqs=seqs,
        active_leaves=["A1", "A2", "B1", "B2"],
    )
    leaves = [seqs[x] for x in ("A1", "A2", "B1", "B2")]
    return gt, root, leaves


def _fake_pools(root: str) -> dict[str, dict[int, list[str]]]:
    """method -> K -> cumulative gen leaf pool (intentionally exclude bare root)."""
    del root  # pools below are absolute sequences
    # "good" method gradually recovers both clade fingerprints
    good = {
        10: ["CXAAAAAAAA", "CDAAAAAAAA"],
        20: ["CXAAAAAAAA", "CDAAAAAAAA", "AGHAAAAAAA", "AGYAAAAAAA"],
        30: ["CXAAAAAAAA", "CDAAAAAAAA", "CEAAAAAAAA",
             "AGHAAAAAAA", "AGIAAAAAAA"],
    }
    # "bad" method mutates at the opposite end of the sequence
    bad = {
        10: ["AAAAAAAAXZ", "AAAAAAAAYZ"],
        20: ["AAAAAAAAXZ", "AAAAAAAAYZ", "AAAAAAAWZZ"],
        30: ["AAAAAAAAXZ", "AAAAAAAAYZ", "AAAAAAAWZZ", "TTTTTTTTTT"],
    }
    return {"good_model": good, "bad_model": bad}


def main():
    out = ROOT / "benchmarks/results/coverage_curves_eabs_synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    e_list = [0, 1, 2, 3, 5]
    gt, root, gt_leaves = _make_gt()
    fps = emerging_clade_fingerprints(gt, root)
    pools = _fake_pools(root)
    ks = [10, 20, 30]
    fields = curve_fields(e_list)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for method, by_k in pools.items():
            for K in ks:
                m = score_pool(
                    gt_leaves, root, by_k[K],
                    eps_frac=0.02, max_gt=60, max_gen=800, seed=0,
                    e_list=e_list, fingerprints=fps,
                )
                row = {
                    "method": method,
                    "N": 4,
                    "K": K,
                    "n_roots": 1,
                    "n_trees_scored": K,
                    "eps_frac": 0.02,
                    "runtime_gen_sec": 0.0,
                    **m,
                }
                w.writerow(row)
                print(
                    f"{method:12s} K={K:2d} "
                    f"cov@ε={m['coverage']:.2f} "
                    f"cov@e0={m['coverage_obs_e0']:.2f} "
                    f"cov@e2={m['coverage_obs_e2']:.2f} "
                    f"frac_gen@e2={m['frac_gen_e2']:.2f} "
                    f"clade_r={m['clade_recall']:.2f} "
                    f"edit={m['mean_min_edit']:.1f}"
                )

    print(f"\nwrote {out}")
    print(f"eligible clade fingerprints: {len(fps)}")
    # Tiny markdown table for quick inspection
    md = ROOT / "benchmarks/results/coverage_curves_eabs_synthetic.md"
    lines = [
        "# Synthetic absolute-e coverage demo",
        "",
        "| method | K | cov@ε | cov@e0 | cov@e2 | frac_gen@e2 | clade_recall | mean_min_edit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    with open(out) as f:
        for row in csv.DictReader(f):
            lines.append(
                f"| {row['method']} | {row['K']} | {float(row['coverage']):.2f} | "
                f"{float(row['coverage_obs_e0']):.2f} | {float(row['coverage_obs_e2']):.2f} | "
                f"{float(row['frac_gen_e2']):.2f} | {float(row['clade_recall']):.2f} | "
                f"{float(row['mean_min_edit']):.1f} |"
            )
    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
