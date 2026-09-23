#!/usr/bin/env python3
"""
Aggregate long-format benchmark results into summary tables.

Input: long-format CSV (one row per method × track × N × root, metrics already
reduced over the K samples). Output: per-(track, N) CSV + LaTeX with mean ± SE
(or bootstrap 95% CI half-width) over roots, plus a main table per track. Pure
csv + numpy (no pandas).

    python benchmarks/make_table.py --results benchmarks/results/results.csv \
        --out benchmarks/results/tables --ci se
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["tree_kl", "split_kl", "rf", "quartet", "branch_w_all", "terminal_edit"]
METRIC_LABELS = {
    "tree_kl": "Tree-KL", "split_kl": "Split-KL", "rf": "RF",
    "quartet": "Quartet", "branch_w_all": "Branch-W1", "terminal_edit": "Term-edit",
}
METHOD_ORDER = [
    "neutral_bd", "empirical_bd", "plm_prior", "artreeformer_adapted",
    "phylovae_adapted", "phylaflow", "phylaflow_adapted", "treesbm",
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _agg(vals: np.ndarray, ci: str):
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    m = float(vals.mean())
    if ci == "boot" and len(vals) > 1:
        boot = [np.random.choice(vals, len(vals), replace=True).mean() for _ in range(1000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return m, float((hi - lo) / 2)
    se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return m, se


def summarize(rows: list[dict], ci: str) -> list[dict]:
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    out = []
    for method in [m for m in METHOD_ORDER if m in by_method] + \
                  [m for m in by_method if m not in METHOD_ORDER]:
        sub = by_method[method]
        row = {"method": method, "n_roots": len({r["root_id"] for r in sub})}
        for met in METRICS:
            m, u = _agg(np.array([_f(r.get(met)) for r in sub]), ci)
            row[met], row[met + "_unc"] = m, u
        out.append(row)
    return out


def to_latex(summary: list[dict], caption: str) -> str:
    header = " & ".join(["Method"] + [METRIC_LABELS[m] + r" $\downarrow$" for m in METRICS])
    lines = [r"\begin{table}[t]\centering", f"\\caption{{{caption}}}",
             r"\begin{tabular}{l" + "c" * len(METRICS) + "}", r"\toprule",
             header + r" \\", r"\midrule"]
    for r in summary:
        cells = [r["method"].replace("_", r"\_")]
        for m in METRICS:
            v, u = r.get(m, float("nan")), r.get(m + "_unc", float("nan"))
            cells.append(f"{v:.3f} $\\pm$ {u:.3f}" if v == v else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def write_csv(summary: list[dict], path: Path):
    if not summary:
        return
    cols = ["method", "n_roots"] + [c for m in METRICS for c in (m, m + "_unc")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in summary:
            w.writerow({k: row.get(k, "") for k in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="benchmarks/results/tables")
    ap.add_argument("--ci", choices=["se", "boot"], default="se")
    args = ap.parse_args()

    with open(args.results) as f:
        rows = [r for r in csv.DictReader(f) if int(_f(r.get("valid", 0))) > 0]
    if not rows:
        print("No valid rows in results — nothing to aggregate."); return

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for track in sorted({r["track"] for r in rows}):
        dtrack = [r for r in rows if r["track"] == track]
        for N in sorted({int(_f(r["N"])) for r in dtrack}):
            s = summarize([r for r in dtrack if int(_f(r["N"])) == N], args.ci)
            write_csv(s, outdir / f"table_{track}_N{N}.csv")
            (outdir / f"table_{track}_N{N}.tex").write_text(
                to_latex(s, f"Tree generation on held-out roots ({track}, N={N})"))
        write_csv(summarize(dtrack, args.ci), outdir / f"table_{track}_main.csv")
        (outdir / f"table_{track}_main.tex").write_text(
            to_latex(summarize(dtrack, args.ci), f"Tree generation on held-out roots ({track})"))
        print(f"[{track}] wrote tables for N in {sorted({int(_f(r['N'])) for r in dtrack})} + main")


if __name__ == "__main__":
    main()
