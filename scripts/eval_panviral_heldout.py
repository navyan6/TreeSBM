#!/usr/bin/env python3
"""Held-out-root forecast eval for panviral TreeSBM checkpoints.

Compares TreeSBM to NeutralBD / pLM / ARTreeFormer-adapted on shared roots.
Writes JSON/CSV summaries under the output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.fit_params import fit_params
from benchmarks.heldout.build_examples import build_examples, list_groups, load_tree
from benchmarks.adapters.branch_length import BranchLengthAdapter
from benchmarks.adapters.sequence import evolve_pyvolve
from benchmarks.methods.bd_methods import NeutralBD
from benchmarks.methods.plm_prior import PLMPrior
from benchmarks.methods.topology_prior import TopologyPriorMethod
from benchmarks.methods.treesbm import TreeSBMMethod
from benchmarks.metrics import trees as T
from benchmarks.metrics.matched import sequence_matched_rf, quartet_distance, terminal_edit_distance
from benchmarks.metrics.branch_lengths import branch_length_wasserstein
from benchmarks.run_table import load_external_pools, rebuild_target, ESM, _cuda
from benchmarks import validity as V
from benchmarks.coverage_curves import (
    score_pool,
    emerging_clade_fingerprints,
    DEFAULT_E_LIST,
)

GLYCO_GENES = ("h3n2_ha", "h1n1_ha", "spike", "env")

# Canonical metric keys for the summary JSON / CSV (see module docstring).
SUMMARY_METRIC_KEYS = [
    "valid_frac",
    "rf", "quartet", "branch_w_all", "terminal_edit",
    "sackin_index", "colless_index", "cherry_count",
    "topological_height", "patristic_height",
    "coverage", "mean_min_edit", "mean_min_edit_frac",
    "site_recall", "mut_f1", "mut_recovery", "cons_retention",
    "aa_acc_given_hit", "best_of_k_identity", "clade_recall",
    "runtime_gen_sec",
]


def _nanmean(xs):
    xs = [x for x in xs if x == x and not (isinstance(x, float) and math.isnan(x))]
    return float(mean(xs)) if xs else float("nan")


def discover_genes(args) -> list[tuple[str, Path, Path]]:
    """Return [(gene_id, train_dir, test_dir), ...]."""
    out = []
    if args.panel == "glyco":
        root = ROOT / args.glyco_root
        genes = list(args.genes) if args.genes else list(GLYCO_GENES)
        for g in genes:
            tr, te = root / g / "train", root / g / "test"
            if list_groups(te):
                out.append((g, tr, te))
            else:
                print(f"WARN: skip {g} — no test trees in {te}")
    elif args.panel == "total":
        tsv = ROOT / args.genes_tsv
        if not tsv.exists():
            raise SystemExit(f"missing {tsv} — run scripts/build_panviral_total_genes.py")
        # genes_test.tsv is gene_id\\tpath\\tn; derive train by replacing /test
        for line in tsv.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            gene, path = parts[0], Path(parts[1])
            if not path.is_absolute():
                path = ROOT / path
            # path may be .../gene/test or .../gene/train in total tsv — normalize
            if path.name == "test":
                te, tr = path, path.parent / "train"
            elif path.name == "train":
                tr, te = path, path.parent / "test"
            else:
                te, tr = path / "test", path / "train"
            if list_groups(te):
                out.append((gene, tr, te))
    else:
        raise SystemExit(f"unknown panel {args.panel}")
    if args.max_genes and len(out) > args.max_genes:
        out = out[: args.max_genes]
    return out


def build_methods(args, params, esm, train_dir: Path, gene_id: str | None = None):
    methods = []
    want = set(args.methods)
    birth, death = params["birth"], params["death"]

    if "neutral_bd" in want:
        methods.append(NeutralBD(birth, death))

    if "plm_prior" in want:
        if esm is None:
            print("WARN: plm_prior skipped (no ESM)")
        else:
            methods.append(PLMPrior(
                esm.lm_logits, birth, death, params.get("subst_scale", 1.0)))

    if "artreeformer_adapted" in want:
        pool_dir = ROOT / "benchmarks/external_pools/sampled"
        prefix = getattr(args, "artreeformer_prefix", None) or "artreeformer"
        pool_by_N = load_external_pools(pool_dir, prefix, [args.N])
        if not pool_by_N:
            print("WARN: artreeformer_adapted skipped — no pools at "
                  f"{pool_dir}/{prefix}_N{args.N}.nwk "
                  "(export topologies and sample an ARTreeFormer pool first)")
        elif not list_groups(train_dir):
            print(f"WARN: artreeformer_adapted skipped — no train trees in {train_dir}")
        else:
            train_trees = [load_tree(train_dir, g) for g in list_groups(train_dir)]
            bl_adapter = BranchLengthAdapter().fit(train_trees)
            seq_fn = lambda topo, root_seq, seed: evolve_pyvolve(
                topo, root_seq, model=args.seq_model, seed=seed)
            methods.append(TopologyPriorMethod(
                "artreeformer_adapted", pool_by_N, bl_adapter, seq_fn))

    if "treesbm" in want:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        if not ckpt.exists():
            print(f"WARN: treesbm skipped — missing checkpoint {ckpt}")
        else:
            methods.append(TreeSBMMethod(
                str(ckpt),
                n_steps=args.n_steps,
                max_seq_len=args.max_seq_len,
                gene_id=gene_id,
            ))

    zs = getattr(args, "zeroshot_checkpoint", "") or ""
    if zs and "treesbm_zeroshot" in want:
        zpath = Path(zs)
        if not zpath.is_absolute():
            zpath = ROOT / zpath
        if not zpath.exists():
            print(f"WARN: treesbm_zeroshot skipped — missing {zpath}")
        else:
            m = TreeSBMMethod(
                str(zpath),
                n_steps=args.n_steps,
                max_seq_len=args.max_seq_len,
                gene_id=gene_id,
            )
            m.name = "treesbm_zeroshot"
            methods.append(m)
    return methods


def _topo_stats(gen) -> dict:
    try:
        return {
            "sackin_index": float(T.sackin_index(gen)),
            "colless_index": float(T.colless_index(gen)),
            "cherry_count": float(T.cherry_count(gen)),
            "topological_height": float(T.topological_height(gen)),
            "patristic_height": float(T.patristic_height(gen)),
        }
    except Exception:
        return {k: float("nan") for k in (
            "sackin_index", "colless_index", "cherry_count",
            "topological_height", "patristic_height")}


def _qd(a, b):
    try:
        return quartet_distance(a, b)
    except Exception:
        return float("nan")


def eval_gene(args, gene: str, train_dir: Path, test_dir: Path, esm) -> dict:
    print(f"\n=== gene={gene} train={train_dir} test={test_dir} ===")
    params = fit_params(train_dir)
    methods = build_methods(args, params, esm, train_dir, gene_id=gene)
    if not methods:
        return {"gene": gene, "error": "no_methods", "methods": {}}

    examples = build_examples(
        test_dir, N=args.N, seed=args.seed,
        max_roots_per_tree=args.max_roots_per_tree,
    )
    if args.max_roots and len(examples) > args.max_roots:
        rng = random.Random(args.seed)
        examples = rng.sample(examples, args.max_roots)
    print(f"  held-out roots: {len(examples)}  methods: {[m.name for m in methods]}")

    gene_out = {"gene": gene, "n_roots": len(examples), "methods": {}}
    sample_rows = []

    for method in methods:
        per = {
            "valid": [], "rf": [], "quartet": [], "branch_w_all": [], "terminal_edit": [],
            "sackin_index": [], "colless_index": [], "cherry_count": [],
            "topological_height": [], "patristic_height": [],
            "pool_metrics": [], "runtime_gen_sec": [],
            "n_attempted": 0, "n_valid": 0,
        }
        for ri, ex in enumerate(examples):
            target = rebuild_target(ex)
            root_seq, N, H = ex["root_seq"], ex["N"], ex["H"]
            root_id = ex["root_id"]
            gt_leaves = [
                target.node_seqs[l] for l in T.leaf_labels(target)
                if target.node_seqs.get(l)
            ]
            fps = emerging_clade_fingerprints(target, root_seq)

            gens = []
            t0 = time.time()
            for k in range(args.K):
                seed = args.seed + 1000 * ri + k
                per["n_attempted"] += 1
                try:
                    gt = method.generate(root_seq, N, H, seed=seed)
                    gen = gt.tree if hasattr(gt, "tree") else gt
                except Exception as e:
                    sample_rows.append({
                        "gene": gene, "method": method.name, "root_id": root_id,
                        "sample": k, "valid": 0, "reasons": f"gen_error:{type(e).__name__}",
                    })
                    continue
                v = V.validate(gen, root_seq, N, H)
                sample_rows.append({
                    "gene": gene, "method": method.name, "root_id": root_id,
                    "sample": k, "valid": int(v["valid"]),
                    "reasons": "|".join(v.get("reasons") or []),
                })
                if not v["valid"]:
                    continue
                per["n_valid"] += 1
                gens.append(gen)
                per["rf"].append(sequence_matched_rf(gen, target))
                per["quartet"].append(_qd(gen, target))
                per["branch_w_all"].append(
                    branch_length_wasserstein(gen, target)["all"])
                per["terminal_edit"].append(
                    terminal_edit_distance(gen, target)["mean"])
                for kk, vv in _topo_stats(gen).items():
                    per[kk].append(vv)
            per["runtime_gen_sec"].append(time.time() - t0)

            # leaf-pool forecast metrics over all valid gens for this root
            gen_pool = []
            for g in gens:
                gen_pool.extend(
                    g.node_seqs[l] for l in T.leaf_labels(g) if g.node_seqs.get(l)
                )
            pool = score_pool(
                gt_leaves, root_seq, gen_pool,
                eps_frac=args.eps_frac,
                max_gt=args.max_gt_leaves,
                max_gen=args.max_gen_leaves,
                seed=args.seed + ri,
                e_list=args.e_list,
                fingerprints=fps,
            )
            per["pool_metrics"].append(pool)

        summary = {
            "n_attempted": per["n_attempted"],
            "n_valid": per["n_valid"],
            "valid_frac": (per["n_valid"] / per["n_attempted"]
                           if per["n_attempted"] else float("nan")),
            "runtime_gen_sec": _nanmean(per["runtime_gen_sec"]),
        }
        for key in ("rf", "quartet", "branch_w_all", "terminal_edit",
                    "sackin_index", "colless_index", "cherry_count",
                    "topological_height", "patristic_height"):
            summary[key] = _nanmean(per[key])
        # average pool metrics across roots
        if per["pool_metrics"]:
            keys = per["pool_metrics"][0].keys()
            for k in keys:
                summary[k] = _nanmean([m[k] for m in per["pool_metrics"]])
        gene_out["methods"][method.name] = summary
        print(f"  {method.name}: valid={summary['valid_frac']:.3f}  "
              f"rf={summary.get('rf', float('nan')):.3f}  "
              f"cov_e2={summary.get('coverage_obs_e2', float('nan')):.3f}  "
              f"mut_rec={summary.get('mut_recovery', float('nan')):.3f}")

    gene_out["sample_rows"] = sample_rows
    return gene_out


def macro_average(gene_results: list[dict]) -> dict:
    methods = sorted({m for g in gene_results for m in g.get("methods", {})})
    out = {}
    for m in methods:
        keys = set()
        for g in gene_results:
            keys |= set(g.get("methods", {}).get(m, {}).keys())
        keys -= {"n_attempted", "n_valid"}
        out[m] = {}
        for k in sorted(keys):
            out[m][k] = _nanmean([
                g["methods"][m][k]
                for g in gene_results
                if m in g.get("methods", {}) and k in g["methods"][m]
            ])
        out[m]["n_genes"] = sum(1 for g in gene_results if m in g.get("methods", {}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", choices=["glyco", "total"], required=True)
    ap.add_argument("--glyco-root", default="data/panviral_glyco")
    ap.add_argument("--genes-tsv", default="data/panviral_total/genes_test.tsv")
    ap.add_argument("--genes", nargs="+", default=None,
                    help="Optional gene filter (glyco names or total gene_ids)")
    ap.add_argument("--max-genes", type=int, default=None)
    ap.add_argument("--checkpoint", required=True,
                    help="TreeSBM ckpt (finetuned / main)")
    ap.add_argument("--zeroshot-checkpoint", default="",
                    help="Optional pretrain ckpt scored as method treesbm_zeroshot")
    ap.add_argument("--artreeformer-prefix", default="artreeformer",
                    help="Pool file prefix under benchmarks/external_pools/sampled/")
    ap.add_argument("--methods", nargs="+",
                    default=["treesbm", "plm_prior", "artreeformer_adapted", "neutral_bd"])
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K", type=int, default=20,
                    help="# trees generated per held-out root")
    ap.add_argument("--max-roots", type=int, default=20)
    ap.add_argument("--max-roots-per-tree", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1280)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--seq-model", default="JTT")
    ap.add_argument("--eps-frac", type=float, default=0.02)
    ap.add_argument("--e-list", type=int, nargs="+", default=list(DEFAULT_E_LIST[:5]))
    ap.add_argument("--max-gt-leaves", type=int, default=64)
    ap.add_argument("--max-gen-leaves", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-esm", action="store_true")
    ap.add_argument("--out", required=True,
                    help="JSON summary path (also writes .csv next to it)")
    args = ap.parse_args()

    genes = discover_genes(args)
    if not genes:
        raise SystemExit("no genes with test trees found")
    print(f"panel={args.panel}  n_genes={len(genes)}  "
          f"methods={args.methods}  N={args.N} K={args.K}")

    esm = None
    if not args.no_esm and any(m in args.methods for m in ("plm_prior",)):
        device = "cuda" if _cuda() else "cpu"
        esm = ESM(device, max_len=args.max_seq_len)
        print(f"ESM loaded on {device}")

    gene_results = []
    all_samples = []
    for gene, tr, te in genes:
        gr = eval_gene(args, gene, tr, te, esm)
        all_samples.extend(gr.pop("sample_rows", []))
        gene_results.append(gr)

    macro = macro_average(gene_results)
    payload = {
        "panel": args.panel,
        "checkpoint": args.checkpoint,
        "zeroshot_checkpoint": args.zeroshot_checkpoint,
        "artreeformer_prefix": args.artreeformer_prefix,
        "methods": args.methods,
        "N": args.N,
        "K": args.K,
        "max_roots": args.max_roots,
        "e_list": args.e_list,
        "metrics_documented": SUMMARY_METRIC_KEYS + [
            f"coverage_obs_e{e}" for e in args.e_list
        ] + [
            f"coverage_obs_unique_e{e}" for e in args.e_list
        ] + [
            f"frac_gen_e{e}" for e in args.e_list
        ],
        "macro": macro,
        "per_gene": gene_results,
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")

    # wide CSV: method × metric (macro)
    csv_path = out.with_suffix(".csv")
    metric_cols = sorted({k for m in macro.values() for k in m if k != "n_genes"})
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_genes"] + metric_cols)
        for m, stats in sorted(macro.items()):
            w.writerow([m, stats.get("n_genes", "")] + [
                stats.get(c, "") for c in metric_cols
            ])
    print(f"wrote {csv_path}")

    # validity sample log
    samp_path = out.with_name(out.stem + "_samples.csv")
    if all_samples:
        with samp_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_samples[0].keys()))
            w.writeheader()
            w.writerows(all_samples)
        print(f"wrote {samp_path}")

    print("\n=== MACRO (valid_frac / rf / coverage_obs_e2 / mut_recovery) ===")
    for m, s in sorted(macro.items()):
        print(f"  {m:24s}  valid={s.get('valid_frac', float('nan')):.3f}  "
              f"rf={s.get('rf', float('nan')):.3f}  "
              f"cov_e2={s.get('coverage_obs_e2', float('nan')):.3f}  "
              f"mut={s.get('mut_recovery', float('nan')):.3f}")


if __name__ == "__main__":
    main()
