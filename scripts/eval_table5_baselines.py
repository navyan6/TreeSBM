#!/usr/bin/env python3
"""
Baseline enrichment eval: pLM prior and ARTreeFormer-adapted generators.

Matches enrichment KPIs (site_recall, aa_acc_given_hit, cons_retention,
lit/pmc hotspot frac, EVEscape means) without TreeSBM generation.

Usage:
  python scripts/eval_table5_baselines.py \\
    --data data/covid/test --train-data data/covid/train \\
    --max-seq-len 1280 --evescape data/covid/evescape_spike_rbd.pt \\
    --lit-hotspot-mask results/covid_mutfreq_vs_lit/mut_hotspot_mask_pmc_lit.pt \\
    --methods plm_prior artreeformer_adapted \\
    --out checkpoints/eval_table5_covid_baselines.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.dataset import TreeDataset
from benchmarks.metrics.sequences import positional_recovery
from benchmarks.methods.bd_methods import NeutralBD, EmpiricalBD
from benchmarks.methods.plm_prior import PLMPrior
from benchmarks.methods.topology_prior import TopologyPriorMethod
from benchmarks.adapters.branch_length import BranchLengthAdapter
from benchmarks.adapters.sequence import evolve_pyvolve
from benchmarks.heldout.build_examples import list_groups, load_tree
from benchmarks.fit_params import fit_params
from benchmarks.run_table import load_external_pools, ESM, _cuda
from scripts.eval_evescape_enrichment import (
    mutation_evescape,
    random_baseline_evescape,
    hotspot_mut_frac,
    load_hotspot_mask,
)
from scripts.eval_single_tree import get_leaves, seq_identity, esm_pll_seq


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def _plm_nll_leaves(esm, gen_seqs, max_seq_len, chunk=8):
    """Mean −PLL/pos over generated leaves (same def as eval_evescape_enrichment)."""
    if esm is None or not gen_seqs:
        return float("nan")
    leaf_nlls = []
    for s0 in range(0, len(gen_seqs), chunk):
        chunk_seqs = gen_seqs[s0 : s0 + chunk]
        log_R0 = esm._gl(
            esm.tok, esm.model, esm.aa, chunk_seqs, max_seq_len, esm.device
        )
        for j, seq in enumerate(chunk_seqs):
            pll = esm_pll_seq(log_R0[j], seq, max_seq_len)
            if pll == pll and pll != float("-inf"):
                leaf_nlls.append(-pll)
    return _mean(leaf_nlls)


def load_evescape_tensor(path: str, L: int) -> torch.Tensor:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    scores = blob["scores"] if isinstance(blob, dict) and "scores" in blob else blob
    scores = torch.as_tensor(scores)
    if scores.ndim != 2 or scores.shape[0] != L:
        raise SystemExit(f"EVEscape shape {tuple(scores.shape)} != [{L}, 20]")
    return scores


def build_methods(args, params, esm):
    methods = []
    birth, death = params["birth"], params["death"]
    want = set(args.methods)
    if "neutral_bd" in want:
        methods.append(NeutralBD(birth, death))
    if "empirical_bd" in want:
        methods.append(EmpiricalBD(birth, death, model=args.seq_model))
    if "plm_prior" in want:
        if esm is None:
            raise SystemExit("plm_prior needs ESM")
        methods.append(PLMPrior(
            esm.lm_logits, birth, death, params.get("subst_scale", 1.0)))
    if "artreeformer_adapted" in want:
        pool_dir = ROOT / "benchmarks/external_pools/sampled"
        pool_by_N = load_external_pools(pool_dir, "artreeformer", [args.N])
        if not pool_by_N:
            print(f"WARN: no artreeformer_N{args.N}.nwk under {pool_dir}; skip AR")
        else:
            train_dir = ROOT / args.train_data
            groups = list_groups(train_dir)
            if not groups:
                print(f"WARN: no train trees in {train_dir}; skip AR")
            else:
                train_trees = [load_tree(train_dir, g) for g in groups]
                bl_adapter = BranchLengthAdapter().fit(train_trees)
                seq_fn = lambda topo, root_seq, seed: evolve_pyvolve(
                    topo, root_seq, model=args.seq_model, seed=seed)
                methods.append(TopologyPriorMethod(
                    "artreeformer_adapted", pool_by_N, bl_adapter, seq_fn))
                print(f"artreeformer pools Ns={sorted(pool_by_N)}")
    return methods


def score_gen(
    root_seq, gen_tree, gt_seqs, evescape, lit_mask, L, rng, gt_leaves_sampled,
    esm=None, compute_plm_nll=True,
):
    gen_leaves = get_leaves(gen_tree)
    gen_seqs = [gen_tree.node_seqs[g] for g in gen_leaves]
    gt_sample = rng.sample(gt_seqs, min(gt_leaves_sampled, len(gt_seqs)))
    t_site, t_aa, t_cons, t_id = [], [], [], []
    t_lit_g, t_lit_gt = [], []
    model_ev, gt_ev = [], []
    model_ev_ag, gt_ev_ag = [], []
    for gt_seq in gt_sample:
        best, best_id = None, -1.0
        for gs in gen_seqs:
            idv = seq_identity(gt_seq, gs)
            if idv > best_id:
                best_id, best = idv, gs
        if best is None:
            continue
        r = positional_recovery(root_seq, gt_seq, best)
        t_site.append(r["site_recall"])
        t_aa.append(r["aa_acc_given_hit"])
        t_cons.append(r["cons_retention"])
        t_id.append(best_id)
        if lit_mask is not None:
            t_lit_g.append(hotspot_mut_frac(root_seq, best, lit_mask, L))
            t_lit_gt.append(hotspot_mut_frac(root_seq, gt_seq, lit_mask, L))
        if evescape is not None:
            for gs in gen_seqs:
                sc, _, _ = mutation_evescape(root_seq, gs, evescape, L)
                model_ev.extend(sc)
                if lit_mask is not None:
                    sc_ag, _, _ = mutation_evescape(
                        root_seq, gs, evescape, L, site_mask=lit_mask)
                    model_ev_ag.extend(sc_ag)
            sc_gt, _, _ = mutation_evescape(root_seq, gt_seq, evescape, L)
            gt_ev.extend(sc_gt)
            if lit_mask is not None:
                sc_ag, _, _ = mutation_evescape(
                    root_seq, gt_seq, evescape, L, site_mask=lit_mask)
                gt_ev_ag.extend(sc_ag)
    plm_nll = (
        _plm_nll_leaves(esm, gen_seqs, L)
        if compute_plm_nll else float("nan")
    )
    return {
        "site_recall": _mean(t_site),
        "aa_acc_given_hit": _mean(t_aa),
        "cons_retention": _mean(t_cons),
        "identity": _mean(t_id),
        "lit_hotspot_mut_frac": _mean(t_lit_g),
        "gt_lit_hotspot_mut_frac": _mean(t_lit_gt),
        "model_evescape": _mean(model_ev),
        "gt_evescape": _mean(gt_ev),
        "evescape_mean_antigenic_muts": _mean(model_ev_ag),
        "gt_evescape_mean_antigenic_muts": _mean(gt_ev_ag),
        "plm_nll": plm_nll,
        "gen_leaves": len(gen_seqs),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--max-seq-len", type=int, default=1280)
    ap.add_argument("--evescape", default=None)
    ap.add_argument("--lit-hotspot-mask", default=None)
    ap.add_argument("--methods", nargs="+",
                    default=["plm_prior", "artreeformer_adapted"])
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--H-default", type=float, default=0.02)
    ap.add_argument("--max-trees", type=int, default=20)
    ap.add_argument("--gt-leaves-sampled", type=int, default=30)
    ap.add_argument("--seq-model", default="JTT")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-esm", action="store_true")
    ap.add_argument(
        "--compute-plm-nll", action="store_true", default=True,
        help="Score mean ESM-2 −PLL/pos on generated leaves (default on).",
    )
    ap.add_argument(
        "--no-compute-plm-nll", action="store_true",
        help="Skip pLM NLL scoring.",
    )
    args = ap.parse_args()
    do_plm_nll = bool(args.compute_plm_nll) and not bool(args.no_compute_plm_nll)

    params_path = Path(args.params) if args.params else (
        ROOT / "benchmarks/results" / f"params_{Path(args.train_data).parent.name}.json"
    )
    if not params_path.exists():
        print(f"Fitting BD params on {args.train_data} -> {params_path}")
        params_path.parent.mkdir(parents=True, exist_ok=True)
        params = fit_params(ROOT / args.train_data)
        params_path.write_text(json.dumps(params, indent=2))
    else:
        params = json.loads(params_path.read_text())

    need_esm = (not args.no_esm) and (
        do_plm_nll or "plm_prior" in set(args.methods)
    )
    esm = None
    if need_esm:
        esm = ESM(
            "cuda" if _cuda() else "cpu",
            max_len=args.max_seq_len,
        )
    methods = build_methods(args, params, esm)
    if not methods:
        raise SystemExit("no methods available")

    ds = TreeDataset(str(ROOT / args.data), max_seq_len=args.max_seq_len)
    n = min(args.max_trees, len(ds))
    L = args.max_seq_len
    evescape = load_evescape_tensor(args.evescape, L) if args.evescape else None
    lit_mask, lit_meta = load_hotspot_mask(args.lit_hotspot_mask, L)

    base_ev = random_baseline_evescape(evescape) if evescape is not None else float("nan")
    base_ev_ag = (
        random_baseline_evescape(evescape, lit_mask)
        if evescape is not None and lit_mask is not None else float("nan")
    )

    out = {
        "n_trees": n, "N": args.N, "data": args.data,
        "lit_hotspot_mask": lit_meta,
        "compute_plm_nll": do_plm_nll,
        "methods": {},
    }
    rng = random.Random(args.seed)

    for method in methods:
        print(f"=== {method.name} ===")
        per = []
        for i in range(n):
            batch = ds[i]
            root_id = batch["node_ids"][batch["root_index"]]
            root_seq = batch["seqs"][root_id]
            has_ch = {p for p, _ in batch["edges"]}
            gt_seqs = [batch["seqs"][nid] for nid in batch["node_ids"]
                       if nid not in has_ch and nid in batch["seqs"]]
            if gt_seqs:
                divs = [1.0 - seq_identity(root_seq, g) for g in gt_seqs]
                H = max(args.H_default, sum(divs) / len(divs))
            else:
                H = args.H_default
            try:
                gen = method.generate(root_seq, args.N, H, seed=args.seed + i)
                tree = gen.tree
            except Exception as e:
                print(f"  [{i}] ERROR {e}")
                continue
            row = score_gen(
                root_seq, tree, gt_seqs, evescape, lit_mask, L, rng,
                args.gt_leaves_sampled,
                esm=esm, compute_plm_nll=do_plm_nll,
            )
            row["tree"] = i
            per.append(row)
            nll_s = (
                f" plm_nll={row['plm_nll']:.4f}"
                if row.get("plm_nll") == row.get("plm_nll") else ""
            )
            print(
                f"  [{i+1}/{n}] site_r={row['site_recall']:.3f} "
                f"aa|hit={row['aa_acc_given_hit']:.3f} cons={row['cons_retention']:.3f}"
                f"{nll_s}"
            )

        keys = [
            "site_recall", "aa_acc_given_hit", "cons_retention", "identity",
            "lit_hotspot_mut_frac", "gt_lit_hotspot_mut_frac",
            "model_evescape", "gt_evescape",
            "evescape_mean_antigenic_muts", "gt_evescape_mean_antigenic_muts",
            "plm_nll",
        ]
        summary = {k: _mean([p[k] for p in per]) for k in keys} if per else {}
        if evescape is not None and summary:
            summary["random_baseline_evescape"] = base_ev
            summary["evescape_score_delta"] = (
                summary["model_evescape"] - summary["gt_evescape"]
            )
            if lit_mask is not None:
                summary["random_baseline_evescape_antigenic"] = base_ev_ag
                summary["evescape_score_delta_antigenic"] = (
                    summary["evescape_mean_antigenic_muts"]
                    - summary["gt_evescape_mean_antigenic_muts"]
                )
                summary["pmc_hotspot_mut_frac"] = summary["lit_hotspot_mut_frac"]
                summary["gt_pmc_hotspot_mut_frac"] = summary["gt_lit_hotspot_mut_frac"]
        out["methods"][method.name] = {"summary": summary, "per_tree": per}
        print(f"  summary keys: {list(summary)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
