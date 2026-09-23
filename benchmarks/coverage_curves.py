#!/usr/bin/env python3
"""Coverage curves on held-out roots.

For each root, generate up to K_max trees per method and score coverage at
prefixes K = K_step … K_max. Writes CSV/JSON under the chosen output directory.

Example::

    python benchmarks/coverage_curves.py --data data/h3n2/test \
        --methods neutral_bd plm_prior artreeformer_adapted treesbm
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tree_state import TreeState
from benchmarks.heldout.build_examples import build_examples, list_groups, load_tree
from benchmarks.methods.bd_methods import NeutralBD
from benchmarks.methods.plm_prior import PLMPrior
from benchmarks.methods.topology_prior import TopologyPriorMethod
from benchmarks.methods.treesbm import TreeSBMMethod
from benchmarks.adapters.branch_length import BranchLengthAdapter
from benchmarks.adapters.sequence import evolve_pyvolve
from benchmarks.metrics import trees as T
from benchmarks.metrics import sequences as S
from benchmarks.run_table import load_external_pools, rebuild_target, ESM, _cuda


DEFAULT_E_LIST = [0, 1, 2, 3, 5, 8, 10]


def curve_fields(e_list: list[int]) -> list[str]:
    base = [
        "method", "N", "K", "n_roots", "n_trees_scored",
        "coverage", "eps_frac",
        "mean_min_edit", "mean_min_edit_frac",
        "site_recall", "mut_f1", "mut_recovery", "cons_retention",
        "aa_acc_given_hit", "best_of_k_identity",
        "clade_recall", "n_eligible_clades",
    ]
    for e in e_list:
        base.append(f"coverage_obs_e{e}")
        base.append(f"coverage_obs_unique_e{e}")
        base.append(f"frac_gen_e{e}")
    base.append("runtime_gen_sec")
    return base


# Backward-compatible name used by older tests / imports
CURVE_FIELDS = curve_fields(DEFAULT_E_LIST)


def _leaf_seqs(tree: TreeState) -> list[str]:
    return [tree.node_seqs[l] for l in T.leaf_labels(tree) if tree.node_seqs.get(l)]


def _safe_root_id(root_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", root_id)[:120] or "root"


def emerging_clade_fingerprints(
    gt_tree: TreeState,
    root_seq: str,
    min_clade_size: int = 2,
    min_shared_muts: int = 1,
) -> list[frozenset[tuple[int, str]]]:
    """
    Eligible GT clades → shared substitution fingerprints vs root.

    A rooted clade C (min_clade_size ≤ |C| < N) is eligible if the intersection
    of leaf mutations_vs_root has size ≥ min_shared_muts. Fingerprint = that
    intersection (set of (pos, aa)).
    """
    leaves = T.leaf_labels(gt_tree)
    n_leaves = len(leaves)
    if n_leaves < min_clade_size + 1:
        return []

    leaf_muts = {
        lab: S.mutations_vs_root(root_seq, gt_tree.node_seqs[lab])
        for lab in leaves
        if gt_tree.node_seqs.get(lab)
    }
    fps: list[frozenset[tuple[int, str]]] = []
    seen: set[frozenset[tuple[int, str]]] = set()
    for clade in T.clades(gt_tree, min_size=min_clade_size):
        if len(clade) >= n_leaves:
            continue
        mut_sets = [leaf_muts[lab] for lab in clade if lab in leaf_muts]
        if len(mut_sets) < min_clade_size:
            continue
        shared = set.intersection(*mut_sets) if mut_sets else set()
        if len(shared) < min_shared_muts:
            continue
        fp = frozenset(shared)
        if fp not in seen:
            seen.add(fp)
            fps.append(fp)
    return fps


def clade_recall_at_k(
    fingerprints: list[frozenset[tuple[int, str]]],
    gen_seqs: list[str],
    root_seq: str,
) -> tuple[float, int]:
    """Return (clade_recall, n_eligible). Gen leaf hits clade if F ⊆ muts(g)."""
    n = len(fingerprints)
    if n == 0:
        return float("nan"), 0
    if not gen_seqs:
        return 0.0, n
    gen_muts = [S.mutations_vs_root(root_seq, g) for g in gen_seqs]
    covered = 0
    for fp in fingerprints:
        if any(fp.issubset(gm) for gm in gen_muts):
            covered += 1
    return covered / n, n


def score_pool(
    gt_leaf_seqs: list[str],
    root_seq: str,
    gen_pool: list[str],
    eps_frac: float,
    max_gt: int,
    max_gen: int,
    seed: int,
    e_list: list[int] | None = None,
    gt_tree: TreeState | None = None,
    min_clade_size: int = 2,
    min_shared_muts: int = 1,
    fingerprints: list[frozenset[tuple[int, str]]] | None = None,
) -> dict:
    """Leaf-pool metrics for one root at a given K (prefix of generated trees)."""
    e_list = list(e_list if e_list is not None else DEFAULT_E_LIST)
    rng = random.Random(seed)
    gt_sub = (rng.sample(gt_leaf_seqs, max_gt)
              if len(gt_leaf_seqs) > max_gt else list(gt_leaf_seqs))
    gen_sub = (rng.sample(gen_pool, max_gen)
               if len(gen_pool) > max_gen else list(gen_pool))

    nan = float("nan")
    out = {
        "coverage": nan, "mean_min_edit": nan, "mean_min_edit_frac": nan,
        "site_recall": nan, "mut_f1": nan, "mut_recovery": nan,
        "cons_retention": nan, "aa_acc_given_hit": nan, "best_of_k_identity": nan,
        "clade_recall": nan, "n_eligible_clades": 0,
    }
    for e in e_list:
        out[f"coverage_obs_e{e}"] = nan
        out[f"coverage_obs_unique_e{e}"] = nan
        out[f"frac_gen_e{e}"] = nan

    if not gt_sub or not gen_sub:
        return out

    cov = S.coverage_at_k(gt_sub, gen_sub, eps_frac=eps_frac)
    edits = [S.min_hamming(g, gen_sub) for g in gt_sub]
    L = max(len(g) for g in gt_sub) or 1
    mean_edit = mean(edits)
    prf_site = S.mutation_pr_f1(gen_pool, gt_leaf_seqs, root_seq, level="site")
    prf_sub = S.mutation_pr_f1(gen_pool, gt_leaf_seqs, root_seq, level="substitution")

    mut_rec, cons_ret, aa_hits = [], [], []
    for g in gt_sub:
        best = max(gen_sub, key=lambda x: S.identity(g, x))
        pr = S.positional_recovery(root_seq, g, best)
        if pr["mut_total"] or pr["cons_total"]:
            if pr["mut_recovery"] == pr["mut_recovery"]:
                mut_rec.append(pr["mut_recovery"])
            if pr["cons_retention"] == pr["cons_retention"]:
                cons_ret.append(pr["cons_retention"])
            if pr["aa_acc_given_hit"] == pr["aa_acc_given_hit"]:
                aa_hits.append(pr["aa_acc_given_hit"])

    gt_unique = list(dict.fromkeys(gt_sub))
    for e in e_list:
        out[f"coverage_obs_e{e}"] = S.coverage_at_e(gt_sub, gen_sub, e=e)
        out[f"coverage_obs_unique_e{e}"] = S.coverage_at_e(gt_unique, gen_sub, e=e)
        out[f"frac_gen_e{e}"] = S.frac_gen_within_e(gt_sub, gen_sub, e=e)

    if fingerprints is None and gt_tree is not None:
        fingerprints = emerging_clade_fingerprints(
            gt_tree, root_seq,
            min_clade_size=min_clade_size,
            min_shared_muts=min_shared_muts,
        )
    if fingerprints is not None:
        cr, n_el = clade_recall_at_k(fingerprints, gen_sub, root_seq)
        out["clade_recall"] = cr
        out["n_eligible_clades"] = n_el

    out.update({
        "coverage": cov,
        "mean_min_edit": mean_edit,
        "mean_min_edit_frac": mean_edit / L,
        "site_recall": prf_site["recall"],
        "mut_f1": prf_sub["f1"],
        "mut_recovery": mean(mut_rec) if mut_rec else float("nan"),
        "cons_retention": mean(cons_ret) if cons_ret else float("nan"),
        "aa_acc_given_hit": mean(aa_hits) if aa_hits else float("nan"),
        "best_of_k_identity": mean(S.best_of_k_identity(g, gen_sub) for g in gt_sub),
    })
    return out


def build_baseline_methods(args, params, esm) -> list:
    methods = []
    want = set(args.methods)
    birth, death = params["birth"], params["death"]

    if "neutral_bd" in want:
        methods.append(NeutralBD(birth, death))

    if "plm_prior" in want:
        if esm is None:
            print("WARN: plm_prior requested but ESM unavailable (--no-esm); skipping")
        else:
            methods.append(PLMPrior(
                esm.lm_logits, birth, death, params.get("subst_scale", 1.0)))

    if "artreeformer_adapted" in want:
        pool_dir = ROOT / "benchmarks/external_pools/sampled"
        pool_by_N = load_external_pools(pool_dir, "artreeformer", args.N)
        if not pool_by_N:
            print("WARN: artreeformer_adapted requested but no pools at "
                  f"{pool_dir}/artreeformer_N*.nwk; skipping "
                  "(run scripts/slurm_artreeformer.sh first)")
        else:
            train_dir = ROOT / args.train_data
            if not list_groups(train_dir):
                print(f"WARN: no train trees in {train_dir} for BranchLengthAdapter; "
                      "skipping artreeformer_adapted")
            else:
                train_trees = [load_tree(train_dir, g) for g in list_groups(train_dir)]
                bl_adapter = BranchLengthAdapter().fit(train_trees)
                seq_fn = lambda topo, root_seq, seed: evolve_pyvolve(
                    topo, root_seq, model=args.seq_model, seed=seed)
                methods.append(TopologyPriorMethod(
                    "artreeformer_adapted", pool_by_N, bl_adapter, seq_fn))
                print(f"artreeformer pools: Ns={sorted(pool_by_N)} "
                      f"sizes={[len(pool_by_N[n]) for n in sorted(pool_by_N)]}")

    if "treesbm" in want:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        if not ckpt.exists():
            print(f"WARN: treesbm requested but checkpoint missing: {ckpt}; skipping")
        else:
            r0_live = None
            if getattr(args, "r0_backend", None):
                from src.r0_backends import build_r0_backend, normalize_backend_name
                r0_name = normalize_backend_name(args.r0_backend)
                print(f"treesbm R0 backend override: {r0_name}")
                r0_live = build_r0_backend(r0_name, model_id=args.r0_model)
            methods.append(TreeSBMMethod(
                str(ckpt),
                n_steps=args.n_steps,
                max_seq_len=args.max_seq_len,
                r0_backend=r0_live,
                fitness_beta=getattr(args, "fitness_beta", None),
                ablate_bridge=bool(getattr(args, "ablate_bridge", False)),
            ))
            print(f"treesbm checkpoint: {ckpt} (ESM cache on via generate_k)")

    return methods


def k_grid(k_max: int, k_step: int) -> list[int]:
    ks = list(range(k_step, k_max + 1, k_step))
    if not ks or ks[-1] != k_max:
        ks.append(k_max)
    return ks


def _cache_root_dir(cache_dir: Path, method_name: str, N: int, root_id: str) -> Path:
    return cache_dir / method_name / f"N{N}" / _safe_root_id(root_id)


def save_trees_cache(cache_dir: Path, method_name: str, N: int, root_id: str,
                     trees: list[TreeState], meta: dict) -> None:
    d = _cache_root_dir(cache_dir, method_name, N, root_id)
    d.mkdir(parents=True, exist_ok=True)
    for i, tree in enumerate(trees):
        (d / f"tree_{i:04d}.json").write_text(json.dumps(tree.to_dict()))
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def load_trees_cache(cache_dir: Path, method_name: str, N: int, root_id: str,
                     k_max: int) -> list[TreeState]:
    d = _cache_root_dir(cache_dir, method_name, N, root_id)
    if not d.is_dir():
        return []
    trees = []
    for i in range(k_max):
        p = d / f"tree_{i:04d}.json"
        if not p.exists():
            break
        trees.append(TreeState.from_dict(json.loads(p.read_text())))
    return trees


def generate_or_load_trees(method, ex, args, N: int, cache_dir: Path | None,
                           rescore_from: Path | None) -> tuple[list[TreeState], float]:
    """Return (trees, gen_seconds). gen_seconds=0 when loading from cache."""
    root_seq, H = ex["root_seq"], ex["H"]
    root_id = ex["root_id"]

    if rescore_from is not None:
        trees = load_trees_cache(rescore_from, method.name, N, root_id, args.K_max)
        if not trees:
            print(f"  [{method.name}] no cache for root={root_id} under {rescore_from}")
        return trees, 0.0

    trees = []
    t0 = time.time()
    for k in range(args.K_max):
        try:
            g = method.generate(
                root_seq, N, H,
                seed=args.seed * 100_000 + hash(ex["root_id"]) % 9973 + k)
            trees.append(g.tree)
        except Exception as e:
            print(f"  [{method.name}] gen fail root={root_id} k={k}: {e}")
    gen_secs = time.time() - t0

    if cache_dir is not None and trees:
        save_trees_cache(
            cache_dir, method.name, N, root_id, trees,
            meta={
                "root_id": root_id,
                "N": N,
                "H": H,
                "K": len(trees),
                "method": method.name,
                "seed": args.seed,
            },
        )
    return trees, gen_secs


def run_for_N(methods, examples, args, ks: list[int], writer, N: int,
              out_fh=None, e_list: list[int] | None = None) -> None:
    """Generate (or load) K_max once per (method, root); score all K prefixes."""
    e_list = list(e_list if e_list is not None else args.e_list)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir and not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    rescore_from = Path(args.rescore_from) if args.rescore_from else None
    if rescore_from and not rescore_from.is_absolute():
        rescore_from = ROOT / rescore_from

    for method in methods:
        per_k: dict[int, list[dict]] = {k: [] for k in ks}
        gen_secs = 0.0
        n_ok = 0
        for ex in examples:
            target = rebuild_target(ex)
            gt_leaves = _leaf_seqs(target)
            root_seq = ex["root_seq"]
            if len(gt_leaves) != N:
                print(f"  skip {ex['root_id']}: gt leaves={len(gt_leaves)} != N={N}")
                continue

            trees, secs = generate_or_load_trees(
                method, ex, args, N, cache_dir, rescore_from)
            gen_secs += secs
            if not trees:
                continue
            n_ok += 1

            fps = emerging_clade_fingerprints(
                target, root_seq,
                min_clade_size=args.min_clade_size,
                min_shared_muts=args.min_shared_muts,
            )

            pool: list[str] = []
            next_i = 0
            for K in ks:
                while next_i < min(K, len(trees)):
                    pool.extend(_leaf_seqs(trees[next_i]))
                    next_i += 1
                metrics = score_pool(
                    gt_leaves, root_seq, pool,
                    eps_frac=args.eps_frac,
                    max_gt=args.max_gt_leaves,
                    max_gen=args.max_gen_pool,
                    seed=args.seed + K,
                    e_list=e_list,
                    gt_tree=target,
                    min_clade_size=args.min_clade_size,
                    min_shared_muts=args.min_shared_muts,
                    fingerprints=fps,
                )
                per_k[K].append(metrics)

            last = per_k[ks[-1]][-1]
            cov_e2 = last.get("coverage_obs_e2", float("nan"))
            print(f"  [{method.name}] root={ex['root_id']} "
                  f"trees={len(trees)} cov@ε={last['coverage']:.3f} "
                  f"cov@e2={cov_e2 if cov_e2 == cov_e2 else float('nan'):.3f} "
                  f"edit={last['mean_min_edit']:.1f} "
                  f"clade_r={last['clade_recall'] if last['clade_recall'] == last['clade_recall'] else float('nan'):.3f} "
                  f"site_r={last['site_recall']:.3f}")

        for K in ks:
            rows = per_k[K]
            if not rows:
                continue

            def _m(key):
                vals = [r[key] for r in rows if r.get(key) == r.get(key)]
                return mean(vals) if vals else float("nan")

            row = {
                "method": method.name,
                "N": N,
                "K": K,
                "n_roots": n_ok,
                "n_trees_scored": K,
                "coverage": _m("coverage"),
                "eps_frac": args.eps_frac,
                "mean_min_edit": _m("mean_min_edit"),
                "mean_min_edit_frac": _m("mean_min_edit_frac"),
                "site_recall": _m("site_recall"),
                "mut_f1": _m("mut_f1"),
                "mut_recovery": _m("mut_recovery"),
                "cons_retention": _m("cons_retention"),
                "aa_acc_given_hit": _m("aa_acc_given_hit"),
                "best_of_k_identity": _m("best_of_k_identity"),
                "clade_recall": _m("clade_recall"),
                "n_eligible_clades": _m("n_eligible_clades"),
                "runtime_gen_sec": gen_secs,
            }
            for e in e_list:
                row[f"coverage_obs_e{e}"] = _m(f"coverage_obs_e{e}")
                row[f"coverage_obs_unique_e{e}"] = _m(f"coverage_obs_unique_e{e}")
                row[f"frac_gen_e{e}"] = _m(f"frac_gen_e{e}")
            writer.writerow(row)
        if out_fh is not None:
            out_fh.flush()
            print(f"  flushed {method.name} rows (n_roots={n_ok})")


def _parse_e_list(values: list[str] | None) -> list[int]:
    if not values:
        return list(DEFAULT_E_LIST)
    out: list[int] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
    # unique, sorted
    return sorted(set(out))


class _RescoreStub:
    """Minimal method stand-in when --rescore-from is set (no generation)."""

    def __init__(self, name: str):
        self.name = name

    def generate(self, *args, **kwargs):
        raise RuntimeError("rescore stub cannot generate")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-data", default="data/h3n2/test")
    ap.add_argument("--train-data", default="data/h3n2/train",
                    help="Train trees for BranchLengthAdapter (ARTreeFormer only)")
    ap.add_argument("--params", default="benchmarks/results/params.json")
    ap.add_argument("--N", type=int, nargs="+", default=[16])
    ap.add_argument("--K-max", type=int, default=100)
    ap.add_argument("--K-step", type=int, default=10)
    ap.add_argument("--max-roots", type=int, default=5,
                    help="Held-out roots to average (locked: 5)")
    ap.add_argument("--methods", nargs="+",
                    default=["neutral_bd", "plm_prior", "artreeformer_adapted", "treesbm"],
                    choices=["neutral_bd", "plm_prior", "artreeformer_adapted", "treesbm"])
    ap.add_argument("--eps-frac", type=float, default=0.02,
                    help="Legacy Coverage@ε fractional Hamming (default: 0.02)")
    ap.add_argument("--e-list", nargs="+", default=None,
                    help="Absolute Hamming radii for coverage_obs_e* / frac_gen_e* "
                         "(default: 0 1 2 3 5 8 10). Accepts comma-separated tokens.")
    ap.add_argument("--min-clade-size", type=int, default=2,
                    help="Min GT clade size for clade_recall eligibility")
    ap.add_argument("--min-shared-muts", type=int, default=1,
                    help="Min shared substitutions vs root for a clade fingerprint")
    ap.add_argument("--cache-dir", default=None,
                    help="Save generated TreeState JSON trees for later rescoring")
    ap.add_argument("--rescore-from", default=None,
                    help="Load trees from a previous --cache-dir; skip generation")
    ap.add_argument("--checkpoint", default="checkpoints/h3n2_v3_lit_hotspot/best.pt",
                    help="TreeSBM checkpoint (required when treesbm in --methods)")
    ap.add_argument("--n-steps", type=int, default=50,
                    help="TreeSBM sampler steps")
    ap.add_argument("--max-seq-len", type=int, default=566,
                    help="TreeSBM / H3N2 max sequence length")
    ap.add_argument("--seq-model", default="JTT",
                    help="Shared sequence adapter for artreeformer_adapted")
    ap.add_argument("--max-gt-leaves", type=int, default=60)
    ap.add_argument("--max-gen-pool", type=int, default=800)
    ap.add_argument("--no-esm", action="store_true")
    ap.add_argument(
        "--r0-backend", default=None,
        help="Swap TreeSBM frozen R0 backend (esm2 / esm2_650m / esmc / jtt / wag / lg).",
    )
    ap.add_argument("--r0-model", default=None)
    ap.add_argument("--fitness-beta", type=float, default=None)
    ap.add_argument("--ablate-bridge", action="store_true",
                    help="Force log R_θ = log R0 (pure reference process, no learned correction).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/results/coverage_curves_h3n2.csv")
    args = ap.parse_args()
    args.e_list = _parse_e_list(args.e_list)

    params_path = ROOT / args.params
    if not params_path.exists():
        raise SystemExit(
            f"missing {params_path}; fit with: "
            f"python benchmarks/fit_params.py --train-data {args.train_data} "
            f"--out {args.params}")
    params = json.loads(params_path.read_text())

    test_dir = ROOT / args.test_data
    if not list_groups(test_dir):
        raise SystemExit(
            f"no processed groups (group_*_rooted.nwk + *_anc_aa.fasta) in {test_dir}. "
            "Typically under data/h3n2/test after the H3N2 pipeline; "
            "local checkout currently has FASTA-only shards.")

    if args.rescore_from:
        methods = [_RescoreStub(m) for m in args.methods]
        print(f"rescore-from={args.rescore_from} (no generation)")
    else:
        esm = None
        if not args.no_esm and "plm_prior" in args.methods:
            # Must match pathogen length (COVID Spike=1280); default 566 truncates
            # and causes "index … out of bounds … size 566" on L>566 roots.
            esm = ESM("cuda" if _cuda() else "cpu", max_len=args.max_seq_len)
        methods = build_baseline_methods(args, params, esm)
        if not methods:
            raise SystemExit("no methods available to run")

    ks = k_grid(args.K_max, args.K_step)
    fields = curve_fields(args.e_list)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"methods={[m.name for m in methods]} Ks={ks} e_list={args.e_list} "
          f"max_roots={args.max_roots}")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for N in args.N:
            # Prefer one root per group so max_roots spans trees (default
            # max_roots_per_tree=5 collapses to the first group only).
            raw = build_examples(
                test_dir, N, seed=args.seed, max_roots_per_tree=1)
            examples = []
            skipped_identical = 0
            for ex in raw:
                leaves = list(ex["target_seqs"].values())
                if not leaves:
                    continue
                mean_edit = mean(S.hamming(ex["root_seq"], s) for s in leaves)
                # Skip zero-diversity COVID-like subtrees (coverage trivially 1.0).
                # Also skip extreme length-mismatch outliers (mean_edit >> L*0.05).
                L = max(len(ex["root_seq"]), 1)
                if mean_edit <= 0:
                    skipped_identical += 1
                    continue
                if mean_edit > 0.05 * L:
                    skipped_identical += 1
                    continue
                examples.append(ex)
                if len(examples) >= args.max_roots:
                    break
            if len(examples) < args.max_roots:
                # Fall back to unfiltered one-per-group if diversity filter too strict
                print(f"WARN: only {len(examples)} diverse roots; "
                      f"falling back to first {args.max_roots} one-per-group "
                      f"(skipped_identical_or_outlier={skipped_identical})")
                examples = raw[: args.max_roots]
            print(f"\n=== N={N}: {len(examples)} held-out roots "
                  f"(groups={[e['group'] for e in examples]}) ===")
            for ex in examples:
                leaves = list(ex["target_seqs"].values())
                me = mean(S.hamming(ex["root_seq"], s) for s in leaves)
                print(f"  group={ex['group']:03d} root={ex['root_id']} "
                      f"H={ex['H']:.4g} mean_edit={me:.2f} uniq={len(set(leaves))}")
            if not examples:
                print(f"  no examples for N={N}; skip")
                continue
            run_for_N(methods, examples, args, ks, w, N=N, out_fh=f,
                      e_list=args.e_list)
            f.flush()

    print(f"\nwrote {out}")
    print("Plot with: Rscript benchmarks/plot_coverage_curves.R", out)


if __name__ == "__main__":
    main()
