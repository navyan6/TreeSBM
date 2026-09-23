#!/usr/bin/env python3
"""
Orchestrator: run every method on the held-out-root generation task, validate
every sample, score both tracks, and write the long-format results.csv that
make_table.py aggregates.

Empirical track  -> per generated sample: RF / quartet / branch-W / terminal-edit
                    vs the single real target subtree (averaged over K, over roots).
Simulated track  -> Tree-JS / Split-JS between the K generated trees and the M
                    reference trees (per regime), plus mean gen-vs-ref distances.

Needs dendropy + pyvolve (baselines), torch/ESM (pLM-prior, TreeSBM); runs on the
cluster. Native methods only unless --checkpoint / ESM available.

    python benchmarks/run_table.py --test-data data/h3n2/test \
        --params benchmarks/results/params.json --checkpoint checkpoints/h3n2_temporal/best.pt \
        --N 16 --K 100 --M 50 --max-roots 100 --out benchmarks/results/results.csv
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.tree_state import TreeState
from benchmarks.heldout.build_examples import build_examples, list_groups, load_tree
from benchmarks.methods.bd_methods import NeutralBD, EmpiricalBD
from benchmarks.methods.plm_prior import PLMPrior
from benchmarks.methods.topology_prior import TopologyPriorMethod
from benchmarks.methods.phylaflow_native import NativePhylaFlowMethod
from benchmarks.adapters.branch_length import BranchLengthAdapter
from benchmarks.adapters.sequence import evolve_pyvolve
from benchmarks import validity as V
from benchmarks.metrics.matched import sequence_matched_rf, quartet_distance, terminal_edit_distance
from benchmarks.metrics.branch_lengths import branch_length_wasserstein
from benchmarks.metrics.distributions import tree_js, split_js
from benchmarks.sim.reference import simulate_reference, REGIMES

FIELDS = ["method", "track", "sim_regime", "root_id", "N", "H", "sample_seed",
          "valid", "tree_kl", "split_kl", "rf", "quartet", "branch_w_all",
          "terminal_edit", "runtime"]


def rebuild_target(ex: dict) -> TreeState:
    edges, bls = [], {}
    for k, v in ex["target_branch_lengths"].items():
        p, c = k.split("|"); edges.append((p, c)); bls[(p, c)] = v
    node_seqs = ex["target_node_seqs"]
    node_ids = list(dict.fromkeys([ex["root_id"]] + [n for e in edges for n in e] + list(node_seqs)))
    leaves = [n for n in node_ids if n not in {p for p, _ in edges}]
    return TreeState(node_ids=node_ids, root_id=ex["root_id"], edges=edges,
                     branch_lengths=bls, node_seqs=node_seqs, active_leaves=leaves)


def _qd(a, b):
    try:
        return quartet_distance(a, b)
    except ImportError:
        return float("nan")


def score_empirical(gen: TreeState, target: TreeState) -> dict:
    return {"tree_kl": float("nan"), "split_kl": float("nan"),
            "rf": sequence_matched_rf(gen, target), "quartet": _qd(gen, target),
            "branch_w_all": branch_length_wasserstein(gen, target)["all"],
            "terminal_edit": terminal_edit_distance(gen, target)["mean"]}


def score_simulated(gens: list[TreeState], refs: list[TreeState], seed: int) -> dict:
    rng = random.Random(seed)
    rf, q, bw, te = [], [], [], []
    for g in gens:
        r = rng.choice(refs)
        rf.append(sequence_matched_rf(g, r)); bw.append(branch_length_wasserstein(g, r)["all"])
        te.append(terminal_edit_distance(g, r)["mean"])
        qv = _qd(g, r)
        if qv == qv:
            q.append(qv)
    return {"tree_kl": tree_js(gens, refs), "split_kl": split_js(gens, refs),
            "rf": mean(rf) if rf else float("nan"),
            "quartet": mean(q) if q else float("nan"),
            "branch_w_all": mean(bw) if bw else float("nan"),
            "terminal_edit": mean(te) if te else float("nan")}


def load_external_pools(pool_dir: Path, prefix: str, Ns: list[int]) -> dict[int, list[str]]:
    """{N: [newick, ...]} from benchmarks/external_pools/sampled/{prefix}_N{N}.nwk,
    produced by scripts/slurm_artreeformer.sh / slurm_phylovae.sh /
    slurm_phylaflow.sh. Skips N's whose pool file doesn't exist yet -- that N
    is just absent from the row."""
    pool_by_N = {}
    for N in Ns:
        p = pool_dir / f"{prefix}_N{N}.nwk"
        if p.exists():
            lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
            if lines:
                pool_by_N[N] = lines
    return pool_by_N


def build_methods(args, params, esm, train_trees: list[TreeState] | None = None):
    methods = [NeutralBD(params["birth"], params["death"]),
               EmpiricalBD(params["birth"], params["death"], model=args.empirical_model)]
    if esm is not None:
        methods.append(PLMPrior(esm.lm_logits, params["birth"], params["death"],
                                params.get("subst_scale", 1.0)))
    if args.checkpoint:
        from benchmarks.methods.treesbm import TreeSBMMethod
        r0_live = None
        if getattr(args, "r0_backend", None):
            from src.r0_backends import build_r0_backend, normalize_backend_name
            r0_live = build_r0_backend(
                normalize_backend_name(args.r0_backend),
                model_id=getattr(args, "r0_model", None),
            )
        methods.append(TreeSBMMethod(
            args.checkpoint, max_seq_len=getattr(args, "max_seq_len", 566),
            r0_backend=r0_live,
            fitness_beta=getattr(args, "fitness_beta", None),
            ablate_bridge=bool(getattr(args, "ablate_bridge", False)),
        ))

    # External topology models (ARTreeFormer / PhyloVAE / PhylaFlow). Pools under
    # benchmarks/external_pools/sampled/ — absent pools skip the row (not an error).
    # PhylaFlow native (`phylaflow`): official sampler topo+BL, shared JTT seqs.
    # PhylaFlow adapted (`phylaflow_adapted`): same pool topo + shared BL adapter.
    # Neither is root-conditioned forward gen (see BLOCKERS.md / EXTERNAL.md).
    pool_dir = ROOT / "benchmarks/external_pools/sampled"
    seq_fn = lambda topo, root_seq, seed: evolve_pyvolve(
        topo, root_seq, model=args.empirical_model, seed=seed)

    phyla_pool = load_external_pools(pool_dir, "phylaflow", args.N)
    if phyla_pool:
        methods.append(NativePhylaFlowMethod(phyla_pool, seq_fn))

    if train_trees:
        bl_adapter = BranchLengthAdapter().fit(train_trees)
        for tag, prefix in [("artreeformer_adapted", "artreeformer"),
                            ("phylovae_adapted", "phylovae"),
                            ("phylaflow_adapted", "phylaflow")]:
            pool_by_N = load_external_pools(pool_dir, prefix, args.N)
            if pool_by_N:
                methods.append(TopologyPriorMethod(tag, pool_by_N, bl_adapter, seq_fn))
    elif phyla_pool:
        # Native PhylaFlow does not need train_trees (no shared BL fit); adapted
        # rows still skipped until train_trees are loaded.
        pass

    want = getattr(args, "methods", None)
    if want:
        allow = set(want)
        methods = [m for m in methods if m.name in allow]
        missing = allow - {m.name for m in methods}
        if missing:
            print(f"WARNING: requested methods not built: {sorted(missing)}")
    return methods


class ESM:
    def __init__(self, device, max_len=566):
        import torch
        from transformers import AutoTokenizer, EsmForMaskedLM
        from scripts.eval_single_tree import get_lm_logits, AA_VOCAB
        self._gl, self._t = get_lm_logits, torch
        mid = "facebook/esm2_t6_8M_UR50D"
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = EsmForMaskedLM.from_pretrained(mid).to(device).eval()
        self.aa = torch.tensor([self.tok.convert_tokens_to_ids(a) for a in AA_VOCAB])
        self.device, self.max_len = device, max_len

    def lm_logits(self, seq):
        return self._gl(self.tok, self.model, self.aa, [seq], self.max_len, self.device)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", default="data/h3n2/test")
    ap.add_argument("--train-data", default="data/h3n2/train",
                    help="Only used to fit the shared BranchLengthAdapter for the "
                         "ARTreeFormer/PhyloVAE/PhylaFlow adapted rows, if pools exist.")
    ap.add_argument("--params", default="benchmarks/results/params.json")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument(
        "--methods", nargs="+", default=None,
        help="Optional method-name filter (e.g. treesbm). Default: all built methods.",
    )
    ap.add_argument("--empirical-model", default="JTT")
    ap.add_argument("--N", type=int, nargs="+", default=[16])
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--max-roots", type=int, default=100)
    ap.add_argument(
        "--regimes", nargs="*", default=None,
        help="Sim-track regimes (default: all REGIMES). Pass empty (--regimes) "
             "to skip simulated track entirely (needed when roots contain X).",
    )
    ap.add_argument(
        "--no-sim", action="store_true",
        help="Skip simulated reference track (empirical RF/quartet/W1/edit only).",
    )
    ap.add_argument("--no-esm", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=566,
                    help="ESM logits + TreeSBM RateHeads length "
                         "(H3N2/H1=566, COVID Spike=1280)")
    ap.add_argument("--r0-backend", default=None,
                    help="Swap TreeSBM frozen R0 prior backend.")
    ap.add_argument("--r0-model", default=None)
    ap.add_argument("--fitness-beta", type=float, default=None)
    ap.add_argument("--ablate-bridge", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/results/results.csv")
    args = ap.parse_args()

    params = json.loads((ROOT / args.params).read_text())
    if args.no_sim:
        regimes = []
    elif args.regimes is None:
        regimes = list(REGIMES)
    else:
        regimes = list(args.regimes)
    print(f"regimes={regimes or '(none — empirical only)'}")
    esm = None if args.no_esm else ESM(
        "cuda" if _cuda() else "cpu", max_len=args.max_seq_len)
    pool_dir = ROOT / "benchmarks/external_pools/sampled"
    have_pools = pool_dir.exists() and any(pool_dir.glob("*.nwk"))
    train_trees = None
    if have_pools:
        train_dir = ROOT / args.train_data
        train_trees = [load_tree(train_dir, g) for g in list_groups(train_dir)]
    methods = build_methods(args, params, esm, train_trees)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        f.flush()
        for N in args.N:
            examples = build_examples(ROOT / args.test_data, N, seed=args.seed)[: args.max_roots]
            print(f"N={N}: {len(examples)} roots x {len(methods)} methods", flush=True)
            for ex in examples:
                target = rebuild_target(ex)
                root_seq, H = ex["root_seq"], ex["H"]
                # refs don't depend on method -- compute once per (root, regime),
                # not once per (root, method, regime) (was a 3x redundant cost).
                regime_refs = {}
                for regime in regimes:
                    try:
                        regime_refs[regime] = simulate_reference(
                            root_seq, N, H, regime, args.M,
                            params["birth"], params["death"],
                            seed=args.seed + hash(ex["root_id"]) % 9973,
                        )
                    except Exception as e:
                        print(
                            f"  SKIP sim regime={regime} root={ex['root_id']}: {e}",
                            flush=True,
                        )
                for method in methods:
                    gens, valids = [], []
                    for k in range(args.K):
                        try:
                            g = method.generate(root_seq, N, H, seed=args.seed * 10000 + k)
                            vr = V.validate(g.tree, root_seq, N, H)
                        except Exception as e:
                            print(
                                f"  GEN/VALID fail method={method.name} "
                                f"root={ex['root_id']} k={k}: {e}",
                                flush=True,
                            )
                            continue
                        gens.append(g); valids.append(vr)
                    valid_trees = [g.tree for g, vr in zip(gens, valids) if vr["valid"]]
                    if not valid_trees and valids:
                        reasons = {}
                        for vr in valids:
                            for r in vr.get("reasons", []) or [str(vr)]:
                                reasons[r] = reasons.get(r, 0) + 1
                        print(
                            f"  0/{len(valids)} valid method={method.name} "
                            f"root={ex['root_id']} reasons={reasons}",
                            flush=True,
                        )
                    base = dict(method=method.name, root_id=ex["root_id"], N=N, H=H,
                                valid=len(valid_trees), sample_seed=args.seed,
                                runtime=sum(g.meta.get("runtime", 0.0) for g in gens))
                    # empirical track (mean over valid samples vs the one true subtree)
                    if valid_trees:
                        es = [score_empirical(t, target) for t in valid_trees]
                        row = {**base, "track": "empirical", "sim_regime": ""}
                        for m in ("rf", "quartet", "branch_w_all", "terminal_edit",
                                  "tree_kl", "split_kl"):
                            vals = [e[m] for e in es if e[m] == e[m]]
                            row[m] = mean(vals) if vals else float("nan")
                        w.writerow(row)
                    # simulated track (per regime)
                    for regime, refs in regime_refs.items():
                        if valid_trees:
                            ss = score_simulated(valid_trees, refs, seed=args.seed)
                            w.writerow({**base, "track": f"sim_{regime}",
                                        "sim_regime": regime, **ss})
                    f.flush()
    print(f"wrote {out}")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
