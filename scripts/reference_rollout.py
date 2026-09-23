#!/usr/bin/env python3
"""
Algorithm 3: reference-process rollout (frozen R0, no learned correction).

Generates trees under P^0 without TreeSBM bridge control:
  Q^0 from --r0-backend (± --fitness-beta tilt) + Poisson λ branching.

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.r0_backends import STUB_BACKENDS, build_r0_backend, list_backends, normalize_backend_name
from src.reference_process import (
    ConstantBranchingIntensity,
    ReferenceProcess,
)


def tree_to_newick(tree) -> str:
    children_map: dict[str, list[str]] = {}
    for p, c in tree.edges:
        children_map.setdefault(p, []).append(c)

    def _fmt(nid: str) -> str:
        bl = tree.branch_lengths.get(
            next(((p, nid) for p, c in tree.edges if c == nid), (None, None)),
            0.0,
        )
        children = children_map.get(nid, [])
        if not children:
            return f"{nid}:{bl:.6f}"
        inner = ",".join(_fmt(c) for c in children)
        return f"({inner}){nid}:{bl:.6f}"

    return _fmt(tree.root_id) + ";"


def main():
    p = argparse.ArgumentParser(description="Alg. 3 ReferenceRollout (P^0 only).")
    p.add_argument("--root-seq", required=True, help="Ancestral AA sequence.")
    p.add_argument("--r0-backend", default="esm2", help=f"One of {list_backends()}.")
    p.add_argument("--r0-model", default=None, help="Optional model id override.")
    p.add_argument(
        "--fitness-beta", "--ref-tilt-beta",
        type=float, default=0.0, dest="fitness_beta",
        help="§4.2 tilt β (0=off).",
    )
    p.add_argument(
        "--fitness-score",
        choices=["log_R0", "log_softmax"],
        default="log_R0",
    )
    p.add_argument(
        "--fitness-tilt-mode",
        choices=["site_local", "full_esm"],
        default="site_local",
        help="site_local (Option A) or full_esm (Option B mutant PLL).",
    )
    p.add_argument(
        "--fitness-esm-top-k",
        type=int,
        default=None,
        help="full_esm: only score top-k untilted AAs per site.",
    )
    p.add_argument(
        "--fitness-esm-batch-size",
        type=int,
        default=8,
    )
    p.add_argument(
        "--ref-lambda",
        type=float,
        default=1.0,
        help="Constant Poisson branching intensity λ (sequence-independent).",
    )
    p.add_argument("--horizon", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--max-nodes", type=int, default=256)
    p.add_argument("--p-stop", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reference_rollout.nwk")
    p.add_argument("--meta-out", default=None, help="Optional JSON sidecar path.")
    args = p.parse_args()

    backend_name = normalize_backend_name(args.r0_backend)
    if backend_name in STUB_BACKENDS:
        raise SystemExit(
            f"Backend {backend_name!r} is stubbed. "
            "Implement/install it or pick esm2 / esmc / jtt / wag / lg."
        )

    rng = np.random.default_rng(args.seed)
    print(f"Building R0 backend={backend_name} …")
    r0 = build_r0_backend(backend_name, model_id=args.r0_model)
    branching = ConstantBranchingIntensity(args.ref_lambda)
    ref = ReferenceProcess(
        r0_backend=r0,
        branching_intensity=branching,
        beta=args.fitness_beta,
        fitness_score=args.fitness_score,
        fitness_tilt_mode=args.fitness_tilt_mode,
        fitness_esm_batch_size=args.fitness_esm_batch_size,
        fitness_esm_top_k=args.fitness_esm_top_k,
        p_stop=args.p_stop,
        rng=rng,
    )

    print(
        f"Rollout horizon={args.horizon} dt={args.dt} λ={args.ref_lambda} "
        f"β={args.fitness_beta} mode={args.fitness_tilt_mode}"
    )
    tree = ref.rollout(
        args.root_seq.upper(),
        horizon=args.horizon,
        dt=args.dt,
        max_nodes=args.max_nodes,
    )
    nwk = tree_to_newick(tree)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(nwk + "\n")
    print(f"Wrote {out_path}  nodes={tree.n_nodes()} leaves={tree.n_leaves()}")

    meta = {
        "r0_backend": backend_name,
        "r0_model": args.r0_model,
        "fitness_beta": args.fitness_beta,
        "fitness_score": args.fitness_score,
        "fitness_tilt_mode": args.fitness_tilt_mode,
        "fitness_esm_top_k": args.fitness_esm_top_k,
        "ref_lambda": args.ref_lambda,
        "horizon": args.horizon,
        "dt": args.dt,
        "max_nodes": args.max_nodes,
        "n_nodes": tree.n_nodes(),
        "n_leaves": tree.n_leaves(),
        "seed": args.seed,
    }
    meta_path = Path(args.meta_out) if args.meta_out else out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {meta_path}")
    r0.close()


if __name__ == "__main__":
    main()
