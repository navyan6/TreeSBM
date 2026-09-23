"""TreeSBM baseline wrapper.

Maps the benchmark ``(root_seq, N, H)`` interface onto TreeSBM's native
root-conditioned sampler (leaf cap / rate scaling).
"""

from __future__ import annotations

import random
import time

from src.tree_state import TreeState
from benchmarks.methods.base import Method, GeneratedTree
from benchmarks.generation import TreeSBMGenerator
from benchmarks.heldout.build_examples import induced_subtree
from benchmarks.metrics import trees as T


class TreeSBMMethod(Method):
    name = "treesbm"

    def __init__(self, checkpoint: str, n_steps: int = 50, branch_rate_scale: float = 6.0,
                 rate_per_H: float = 1.2, max_seq_len: int = 566, cushion: float = 1.6,
                 max_retries: int = 4, r0_backend=None, fitness_beta: float | None = None,
                 ablate_bridge: bool = False, gene_id: str | None = None):
        self.gen = TreeSBMGenerator(
            checkpoint, max_seq_len=max_seq_len,
            r0_backend=r0_backend, fitness_beta=fitness_beta,
            ablate_bridge=ablate_bridge, gene_id=gene_id,
        )
        self.n_steps = n_steps
        self.branch_rate_scale = branch_rate_scale
        self.rate_per_H = rate_per_H
        self.cushion = cushion
        self.max_retries = max_retries

    def generate(self, root_seq: str, N: int, H: float, seed: int) -> GeneratedTree:
        t0 = time.time()
        mut_scale = max(1e-4, H * self.rate_per_H)      # horizon -> divergence
        cap = int(N * self.cushion) + 2
        tree = None
        for r in range(self.max_retries):
            g = self.gen.generate_k(
                root_seq, K=1, n_steps=self.n_steps + 10 * r, max_leaves=cap,
                branch_rate_scale=self.branch_rate_scale * (1 + 0.3 * r),
                mutation_rate_scale=mut_scale, base_seed=seed + 100 * r,
            )[0]
            if len(T.leaf_labels(g)) >= N:
                tree = g
                break
        if tree is None:
            tree = g  # accept the last attempt; validity will flag if < N

        leaves = T.leaf_labels(tree)
        if len(leaves) > N:
            sel = random.Random(seed).sample(leaves, N)
            tree = induced_subtree(tree, tree.root_id, sel)   # exactly N, collapse+sum bls
            leaves = T.leaf_labels(tree)

        # (N,H) conditioning: rescale branch lengths so mean root-to-tip == H, exactly
        # as the BD baselines do. TreeSBM supplies topology + sequences; H sets the
        # timescale. Without this, TreeSBM's (miscalibrated) branch lengths fail the
        # horizon check and the row is dropped.
        times = T.node_times(tree)
        cur = sum(times[l] for l in leaves) / len(leaves) if leaves else 0.0
        if cur > 0 and H > 0:
            s = H / cur
            tree = TreeState(
                node_ids=tree.node_ids, root_id=tree.root_id, edges=tree.edges,
                branch_lengths={e: v * s for e, v in tree.branch_lengths.items()},
                node_seqs=tree.node_seqs, active_leaves=list(tree.active_leaves))

        return GeneratedTree(tree, {"runtime": time.time() - t0,
                                    "target_N": N, "target_H": H, "mut_scale": mut_scale})
