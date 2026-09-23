"""PhylaFlow baseline using native sampled topologies.

Loads PhylaFlow trees and applies the shared branch-length adapter when needed.
"""

from __future__ import annotations

import io
import random
import time
from collections import deque

from Bio import Phylo

from src.tree_state import TreeState
from benchmarks.methods.base import Method, GeneratedTree, attach_sequences
from benchmarks.metrics import trees as T


def parse_phylaflow_tree(newick: str) -> TreeState:
    """Newick (optionally with BLs) → TreeState; preserves branch lengths when present."""
    tree = Phylo.read(io.StringIO(newick), "newick")
    edges, bls, counter = [], {}, [0]

    def name(clade):
        if clade.name:
            return str(clade.name)
        clade.name = f"NODE_{counter[0]:06d}"
        counter[0] += 1
        return clade.name

    def walk(parent):
        pn = name(parent)
        for ch in parent.clades:
            cn = name(ch)
            edges.append((pn, cn))
            bls[(pn, cn)] = float(ch.branch_length or 0.0)
            walk(ch)

    walk(tree.root)
    root = name(tree.root)
    cm: dict[str, list[str]] = {}
    for p, c in edges:
        cm.setdefault(p, []).append(c)
    node_ids, seen, q = [root], {root}, deque([root])
    while q:
        for c in cm.get(q.popleft(), []):
            if c not in seen:
                seen.add(c)
                node_ids.append(c)
                q.append(c)
    leaves = [n for n in node_ids if n not in cm]
    return TreeState(
        node_ids=node_ids, root_id=root, edges=edges, branch_lengths=bls,
        node_seqs={}, active_leaves=leaves,
    )


def rescale_to_horizon(tree: TreeState, H: float) -> TreeState:
    """Rescale branch lengths so mean root-to-tip == H (same H adapter as TreeSBM)."""
    leaves = T.leaf_labels(tree)
    times = T.node_times(tree)
    cur = sum(times[l] for l in leaves) / len(leaves) if leaves else 0.0
    if cur <= 0 or H <= 0:
        return tree
    s = H / cur
    return TreeState(
        node_ids=tree.node_ids, root_id=tree.root_id, edges=tree.edges,
        branch_lengths={e: v * s for e, v in tree.branch_lengths.items()},
        node_seqs=tree.node_seqs, active_leaves=list(tree.active_leaves),
    )


class NativePhylaFlowMethod(Method):
    """Official PhylaFlow trees (topo+BL) + shared sequence adapter. Row: `phylaflow`."""

    name = "phylaflow"

    def __init__(self, pool_by_N: dict[int, list[str]], seq_adapter_fn):
        self.pool_by_N = pool_by_N
        self.seq_adapter_fn = seq_adapter_fn

    def generate(self, root_seq: str, N: int, H: float, seed: int) -> GeneratedTree:
        t0 = time.time()
        pool = self.pool_by_N.get(N)
        if not pool:
            raise ValueError(f"no PhylaFlow pool for N={N}")
        rng = random.Random(seed)
        tree = None
        for _ in range(12):
            cand = parse_phylaflow_tree(rng.choice(pool))
            if len(T.leaf_labels(cand)) == N:
                tree = cand
                break
        if tree is None:
            tree = parse_phylaflow_tree(rng.choice(pool))
        tree = rescale_to_horizon(tree, H)
        seqs = self.seq_adapter_fn(tree, root_seq, seed)
        tree = attach_sequences(tree, seqs, root_seq)
        return GeneratedTree(
            tree,
            {
                "runtime": time.time() - t0,
                "topology_source": "phylaflow_native",
                "adapter": "phylaflow-BL(rescaled-H) + shared-seq",
                "note": "posterior-basin samples; not root-conditioned forward gen",
            },
        )
