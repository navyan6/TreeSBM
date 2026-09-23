"""Topology-prior baselines (ARTreeFormer, PhyloVAE, PhylaFlow).

Draw an unconditional topology of size N from a precomputed Newick pool,
assign branch lengths with the shared BranchLengthAdapter, and fill sequences
with the shared sequence adapter. Only the topology comes from the external model.
"""

from __future__ import annotations

import io
import random
from collections import deque

from Bio import Phylo

from src.tree_state import TreeState
from benchmarks.methods.base import Method, GeneratedTree, attach_sequences
from benchmarks.metrics import trees as T


def parse_topology(newick: str) -> TreeState:
    """Newick string -> bare TreeState (internal nodes named NODE_*; branch lengths ignored)."""
    tree = Phylo.read(io.StringIO(newick), "newick")
    edges, counter = [], [0]

    def name(clade):
        if clade.name:
            return str(clade.name)
        clade.name = f"NODE_{counter[0]:06d}"; counter[0] += 1
        return clade.name

    def walk(parent):
        pn = name(parent)
        for ch in parent.clades:
            cn = name(ch); edges.append((pn, cn)); walk(ch)

    walk(tree.root)
    root = name(tree.root)
    cm = {}
    for p, c in edges:
        cm.setdefault(p, []).append(c)
    node_ids, seen, q = [root], {root}, deque([root])
    while q:
        for c in cm.get(q.popleft(), []):
            if c not in seen:
                seen.add(c); node_ids.append(c); q.append(c)
    leaves = [n for n in node_ids if n not in cm]
    return TreeState(node_ids=node_ids, root_id=root, edges=edges,
                     branch_lengths={e: 0.0 for e in edges},
                     node_seqs={}, active_leaves=leaves)


class TopologyPriorMethod(Method):
    def __init__(self, name: str, pool_by_N: dict[int, list[str]],
                 bl_adapter, seq_adapter_fn):
        """
        name: e.g. "artreeformer_adapted" / "phylaflow_adapted".
        pool_by_N: sampled topologies from the external repo, keyed by N.
        bl_adapter: fitted BranchLengthAdapter.
        seq_adapter_fn: (topology, root_seq, seed) -> {node: seq} shared adapter.
        """
        self.name = name
        self.pool_by_N = pool_by_N
        self.bl_adapter = bl_adapter
        self.seq_adapter_fn = seq_adapter_fn

    def generate(self, root_seq: str, N: int, H: float, seed: int) -> GeneratedTree:
        pool = self.pool_by_N.get(N)
        if not pool:
            raise ValueError(f"no topology pool for N={N} ({self.name})")
        rng = random.Random(seed)
        topo = parse_topology(rng.choice(pool))
        if len(T.leaf_labels(topo)) != N:
            # external sampler produced a different size; reject up to a few times
            for _ in range(10):
                topo = parse_topology(rng.choice(pool))
                if len(T.leaf_labels(topo)) == N:
                    break
        topo = self.bl_adapter.assign(topo, H, seed)          # shared branch lengths -> H
        seqs = self.seq_adapter_fn(topo, root_seq, seed)      # shared sequence adapter
        tree = attach_sequences(topo, seqs, root_seq)
        return GeneratedTree(tree, {"topology_source": self.name,
                                    "adapter": "shared-BL + shared-seq"})
