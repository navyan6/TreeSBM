#!/usr/bin/env python3
"""Export train-set topologies for external topology models.

Writes binary Newick pools sized by leaf count N for ARTreeFormer / PhyloVAE
training or sampling.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.heldout.build_examples import build_examples


def binarize(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Resolve every polytomy (node with >2 children) into an arbitrary cascade
    of binary splits via synthetic internal nodes. Real phylogenetic trees
    (fasttree/treetime output) can have genuine unresolved multifurcations at
    *any* internal node -- induced_subtree()'s collapse only removes degree-1
    pass-through nodes, it doesn't touch true polytomies, and
    ARTreeFormer/PhyloVAE's tensor encoding requires strictly binary internal
    structure everywhere (root aside, handled separately by
    unroot_for_trifurcating_root, which assumes its input is already binary).
    Arbitrary resolution is standard practice for this -- there's no branch-
    support signal available to prefer one resolution over another for a
    topology-prior training pool.
    """
    cm: dict[str, list[str]] = {}
    for p, c in edges:
        cm.setdefault(p, []).append(c)
    new_edges = []
    counter = [0]
    for p, kids in cm.items():
        if len(kids) <= 2:
            new_edges.extend((p, c) for c in kids)
            continue
        cur = p
        for c in kids[:-2]:
            new_edges.append((cur, c))
            nxt = f"_bin{counter[0]}"; counter[0] += 1
            new_edges.append((cur, nxt))
            cur = nxt
        new_edges.append((cur, kids[-2]))
        new_edges.append((cur, kids[-1]))
    return new_edges


def unroot_for_trifurcating_root(edges: list[tuple[str, str]], root: str
                                 ) -> tuple[list[tuple[str, str]], str] | None:
    """
    ARTreeFormer/PhyloVAE assume the unrooted-tree convention baked into their
    tensor shapes (edge_mask, vec2tree's init_tree): the encoding "root" must
    have exactly 3 neighbors, matching how an unrooted binary tree looks when
    anchored at an arbitrary internal point. Our induced_subtree() output is a
    normal *rooted* bifurcating tree (root has 2 children, occasionally 1 --
    see induced_subtree's own docstring: "root is always kept, it may end up
    with a single child"). Neither matches, and PhyloVAE's node_embedding()
    errors on the resulting ragged edge_index (ValueError: expected sequence
    of length 2, got 3) rather than silently producing something wrong.

    Fix: first collapse a degree-1 root down to the first real branch point
    (same idea as benchmarks/metrics/matched.py's _drop_unary_root), then drop
    the (now guaranteed >=2-children) root and reattach one of its subtrees as
    a third child of the other -- the standard "unroot a bifurcating tree"
    move. Which child hosts the reattachment doesn't matter (the *unrooted*
    topology is identical either way, "root" is just a tensor-encoding
    anchor) -- picks whichever is internal so the new root isn't a leaf.
    Returns None if the tree is too degenerate to unroot this way (both
    children of the branch point are leaves -- only possible at N=2, not a
    real case at our N>=16).
    """
    cm: dict[str, list[str]] = {}
    for p, c in edges:
        cm.setdefault(p, []).append(c)

    # collapse a degree-1 (unary) root chain down to the first real branch
    # point, tracking every unary node along the way so its now-stale edges
    # get dropped too (not just the final node's)
    r = root
    dropped_unary: set[str] = set()
    kids = cm.get(r, [])
    while len(kids) == 1:
        dropped_unary.add(r)
        r = kids[0]
        kids = cm.get(r, [])

    if len(kids) < 2:
        return None
    edges = [(p, c) for p, c in edges if p not in dropped_unary]
    if len(kids) > 2:
        return edges, r   # already trifurcating (or more), nothing else to do

    a, b = kids
    edges = [(p, c) for p, c in edges if p != r]
    if cm.get(a):
        return edges + [(a, b)], a
    if cm.get(b):
        return edges + [(b, a)], b
    return None   # both children are leaves -- degenerate (N=2)


def topology_newick(edges: list[tuple[str, str]], root: str, relabel: dict[str, str]) -> str:
    """Bare topology newick: nested parens + anonymized leaf names, no branch lengths/labels."""
    cm: dict[str, list[str]] = {}
    for p, c in edges:
        cm.setdefault(p, []).append(c)

    def render(n: str) -> str:
        kids = cm.get(n, [])
        if not kids:
            return relabel[n]
        return "(" + ",".join(render(c) for c in kids) + ")"

    return render(root) + ";"


def anonymize_leaves(edges: list[tuple[str, str]], root: str, n: int) -> dict[str, str]:
    """Map this example's actual leaf ids -> the fixed generic alphabet "0".."n-1",
    in the order leaves appear via preorder traversal (order is arbitrary but
    deterministic; what matters is every example uses the same n-leaf alphabet)."""
    cm: dict[str, list[str]] = {}
    for p, c in edges:
        cm.setdefault(p, []).append(c)
    leaves = []
    stack = [root]
    while stack:
        node = stack.pop()
        kids = cm.get(node, [])
        if not kids:
            leaves.append(node)
        else:
            stack.extend(reversed(kids))
    assert len(leaves) == n, f"expected {n} leaves, found {len(leaves)}"
    return {leaf: str(i) for i, leaf in enumerate(leaves)}


def write_trprobs(topologies: list[str], out_path: Path):
    """Uniform-weighted NEXUS trees block; literal leaf names (no Translate
    block needed -- validated against Bio.Phylo.parse(..., 'nexus'))."""
    n = len(topologies)
    w = 1.0 / n
    lines = ["#NEXUS", "Begin trees;"]
    for i, nwk in enumerate(topologies, start=1):
        lines.append(f"    tree tree_{i} = [&W {w:.10f}] [&U] {nwk}")
    lines.append("End;")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/h3n2/train")
    ap.add_argument("--N", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--per-tree", type=int, default=20,
                    help="Max sampled roots per tree per N (build_examples' max_roots_per_tree)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="benchmarks/external_pools")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for N in args.N:
        examples = build_examples(ROOT / args.data_dir, N, seed=args.seed,
                                  max_roots_per_tree=args.per_tree)
        lines = []
        skipped = 0
        for ex in examples:
            edges = [tuple(k.split("|")) for k in ex["target_branch_lengths"]]
            edges = binarize(edges)
            unrooted = unroot_for_trifurcating_root(edges, ex["root_id"])
            if unrooted is None:
                skipped += 1
                continue
            edges, root = unrooted
            relabel = anonymize_leaves(edges, root, N)
            lines.append(topology_newick(edges, root, relabel))

        nwk_path = out_dir / f"train_topologies_N{N}.nwk"
        nwk_path.write_text("\n".join(lines) + "\n")

        trprobs_path = out_dir / f"train_topologies_N{N}.trprobs"
        write_trprobs(lines, trprobs_path)

        print(f"N={N}: {len(lines)} topologies ({skipped} skipped as degenerate) from "
              f"{len({e['group'] for e in examples})} train trees -> "
              f"{nwk_path.name}, {trprobs_path.name}")


if __name__ == "__main__":
    main()
