"""Unit tests for ``benchmarks.metrics.trees``."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tree_state import TreeState
from benchmarks.metrics import trees as T


def _tree():
    return TreeState(
        node_ids=["root", "a", "b", "c", "d"],
        root_id="root",
        edges=[("root", "a"), ("root", "b"), ("a", "c"), ("a", "d")],
        branch_lengths={
            ("root", "a"): 1.0, ("root", "b"): 1.0,
            ("a", "c"): 1.0, ("a", "d"): 1.0,
        },
        node_seqs={n: "AA" for n in ["root", "a", "b", "c", "d"]},
        active_leaves=["b", "c", "d"],
    )


def test_leaves_and_depths():
    t = _tree()
    assert set(T.leaf_labels(t)) == {"b", "c", "d"}
    d = T.node_depths(t)
    assert d == {"root": 0, "a": 1, "b": 1, "c": 2, "d": 2}


def test_shape_stats():
    t = _tree()
    assert T.sackin_index(t) == 5
    assert T.cherry_count(t) == 1
    assert T.colless_index(t) == 1
    assert T.topological_height(t) == 2
    assert T.patristic_height(t) == 2.0
    # widths per depth: depth0=1(root), depth1=2(a,b), depth2=2(c,d) -> max 2
    assert T.tree_width(t) == 2


def test_clades_and_rf_self_zero():
    t = _tree()
    cl = T.clades(t)
    assert cl == {frozenset({"c", "d"})}   # {b,c,d} is the full set, excluded
    assert T.rf_distance(t, t) == 0
    assert T.normalized_rf(t, t) == 0.0


def test_rf_detects_topology_change():
    t1 = _tree()
    # t2: regroup so that (b,c) are the cherry under a, d attaches at root
    t2 = TreeState(
        node_ids=["root", "a", "b", "c", "d"],
        root_id="root",
        edges=[("root", "a"), ("root", "d"), ("a", "b"), ("a", "c")],
        branch_lengths={
            ("root", "a"): 1.0, ("root", "d"): 1.0,
            ("a", "b"): 1.0, ("a", "c"): 1.0,
        },
        node_seqs={n: "AA" for n in ["root", "a", "b", "c", "d"]},
        active_leaves=["b", "c", "d"],
    )
    # t1 informative clade {c,d}; t2 informative clade {b,c}; symmetric diff = 2
    assert T.rf_distance(t1, t2) == 2
    assert 0.0 < T.normalized_rf(t1, t2) <= 1.0


def test_rf_requires_matching_leafsets():
    t = _tree()
    t_small = TreeState(
        node_ids=["root", "c", "d"], root_id="root",
        edges=[("root", "c"), ("root", "d")],
        branch_lengths={("root", "c"): 1.0, ("root", "d"): 1.0},
        node_seqs={"root": "AA", "c": "AA", "d": "AA"},
        active_leaves=["c", "d"],
    )
    try:
        T.rf_distance(t, t_small)
        assert False, "expected ValueError on mismatched leaf sets"
    except ValueError:
        pass


def test_newick_roundtrips_leaves():
    t = _tree()
    nwk = T.to_newick(t)
    assert nwk.endswith(";")
    for leaf in ("b", "c", "d"):
        assert leaf in nwk
    # balanced parentheses
    assert nwk.count("(") == nwk.count(")")


def test_newick_can_omit_internal_names():
    t = _tree()
    nwk = T.to_newick(t, with_names=False)
    assert nwk.endswith(";")
    assert "root" not in nwk
    assert nwk.count("(") == nwk.count(")")


def test_ltt_starts_at_two_for_bifurcating_root():
    t = _tree()
    grid, counts = T.ltt(t, n_points=11)
    assert len(grid) == len(counts) == 11
    assert counts[0] == 2       # root bifurcation -> 2 lineages just after t=0
    assert max(counts) >= 2


if __name__ == "__main__":
    # Runnable without pytest: execute every test_* in this module.
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
