"""
Full end-to-end test: Tree → TreeEncoderInput → NodeEncoder → Hidden states
"""
import torch
from src.tree_state import TreeState
from src.treeencoder.tree_adapter import tree_state_to_encoder_input
from src.treeencoder.node_encoder import NodeEncoder


def test_full_pipeline():
    """End-to-end pipeline test with multiple tree sizes."""
    print("=" * 70)
    print("FULL END-TO-END PIPELINE TEST")
    print("=" * 70)

    test_cases = [
        {
            "name": "Single node (root only)",
            "nodes": ["root"],
            "edges": [],
            "branch_lengths": {},
            "n_nodes": 1,
        },
        {
            "name": "Root with 2 children",
            "nodes": ["root", "A", "B"],
            "edges": [("root", "A"), ("root", "B")],
            "branch_lengths": {("root", "A"): 0.1, ("root", "B"): 0.2},
            "n_nodes": 3,
        },
        {
            "name": "3-level tree (from dry run)",
            "nodes": ["root", "A", "B", "C"],
            "edges": [("root", "A"), ("root", "B"), ("A", "C")],
            "branch_lengths": {
                ("root", "A"): 0.1,
                ("root", "B"): 0.2,
                ("A", "C"): 0.05,
            },
            "n_nodes": 4,
        },
        {
            "name": "Larger tree (8 nodes)",
            "nodes": ["r", "a", "b", "c", "d", "e", "f", "g"],
            "edges": [
                ("r", "a"), ("r", "b"),
                ("a", "c"), ("a", "d"),
                ("b", "e"), ("b", "f"),
                ("c", "g"),
            ],
            "branch_lengths": {
                ("r", "a"): 0.1, ("r", "b"): 0.15,
                ("a", "c"): 0.05, ("a", "d"): 0.08,
                ("b", "e"): 0.12, ("b", "f"): 0.1,
                ("c", "g"): 0.03,
            },
            "n_nodes": 8,
        },
    ]

    results = []

    for test_case in test_cases:
        print(f"\n{'─' * 70}")
        print(f"TEST: {test_case['name']}")
        print(f"{'─' * 70}")

        # Create tree
        n_nodes = test_case["n_nodes"]
        node_ids = test_case["nodes"]
        edges = test_case["edges"]
        branch_lengths = test_case["branch_lengths"]

        node_seqs = {node: "AAAA" for node in node_ids}

        try:
            tree = TreeState(
                node_ids=node_ids,
                root_id=node_ids[0],
                edges=edges,
                branch_lengths=branch_lengths,
                node_seqs=node_seqs,
            )
            print(f"✓ Tree created: {n_nodes} nodes, {len(edges)} edges")
        except Exception as e:
            print(f"✗ Tree creation failed: {e}")
            results.append((test_case["name"], False))
            continue

        # Create embeddings
        d_plm = 320
        node_embeddings = torch.randn(n_nodes, d_plm)

        # Run through adapter
        try:
            encoder_input = tree_state_to_encoder_input(
                tree=tree,
                node_embeddings=node_embeddings,
                laplacian_dim=8,
            )
            print(f"✓ Encoder input created: x {encoder_input.x.shape}, "
                  f"struct {encoder_input.structural_features.shape}, "
                  f"lap_pe {encoder_input.lap_pe.shape}")
        except Exception as e:
            print(f"✗ Encoder input failed: {e}")
            results.append((test_case["name"], False))
            continue

        # Create and apply node encoder
        try:
            node_encoder = NodeEncoder(
                d_plm=d_plm,
                d_struct=3,
                d_laplacian=8,
                d_node=128,
                activation="relu",
            )
            h = node_encoder(
                encoder_input.x,
                encoder_input.structural_features,
                encoder_input.lap_pe,
            )
            print(f"✓ Node encoder applied: output shape {h.shape}")
        except Exception as e:
            print(f"✗ Node encoder failed: {e}")
            results.append((test_case["name"], False))
            continue

        # Validate
        try:
            assert h.shape == (n_nodes, 128), f"Expected shape ({n_nodes}, 128), got {h.shape}"
            assert torch.isfinite(h).all(), "Output contains NaN or Inf"
            print(f"✓ Output validated")
            results.append((test_case["name"], True))
        except AssertionError as e:
            print(f"✗ Validation failed: {e}")
            results.append((test_case["name"], False))

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8s} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED - Full pipeline is working!")
        return True
    else:
        print("\n✗ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = test_full_pipeline()
    exit(0 if success else 1)
