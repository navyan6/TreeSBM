"""
Validation script: test the tree encoder pipeline on a small tree.
"""
import torch
from src.tree_state import TreeState
from src.treeencoder.tree_adapter import tree_state_to_encoder_input


def test_small_tree():
    """Test pipeline on a 4-node tree."""
    print("=" * 60)
    print("VALIDATING TREE ENCODER PIPELINE")
    print("=" * 60)

    # Create a small tree: root with two children, one child has a grandchild
    tree = TreeState(
        node_ids=["root", "A", "B", "C"],
        root_id="root",
        edges=[("root", "A"), ("root", "B"), ("A", "C")],
        branch_lengths={
            ("root", "A"): 0.1,
            ("root", "B"): 0.2,
            ("A", "C"): 0.05,
        },
        node_seqs={
            "root": "AAAA",
            "A": "AAAT",
            "B": "AATA",
            "C": "AATT",
        },
        active_leaves=["B", "C"],
    )

    print("\n1. Tree structure:")
    print(f"   Nodes: {tree.node_ids}")
    print(f"   Root: {tree.root_id}")
    print(f"   Edges: {tree.edges}")
    print(f"   N nodes: {tree.n_nodes()}")
    print(f"   N leaves: {tree.n_leaves()}")

    # Mock PLM embeddings (normally from ESM2 or similar)
    d_plm = 320  # ESM2 small is 320-dim
    node_embeddings = torch.randn(4, d_plm)
    print(f"\n2. PLM embeddings shape: {node_embeddings.shape}")

    # Run through pipeline
    print("\n3. Running tree_state_to_encoder_input...")
    laplacian_dim = 8
    try:
        encoder_input = tree_state_to_encoder_input(
            tree=tree,
            node_embeddings=node_embeddings,
            laplacian_dim=laplacian_dim,
        )
        print("   ✓ Success!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Validate shapes
    print("\n4. Validating output shapes:")
    checks = [
        ("node_ids", encoder_input.node_ids == tree.node_ids),
        ("x shape", encoder_input.x.shape == (4, d_plm)),
        ("structural_features shape", encoder_input.structural_features.shape == (4, 3)),
        ("lap_pe shape", encoder_input.lap_pe.shape == (4, laplacian_dim)),
        ("edge_index shape", encoder_input.edge_index.shape == (2, 6)),  # 3 edges * 2 directions
        ("edge_type shape", encoder_input.edge_type.shape == (6,)),
        ("edge_attr shape", encoder_input.edge_attr.shape == (6, 1)),
        ("root_index", encoder_input.root_index == 0),
    ]

    all_passed = True
    for name, check in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {name}")
        if not check:
            all_passed = False
            if name == "x shape":
                print(f"      Expected (4, {d_plm}), got {encoder_input.x.shape}")
            elif name == "structural_features shape":
                print(f"      Expected (4, 3), got {encoder_input.structural_features.shape}")
            elif name == "lap_pe shape":
                print(f"      Expected (4, {laplacian_dim}), got {encoder_input.lap_pe.shape}")
            elif name == "edge_index shape":
                print(f"      Expected (2, 6), got {encoder_input.edge_index.shape}")
            elif name == "edge_type shape":
                print(f"      Expected (6,), got {encoder_input.edge_type.shape}")
            elif name == "edge_attr shape":
                print(f"      Expected (6, 1), got {encoder_input.edge_attr.shape}")

    # Validate finiteness
    print("\n5. Validating tensor finiteness:")
    finiteness_checks = [
        ("x", torch.isfinite(encoder_input.x).all()),
        ("structural_features", torch.isfinite(encoder_input.structural_features).all()),
        ("lap_pe", torch.isfinite(encoder_input.lap_pe).all()),
        ("edge_attr", torch.isfinite(encoder_input.edge_attr).all()),
    ]

    for name, check in finiteness_checks:
        status = "✓" if check else "✗"
        print(f"   {status} {name} is finite")
        if not check:
            all_passed = False

    # Print sample data
    print("\n6. Sample data:")
    print(f"   structural_features (first 2 nodes):\n{encoder_input.structural_features[:2]}")
    print(f"   lap_pe (first 2 nodes, first 3 cols):\n{encoder_input.lap_pe[:2, :3]}")
    print(f"   edge_index (first 3 edges):\n{encoder_input.edge_index[:, :3]}")
    print(f"   edge_type (first 3):\n{encoder_input.edge_type[:3]}")
    print(f"   edge_attr (first 3):\n{encoder_input.edge_attr[:3]}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ VALIDATION PASSED - Pipeline works correctly!")
    else:
        print("✗ VALIDATION FAILED - Check errors above")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_small_tree()
    exit(0 if success else 1)
