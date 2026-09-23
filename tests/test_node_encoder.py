"""
Test node encoder on the small tree from validation pipeline.
"""
import torch
from src.tree_state import TreeState
from src.treeencoder.tree_adapter import tree_state_to_encoder_input
from src.treeencoder.node_encoder import NodeEncoder


def test_node_encoder():
    """Test node encoder on a 4-node tree."""
    print("=" * 60)
    print("TESTING NODE ENCODER")
    print("=" * 60)

    # Create the same small tree from validation
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

    # Create mock PLM embeddings (ESM2-small is 320-dim)
    d_plm = 320
    node_embeddings = torch.randn(4, d_plm)

    # Get encoder input
    print("\n1. Creating encoder input...")
    encoder_input = tree_state_to_encoder_input(
        tree=tree,
        node_embeddings=node_embeddings,
        laplacian_dim=8,
    )
    print("   ✓ Encoder input created")

    # Create node encoder
    print("\n2. Creating node encoder...")
    node_encoder = NodeEncoder(
        d_plm=d_plm,
        d_struct=3,
        d_laplacian=8,
        d_node=128,
        activation="relu",
    )
    print("   ✓ Node encoder created")

    # Encode nodes
    print("\n3. Encoding nodes...")
    try:
        h = node_encoder(
            x=encoder_input.x,
            structural_features=encoder_input.structural_features,
            lap_pe=encoder_input.lap_pe,
        )
        print("   ✓ Encoding successful")
    except Exception as e:
        print(f"   ✗ Encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Validate output
    print("\n4. Validating output:")
    checks = [
        ("Shape", h.shape == (4, 128)),
        ("Dtype", h.dtype == torch.float32),
        ("Finite", torch.isfinite(h).all()),
        ("Non-zero", (h.abs() > 0).any()),  # At least some non-zero values
    ]

    all_passed = True
    for name, check in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {name}")
        if not check:
            all_passed = False
            if name == "Shape":
                print(f"      Expected (4, 128), got {h.shape}")
            elif name == "Dtype":
                print(f"      Expected float32, got {h.dtype}")

    # Print sample output
    print("\n5. Sample node encodings (first 2 nodes, first 5 dims):")
    print(h[:2, :5])

    # Verify gradients flow
    print("\n6. Verifying gradient flow...")
    try:
        loss = h.sum()
        loss.backward()
        has_grads = node_encoder.projection.weight.grad is not None
        if has_grads:
            print("   ✓ Gradients computed successfully")
        else:
            print("   ✗ No gradients computed")
            all_passed = False
    except Exception as e:
        print(f"   ✗ Gradient computation failed: {e}")
        all_passed = False

    # Test different activations
    print("\n7. Testing different activations:")
    for activation in ["relu", "gelu", "none"]:
        try:
            encoder = NodeEncoder(
                d_plm=d_plm,
                d_struct=3,
                d_laplacian=8,
                d_node=128,
                activation=activation,
            )
            output = encoder(
                encoder_input.x,
                encoder_input.structural_features,
                encoder_input.lap_pe,
            )
            print(f"   ✓ {activation:6s} - output shape {output.shape}")
        except Exception as e:
            print(f"   ✗ {activation:6s} - {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ NODE ENCODER TESTS PASSED")
    else:
        print("✗ NODE ENCODER TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_node_encoder()
    exit(0 if success else 1)
