#!/usr/bin/env python3
"""
Main pipeline: Build H3N2 HA mutation landscape dataset using gget virus.

This script orchestrates:
1. Install dependencies
2. Query all H3N2 HA variants (human, avian, swine)
3. Build mutation landscapes for each sequence/time window
4. Compute statistics and save dataset

Usage:
    python run_pipeline.py [--test]
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def install_dependencies():
    """Install required packages."""
    print("Installing dependencies...")
    req_file = PROJECT_ROOT / "requirements.txt"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
        check=False
    )
    print("Dependencies installed.\n")


def main(test_mode=False):
    """Run the full pipeline."""
    print("=" * 70)
    print("TREE FLOWS: H3N2 HA MUTATION LANDSCAPE DATASET BUILDER")
    print("Using gget virus for efficient deterministic querying")
    print("Reference: https://arxiv.org/pdf/2606.06749")
    print("=" * 70)
    print()

    # Install dependencies
    install_dependencies()

    if test_mode:
        print("TEST MODE: Using synthetic sequences for demonstration\n")
        from src.build_dataset import create_demo_sequences, annotate_sequences_with_dates
        fasta_file = create_demo_sequences()
    else:
        print("PRODUCTION MODE: Querying NCBI Virus database\n")

        # Step 1: Query all H3N2 variants
        print("Step 1: Querying H3N2 HA sequences from NCBI...")
        print("-" * 70)
        from src.gget_virus_queries import fetch_all_h3n2_landscapes
        summary = fetch_all_h3n2_landscapes()
        print()

        # Step 2: Fetch and merge sequences
        print("Step 2: Fetching sequences...")
        print("-" * 70)
        from src.data_fetching import fetch_with_gget_virus
        fasta_file = fetch_with_gget_virus(limit=50000)

        if fasta_file is None:
            print("Warning: gget virus returned no results, falling back to NCBI Entrez...")
            from src.data_fetching import fetch_from_ncbi_flu_resource
            fasta_file = fetch_from_ncbi_flu_resource(limit=10000)
        print()

    # Step 3: Build mutation landscape dataset
    print("Step 3: Building mutation landscape dataset...")
    print("-" * 70)
    from src.build_dataset import build_mutation_landscape_dataset

    dataset = build_mutation_landscape_dataset(
        min_year=2005 if not test_mode else 2010,
        max_year=2024 if not test_mode else 2023,
        time_window_size=1,
        min_sequences_per_window=5,
    )

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Output directory: {PROJECT_ROOT / 'data' / 'landscapes'}")
    print()
    print("Generated files:")
    print("  - h3n2_ha_landscapes.json          [Mutation landscapes for all years]")
    print("  - dataset_report.txt               [Summary statistics]")
    print("  - h3n2_ha_annotated.csv            [Sequence metadata]")
    print()
    print("Next steps:")
    print("  1. Explore: python notebooks/eda.ipynb")
    print("  2. Train flow matching model on landscapes")
    print("  3. Evaluate predictions on held-out years")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build H3N2 HA mutation landscape dataset"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with synthetic sequences"
    )

    args = parser.parse_args()
    main(test_mode=args.test)
