#!/usr/bin/env python3
"""
Precompute reference mutation log-rates [N, L, 20] for all groups.

Supports multiple frozen R0 backends:
  --r0-backend esm2|esm2_650m|esmc|jtt|wag|lg|neutral|progen2|evo2

Default ``esm2`` writes legacy ``group_XXX_ref_rates.pt``.
Other backends write ``group_XXX_ref_rates_<tag>.pt`` (see src/r0_backends.py).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from Bio import SeqIO

from src.dataset import parse_newick
from src.r0_backends import (
    STUB_BACKENDS,
    build_r0_backend,
    cache_tag_for_backend,
    list_backends,
    normalize_backend_name,
    ref_rates_filename,
)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute R0 mutation log-rates for TreeSBM."
    )
    parser.add_argument("--data", default="data/train")
    parser.add_argument("--max-seq-len", type=int, default=566)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--r0-backend",
        default="esm2",
        help=f"Mutation prior backend. One of: {', '.join(list_backends())}.",
    )
    parser.add_argument(
        "--r0-model",
        default=None,
        help="Optional model id override (HF id for ESM-2, esmc_300m/esmc_600m for ESM-C).",
    )
    parser.add_argument(
        "--ref-rates-tag",
        default=None,
        help="Override cache filename tag (default derived from --r0-backend). "
             "Empty string forces legacy group_*_ref_rates.pt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if the cache file already exists.",
    )
    args = parser.parse_args()

    backend_name = normalize_backend_name(args.r0_backend)
    if backend_name in STUB_BACKENDS:
        raise SystemExit(
            f"Backend {backend_name!r} is stubbed and cannot precompute. "
            "See src/r0_backends.py."
        )

    tag = cache_tag_for_backend(backend_name, override=args.ref_rates_tag)
    data_dir = ROOT / args.data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"R0 backend: {backend_name}  model={args.r0_model or '(default)'}  tag={tag!r}")

    backend = build_r0_backend(backend_name, model_id=args.r0_model, device=device)

    groups = sorted([
        int(p.stem.split("_")[1])
        for p in data_dir.glob("group_*_rooted.nwk")
        if (data_dir / p.name.replace("_rooted.nwk", "_anc_aa.fasta")).exists()
    ])
    print(f"Found {len(groups)} complete groups\n")

    for g in groups:
        out_path = data_dir / ref_rates_filename(g, tag)
        if out_path.exists() and not args.overwrite:
            print(f"[{g:03d}] already cached ({out_path.name}), skipping")
            continue

        root_id, node_ids, _, _ = parse_newick(
            str(data_dir / f"group_{g:03d}_rooted.nwk")
        )
        del root_id
        seqs = {
            rec.id: str(rec.seq)
            for rec in SeqIO.parse(data_dir / f"group_{g:03d}_anc_aa.fasta", "fasta")
        }
        ref_len = len(next(iter(seqs.values())))
        for nid in node_ids:
            if nid not in seqs:
                seqs[nid] = "-" * ref_len
        sequences = [seqs[nid] for nid in node_ids]
        N, L = len(sequences), args.max_seq_len

        print(f"[{g:03d}] {N} sequences via {backend_name} ...", end=" ", flush=True)
        log_mut_rates = torch.zeros(N, L, 20, dtype=torch.float32)

        for start in range(0, N, args.batch_size):
            batch_seqs = sequences[start : start + args.batch_size]
            log_batch = backend.log_mutation_rates(batch_seqs, max_seq_len=L)
            log_mut_rates[start : start + len(batch_seqs)] = log_batch

        torch.save(
            {
                "node_ids": node_ids,
                "log_mut_rates": log_mut_rates,
                "r0_backend": backend_name,
                "r0_model": args.r0_model,
            },
            out_path,
        )
        print(f"done  shape={tuple(log_mut_rates.shape)} → {out_path.name}")

    backend.close()
    print("\nAll groups done.")


if __name__ == "__main__":
    main()
