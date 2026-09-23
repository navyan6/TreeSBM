#!/usr/bin/env python3
"""Score generated trees with ESM-2 mean per-position PLL on leaf sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.r0_backends import AA_TO_IDX, build_r0_backend, normalize_backend_name


def parse_fasta(path: Path) -> list[tuple[str, str, str]]:
    """Return list of (full_header, node_id, seq). Header may be 'nid|role'."""
    records: list[tuple[str, str, str]] = []
    header = None
    chunks: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    full = header
                    nid = full.split("|", 1)[0]
                    records.append((full, nid, "".join(chunks)))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            full = header
            nid = full.split("|", 1)[0]
            records.append((full, nid, "".join(chunks)))
    return records


def newick_tip_ids(nwk: str) -> set[str]:
    """Rough extract of tip labels from Newick (labels before ':' or ',' or ')')."""
    tips: set[str] = set()
    i = 0
    n = len(nwk)
    while i < n:
        c = nwk[i]
        if c in "(),:; \n\t":
            i += 1
            continue
        j = i
        while j < n and nwk[j] not in "(),:;\n\t ":
            j += 1
        label = nwk[i:j]
        # internal NODE_ / after ':' branch lengths already skipped
        if label and not label.replace(".", "").replace("-", "").isdigit():
            # tip if next non-space is ':' or ',' or ')' — already at end of label
            tips.add(label)
        i = j
    return tips


def mean_pll(log_R0: torch.Tensor, seq: str, max_seq_len: int) -> float:
    vals = [
        float(log_R0[pos, AA_TO_IDX[aa]].item())
        for pos, aa in enumerate(seq[:max_seq_len])
        if aa in AA_TO_IDX
    ]
    return sum(vals) / len(vals) if vals else float("-inf")


def score_tree(
    fasta_path: Path,
    nwk_path: Path | None,
    r0_backend,
    max_seq_len: int,
    batch_size: int,
    leaves_only: bool = True,
) -> dict:
    records = parse_fasta(fasta_path)
    if leaves_only:
        # generate_tree.py writes ">node_id|leaf"
        records = [r for r in records if r[0].endswith("|leaf")]

    tip_ids = newick_tip_ids(nwk_path.read_text()) if nwk_path and nwk_path.exists() else set()

    leaf_pll: dict[str, float] = {}
    leaf_meta: dict[str, dict] = {}
    seqs = [r[2] for r in records]
    headers = [r[0] for r in records]
    node_ids = [r[1] for r in records]

    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        log_R0 = r0_backend.log_mutation_rates(chunk, max_seq_len=max_seq_len)
        for j, (full, nid, seq) in enumerate(
            zip(headers[start : start + batch_size], node_ids[start : start + batch_size], chunk)
        ):
            pll = mean_pll(log_R0[j], seq, max_seq_len)
            leaf_pll[full] = pll
            leaf_meta[full] = {
                "node_id": nid,
                "in_newick": nid in tip_ids if tip_ids else None,
                "length": len(seq),
                "pll": pll,
            }

    vals = [v for v in leaf_pll.values() if v == v and v != float("-inf")]
    vals_sorted = sorted(vals)
    summary = {
        "n_leaves": len(leaf_pll),
        "n_scored": len(vals),
        "mean": sum(vals) / len(vals) if vals else None,
        "min": vals_sorted[0] if vals_sorted else None,
        "max": vals_sorted[-1] if vals_sorted else None,
        "median": vals_sorted[len(vals_sorted) // 2] if vals_sorted else None,
    }
    n_mismatch = sum(1 for m in leaf_meta.values() if m["in_newick"] is False)

    return {
        "stem": fasta_path.stem,
        "fasta": str(fasta_path),
        "nwk": str(nwk_path) if nwk_path else None,
        "method": "esm_r0_mean_pll",
        "method_note": (
            "Mean per-position log p(aa) under one unmasked R0 forward "
            "(same proxy as generate_tree fitness gate / esm_pll_seq)."
        ),
        "max_seq_len": max_seq_len,
        "label_convention": {
            "fasta_header": "node_id|role  (role in {root,internal,leaf})",
            "json_leaf_keys": "full FASTA header (…|leaf)",
            "newick_tips": "node_id only (no |leaf suffix)",
        },
        "n_leaf_headers_missing_from_newick": n_mismatch,
        "summary": summary,
        "leaf_pll": leaf_pll,
        "leaves": leaf_meta,
    }


def default_max_len(path: Path) -> int:
    parts = {p.lower() for p in path.parts}
    if "covid" in parts:
        return 1280
    return 566


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=ROOT / "data" / "generated")
    ap.add_argument("--recursive", action="store_true", default=True)
    ap.add_argument("--no-recursive", action="store_false", dest="recursive")
    ap.add_argument("--pattern", default="*generated*.fasta")
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help="Override; default 1280 for covid paths else 566")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--r0-backend", default="esm2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    ap.add_argument("--limit", type=int, default=None, help="Max trees (debug)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backend_name = normalize_backend_name(args.r0_backend)
    print(f"R0 backend={backend_name} device={device}")
    r0 = build_r0_backend(backend_name, device=device)

    root = args.input_dir
    if args.recursive:
        fastas = sorted(root.rglob(args.pattern))
    else:
        fastas = sorted(root.glob(args.pattern))
    # Prefer sibling pairs that look like generate_tree outputs
    fastas = [p for p in fastas if "generated" in p.name]
    if args.limit is not None:
        fastas = fastas[: args.limit]

    print(f"Found {len(fastas)} FASTA files under {root}")
    for fa in fastas:
        out = fa.with_suffix("").with_suffix(".pll.json")
        # group_XXX_generated_matched.fasta -> group_XXX_generated_matched.pll.json
        out = fa.parent / (fa.stem + ".pll.json")
        if args.skip_existing and out.exists():
            print(f"SKIP {out.name}")
            continue
        nwk = fa.with_suffix(".nwk")
        L = args.max_seq_len or default_max_len(fa)
        print(f"Score {fa.relative_to(ROOT)}  L={L}")
        payload = score_tree(fa, nwk if nwk.exists() else None, r0, L, args.batch_size)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        s = payload["summary"]
        if s["mean"] is not None:
            print(f"  -> {out.name}  n={s['n_leaves']}  mean={s['mean']:.4f}")
        else:
            print(f"  -> {out.name}  empty")

    r0.close()
    print("Done.")


if __name__ == "__main__":
    main()
