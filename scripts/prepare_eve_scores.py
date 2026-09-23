#!/usr/bin/env python3
"""Build a static [L, 20] EVE evolutionary-index tensor from CSV output."""

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from Bio import SeqIO

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

MUT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$", re.I)

INSTRUCTIONS = """
EVE score matrix not found. To produce data/h3n2/eve_ha.pt (or similar):

  git clone https://github.com/OATML-Markslab/EVE $LABHOME/baselines/EVE
  cd $LABHOME/baselines/EVE
  # follow examples/train_VAE.sh and examples/compute_evol_indices.sh for your protein
  # then:
  python scripts/prepare_eve_scores.py \\
      --csv-path /path/to/protein_20000_samples.csv \\
      --output data/h3n2/eve_ha.pt \\
      --data data/h3n2/train --ref-from-group 1 --max-seq-len 566

The CSV must contain mutation strings (e.g. M1A) and evol_indices (or --score-col).
"""


def load_reference_seq(args) -> str:
    if args.ref_seq:
        return args.ref_seq.strip().upper()
    from src.dataset import parse_newick

    g = args.ref_from_group
    anc = ROOT / args.data / f"group_{g:03d}_anc_aa.fasta"
    if not anc.exists():
        raise SystemExit(f"Reference FASTA not found: {anc}\n{INSTRUCTIONS}")
    seqs = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(anc, "fasta")}
    try:
        root_id, _, _, _ = parse_newick(str(ROOT / args.data / f"group_{g:03d}_rooted.nwk"))
        ref = seqs.get(root_id, "")
    except Exception:
        ref = ""
    if len(ref) != args.max_seq_len:
        full = [s for s in seqs.values() if len(s) == args.max_seq_len]
        if not full:
            raise SystemExit(
                f"No length-{args.max_seq_len} sequence in {anc}; pass --ref-seq.\n{INSTRUCTIONS}"
            )
        ref = full[0]
    return ref


def parse_mutation(raw: str):
    m = MUT_RE.match(raw.strip())
    if not m:
        return None
    wt, pos_s, mut = m.group(1).upper(), m.group(2), m.group(3).upper()
    return wt, int(pos_s), mut


def read_eve_csv(csv_path: Path, score_col: str | None):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        lower = {h.lower(): h for h in header}

        mut_col = lower.get("mutations") or lower.get("mutant")
        if not mut_col:
            raise SystemExit(
                f"CSV must have 'mutations' or 'mutant' column; got {header}\n{INSTRUCTIONS}"
            )
        sc_col = score_col
        if not sc_col:
            for alias in ("evol_indices", "model_score", "eve_score", "score"):
                if alias in lower:
                    sc_col = lower[alias]
                    break
        if not sc_col:
            raise SystemExit(
                f"Could not find score column in {header}; pass --score-col.\n{INSTRUCTIONS}"
            )

        rows = []
        for r in reader:
            parsed = parse_mutation(r[mut_col])
            if parsed is None:
                continue
            try:
                sc = float(r[sc_col])
            except (ValueError, KeyError):
                continue
            wt, pos, mut = parsed
            if wt in AA_TO_IDX and mut in AA_TO_IDX:
                rows.append((pos, wt, mut, sc))
        return rows, mut_col, sc_col


def build_eve_wt(rows) -> tuple[str, dict[int, int]]:
    wt_by_pos: dict[int, str] = {}
    for pos, wt, _mut, _sc in rows:
        wt_by_pos.setdefault(pos, wt)
    positions = sorted(wt_by_pos)
    wt_seq = "".join(wt_by_pos[p] for p in positions)
    pos_to_wtidx = {p: i for i, p in enumerate(positions)}
    return wt_seq, pos_to_wtidx


def align_positions(eve_wt: str, reference: str) -> dict[int, int]:
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = "global"
    aln = aligner.align(eve_wt, reference)[0]
    mapping: dict[int, int] = {}
    blocks_a, blocks_b = aln.aligned
    for (a0, a1), (b0, b1) in zip(blocks_a, blocks_b):
        for k in range(a1 - a0):
            mapping[a0 + k] = b0 + k
    return mapping


def main():
    p = argparse.ArgumentParser(description="Convert EVE CSV → [L,20] .pt for eval_eve_baseline.py")
    p.add_argument("--csv-path", required=True, help="EVE compute_evol_indices output CSV")
    p.add_argument("--output", default="data/eve_scores.pt")
    p.add_argument("--max-seq-len", type=int, default=566)
    p.add_argument("--data", default="data/train")
    p.add_argument("--ref-from-group", type=int, default=1)
    p.add_argument("--ref-seq", default=None)
    p.add_argument("--score-col", default=None)
    p.add_argument("--position-offset", type=int, default=0,
                   help="Add to EVE 1-based position before alignment (e.g. +16 for signal peptide)")
    p.add_argument("--min-match-rate", type=float, default=0.90)
    args = p.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        raise SystemExit(f"EVE CSV not found: {csv_path}\n{INSTRUCTIONS}")

    L = args.max_seq_len
    reference = load_reference_seq(args)
    print(f"Reference length: {len(reference)}  head: {reference[:24]}")

    rows, mut_col, sc_col = read_eve_csv(csv_path, args.score_col)
    if not rows:
        raise SystemExit(f"No parseable rows in {csv_path} (columns {mut_col}, {sc_col}).\n{INSTRUCTIONS}")
    print(f"Parsed {len(rows)} mutations from {csv_path}  (cols: {mut_col}, {sc_col})")

    if args.position_offset:
        rows = [(pos + args.position_offset, wt, mut, sc) for pos, wt, mut, sc in rows]

    eve_wt, pos_to_wtidx = build_eve_wt(rows)
    print(f"EVE WT reconstructed: {len(eve_wt)} residues")
    wtidx_to_col = align_positions(eve_wt, reference)

    matched = total = 0
    for pos, widx in pos_to_wtidx.items():
        col = wtidx_to_col.get(widx)
        if col is None:
            continue
        total += 1
        if col < len(reference) and reference[col] == eve_wt[widx]:
            matched += 1
    match_rate = matched / total if total else 0.0
    print(f"WT match rate at mapped columns: {match_rate:.3f} ({matched}/{total})")
    if match_rate < args.min_match_rate:
        raise SystemExit(
            f"Match rate {match_rate:.3f} < {args.min_match_rate}. "
            f"Try --ref-seq, --position-offset, or a different EVE MSA.\n{INSTRUCTIONS}"
        )

    scores = torch.zeros(L, 20, dtype=torch.float32)
    filled = skipped = 0
    for pos, wt, mut, sc in rows:
        if wt == mut:
            skipped += 1
            continue
        widx = pos_to_wtidx.get(pos)
        col = wtidx_to_col.get(widx) if widx is not None else None
        if col is None or col >= L:
            skipped += 1
            continue
        scores[col, AA_TO_IDX[mut]] = sc
        filled += 1
    print(f"Filled {filled} (col,aa) entries; skipped {skipped}")

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scores": scores,
            "reference_seq": reference,
            "match_rate": match_rate,
            "source_csv": str(csv_path.resolve()),
        },
        out,
    )
    nz = int((scores != 0).sum())
    print(f"Saved {out}  shape={tuple(scores.shape)}  nonzero={nz}  "
          f"range=[{scores.min():.3f},{scores.max():.3f}]")


if __name__ == "__main__":
    main()
