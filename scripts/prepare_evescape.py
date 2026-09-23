#!/usr/bin/env python3
"""Build a static [L, 20] EVEscape score tensor from the official CSV.

Writes a ``.pt`` matrix aligned to the pathogen sequence length. Use a
pathogen-matched matrix (Spike RBD vs flu HA) at eval time.

Example::

    python scripts/prepare_evescape.py --csv path/to/evescape.csv \
        --out data/covid/evescape_spike_rbd.pt --max-seq-len 1280
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from Bio import SeqIO

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

# Prefer official final EVEscape; never auto-pick fitness_eve / evol_indices.
POS_ALIASES = ["position", "pos", "i", "site", "wt_pos"]
WT_ALIASES = ["wildtype_aa", "wildtype", "wt_aa", "wt", "wildtype_res", "aa_wt"]
MUT_ALIASES = ["mutant_aa", "mutant", "mut_aa", "mut", "mutation_aa", "aa_mut"]
# "score" / "escape" last — too ambiguous; "fitness_eve" intentionally omitted.
SCORE_ALIASES = ["evescape_score", "evescape", "escape_score", "escape", "score"]

COMPONENT_COLS = (
    "fitness_eve",
    "accessibility_wcn",
    "dissimilarity_charge_hydro",
)


def _pick_col(header: list[str], aliases: list[str], override: str | None) -> str:
    if override:
        if override not in header:
            raise SystemExit(f"Column '{override}' not in CSV header {header}")
        return override
    lower = {h.lower(): h for h in header}
    for a in aliases:
        if a in lower:
            return lower[a]
    raise SystemExit(
        f"Could not auto-detect a column among {aliases} in header {header}. "
        f"Pass an explicit --*-col."
    )


def load_reference_seq(args) -> str:
    """Reference defining the TreeSBM column frame."""
    if args.ref_seq:
        return args.ref_seq.strip().upper()
    if args.ref_seq_file:
        text = Path(args.ref_seq_file).read_text().strip()
        if text.startswith(">"):
            # take first FASTA record
            lines = [ln.strip() for ln in text.splitlines() if ln and not ln.startswith(">")]
            return "".join(lines).upper()
        return text.replace("\n", "").replace(" ", "").upper()

    from src.dataset import parse_newick

    g = args.ref_from_group
    anc = ROOT / args.data / f"group_{g:03d}_anc_aa.fasta"
    # H1/H3 local layouts sometimes use <prefix>_group_XXX.fasta instead.
    if not anc.exists():
        alt = list((ROOT / args.data).glob(f"*group_{g:03d}*.fasta"))
        if alt:
            anc = alt[0]
    seqs = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(anc, "fasta")}
    try:
        nwk = ROOT / args.data / f"group_{g:03d}_rooted.nwk"
        if nwk.exists():
            root_id, _, _, _ = parse_newick(str(nwk))
            ref = seqs.get(root_id, "")
        else:
            ref = ""
    except Exception:
        ref = ""
    if len(ref) != args.max_seq_len:
        full = [s for s in seqs.values() if len(s) == args.max_seq_len]
        if not full:
            raise SystemExit(
                f"No length-{args.max_seq_len} sequence found in {anc}; "
                f"pass --ref-seq / --ref-seq-file explicitly."
            )
        ref = full[0]
    return ref


def read_csv_rows(args):
    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        pos_c = _pick_col(header, POS_ALIASES, args.pos_col)
        wt_c = _pick_col(header, WT_ALIASES, args.wt_col)
        mut_c = _pick_col(header, MUT_ALIASES, args.mut_col)
        score_c = _pick_col(header, SCORE_ALIASES, args.score_col)
        if score_c.lower() in {"fitness_eve", "evol_indices", "eve"}:
            raise SystemExit(
                f"Refusing score column '{score_c}'. That is raw EVE, not EVEscape. "
                f"Use column 'evescape' from summaries_with_scores/*.csv "
                f"(or pass --score-col evescape)."
            )
        print(f"Columns: pos={pos_c} wt={wt_c} mut={mut_c} score={score_c}")
        present_components = [c for c in COMPONENT_COLS if c in header]
        if present_components:
            print(f"Component columns present: {present_components}")
        rows = []
        for r in reader:
            try:
                pos = int(float(r[pos_c]))
                wt = r[wt_c].strip().upper()
                mut = r[mut_c].strip().upper()
                sc = float(r[score_c])
            except (ValueError, KeyError):
                continue
            if len(wt) != 1 or len(mut) != 1:
                continue
            comps = {}
            for c in present_components:
                raw = r.get(c, "")
                if raw in ("", None):
                    continue
                try:
                    comps[c] = float(raw)
                except ValueError:
                    pass
            rows.append((pos, wt, mut, sc, comps))
    return rows, score_c, present_components


def build_evescape_wt(rows) -> tuple[str, dict[int, int]]:
    """Reconstruct the EVEscape WT sequence and a position->wt_string_index map."""
    wt_by_pos: dict[int, str] = {}
    for pos, wt, _mut, _sc, _comps in rows:
        if wt in AA_TO_IDX:
            wt_by_pos.setdefault(pos, wt)
    positions = sorted(wt_by_pos)
    wt_seq = "".join(wt_by_pos[p] for p in positions)
    pos_to_wtidx = {p: i for i, p in enumerate(positions)}
    return wt_seq, pos_to_wtidx


def align_positions(evescape_wt: str, reference: str) -> dict[int, int]:
    """
    Map EVEscape-WT string index -> reference column via global protein alignment.
    Returns {wt_string_index: reference_column}.
    """
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = "global"
    aln = aligner.align(evescape_wt, reference)[0]

    mapping: dict[int, int] = {}
    blocks_a, blocks_b = aln.aligned
    for (a0, a1), (b0, b1) in zip(blocks_a, blocks_b):
        for k in range(a1 - a0):
            mapping[a0 + k] = b0 + k
    return mapping


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv-path", required=True, help="EVEscape per-substitution CSV")
    p.add_argument("--output", default="data/evescape_h1n1_ha.pt")
    p.add_argument("--max-seq-len", type=int, default=566)
    p.add_argument(
        "--pathogen",
        default="auto",
        choices=["auto", "flu_h1", "covid_spike_rbd", "other"],
        help="Stored in metadata; auto-inferred from output path / CSV name",
    )
    # reference frame
    p.add_argument("--data", default="data/train")
    p.add_argument(
        "--ref-from-group",
        type=int,
        default=1,
        help="group whose root defines the column frame",
    )
    p.add_argument("--ref-seq", default=None, help="explicit reference AA string")
    p.add_argument(
        "--ref-seq-file",
        default=None,
        help="path to plain AA string or FASTA with the reference",
    )
    # column overrides (else auto-detected)
    p.add_argument("--pos-col", default=None)
    p.add_argument("--wt-col", default=None)
    p.add_argument("--mut-col", default=None)
    p.add_argument(
        "--score-col",
        default=None,
        help="Must be the official 'evescape' column (default: auto-detect). "
             "Refuses fitness_eve / evol_indices.",
    )
    # normalization — default OFF for official EVEscape CSVs
    p.add_argument(
        "--standardize",
        action="store_true",
        default=False,
        help="z-score nonzero entries (NOT recommended for official evescape)",
    )
    p.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="keep official evescape values (default)",
    )
    p.add_argument(
        "--min-match-rate",
        type=float,
        default=0.75,
        help="abort if WT identity at mapped columns is below this "
             "(flu cross-strain often ~0.80–0.85; COVID RBD vs Wuhan ~1.0)",
    )
    p.add_argument(
        "--save-components",
        action="store_true",
        default=True,
        help="also store fitness_eve / accessibility_wcn / dissimilarity tensors",
    )
    p.add_argument(
        "--no-save-components",
        dest="save_components",
        action="store_false",
    )
    args = p.parse_args()

    L = args.max_seq_len
    reference = load_reference_seq(args)
    print(f"Reference length: {len(reference)}  head: {reference[:24]}")

    rows, score_c, present_components = read_csv_rows(args)
    if not rows:
        raise SystemExit("No usable rows parsed from CSV.")
    print(f"Parsed {len(rows)} (pos,wt,mut,score) rows")

    # sanity: official evescape is almost always negative / outside [0,1]
    score_vals = [r[3] for r in rows]
    s_mean = sum(score_vals) / len(score_vals)
    frac01 = sum(1 for v in score_vals if 0.0 <= v <= 1.0) / len(score_vals)
    print(
        f"Score col '{score_c}' raw stats: mean={s_mean:.4f} "
        f"range=[{min(score_vals):.4f},{max(score_vals):.4f}] "
        f"frac_in_[0,1]={frac01:.3f}"
    )
    if frac01 > 0.9 and s_mean > 0:
        print(
            "WARNING: scores look like probabilities in [0,1], not official "
            "log-EVEscape (expected mean ~−2). Check --score-col."
        )
    if s_mean < -5:
        print(
            "WARNING: mean < −5 looks like raw EVE (fitness_eve), not EVEscape. "
            "Use summaries_with_scores/*_evescape.csv column 'evescape'."
        )

    evescape_wt, pos_to_wtidx = build_evescape_wt(rows)
    print(f"EVEscape WT reconstructed: {len(evescape_wt)} residues")

    wtidx_to_col = align_positions(evescape_wt, reference)

    matched = total = 0
    for pos, widx in pos_to_wtidx.items():
        col = wtidx_to_col.get(widx)
        if col is None:
            continue
        total += 1
        if col < len(reference) and reference[col] == evescape_wt[widx]:
            matched += 1
    match_rate = matched / total if total else 0.0
    print(f"WT match rate at mapped columns: {match_rate:.3f} ({matched}/{total})")
    if match_rate < args.min_match_rate:
        raise SystemExit(
            f"Match rate {match_rate:.3f} < {args.min_match_rate}. The EVEscape "
            f"reference likely does not correspond to this frame — "
            f"check --ref-seq / pathogen (H1 vs Spike vs H3)."
        )
    if match_rate < 0.90:
        print(
            "NOTE: match_rate < 0.90 usually means strain divergence (e.g. WSN33 "
            "EVEscape WT vs seasonal H1 tree root), not a broken map. Homologous "
            "columns are still used for enrichment lookup."
        )

    scores = torch.zeros(L, 20, dtype=torch.float32)
    components = {
        c: torch.zeros(L, 20, dtype=torch.float32)
        for c in present_components
        if args.save_components
    }
    filled = skipped = 0
    for pos, wt, mut, sc, comps in rows:
        if mut not in AA_TO_IDX or wt == mut:
            skipped += 1
            continue
        widx = pos_to_wtidx.get(pos)
        col = wtidx_to_col.get(widx) if widx is not None else None
        if col is None or col >= L:
            skipped += 1
            continue
        scores[col, AA_TO_IDX[mut]] = sc
        for c, v in comps.items():
            if c in components:
                components[c][col, AA_TO_IDX[mut]] = v
        filled += 1
    print(f"Filled {filled} (col,aa) entries; skipped {skipped}")

    mean = std = 0.0
    if args.standardize:
        print(
            "WARNING: --standardize z-scores official EVEscape and changes the "
            "paper scale (COVID legacy tensors did this; prefer --no-standardize)."
        )
        nz = scores != 0
        if nz.any():
            vals = scores[nz]
            mean = float(vals.mean())
            std = float(vals.std()) or 1.0
            scores[nz] = (vals - mean) / std
            print(f"Standardized non-zero entries: mean={mean:.4f} std={std:.4f}")

    pathogen = args.pathogen
    if pathogen == "auto":
        blob = f"{args.csv_path} {args.output}".lower()
        if "flu_h1" in blob or "h1n1" in blob:
            pathogen = "flu_h1"
        elif "spike" in blob or "rbd" in blob or "covid" in blob:
            pathogen = "covid_spike_rbd"
        else:
            pathogen = "other"

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scores": scores,
        "reference_seq": reference,
        "positions": sorted(pos_to_wtidx),
        "match_rate": match_rate,
        "standardized": bool(args.standardize),
        "mean": mean if args.standardize else float(s_mean),
        "std": std,
        "score_col": score_c,
        "source_csv": str(Path(args.csv_path).resolve()),
        "pathogen": pathogen,
        "score_scale": (
            "official_evescape_log_product"
            if not args.standardize
            else "zscored_evescape"
        ),
        "score_scale_note": (
            "Official EVEscape = sum of log(logistic(z(component)/T)). "
            "Typically negative / outside [0,1]. Not raw EVE; not a probability."
        ),
    }
    for c, t in components.items():
        payload[c] = t
    torch.save(payload, out)
    nz = int((scores != 0).sum())
    print(
        f"Saved {out}  shape={tuple(scores.shape)}  nonzero={nz}  "
        f"range=[{scores[scores != 0].min():.3f},{scores[scores != 0].max():.3f}]  "
        f"pathogen={pathogen}  standardized={args.standardize}"
    )


if __name__ == "__main__":
    main()
