#!/usr/bin/env python3
"""Evaluate VOC mutation recovery on generated or observed leaf sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from voc_threat_lib import (  # noqa: E402
    AA_TO_IDX,
    annotate_muts_evescape,
    find_root_seq,
    leaf_seqs,
    load_evescape,
    load_fasta_seqs,
    load_panel,
    load_wt_seq,
    muts_present,
    muts_vs_root,
    parse_mut,
    score_group_against_voc,
    voc_by_id,
)


def rbd_hamming(a: str, b: str, start1: int = 331, end1: int = 531) -> int | None:
    if len(a) < end1 or len(b) < end1:
        return None
    d = 0
    for pos in range(start1, end1 + 1):
        aa, bb = a[pos - 1], b[pos - 1]
        if aa not in AA_TO_IDX or bb not in AA_TO_IDX:
            continue
        if aa != bb:
            d += 1
    return d


def topk_union_stats(
    leaves: dict[str, str],
    root_seq: str,
    voc: dict,
    ref_seq: str | None,
    topk: int,
) -> dict:
    items = []
    for lid, seq in leaves.items():
        if ref_seq is not None:
            h = rbd_hamming(seq, ref_seq)
            if h is None:
                h = sum(1 for x, y in zip(seq, ref_seq) if x != y)
        else:
            h = sum(1 for x, y in zip(seq, root_seq) if x != y)
        items.append((h, lid, seq))
    items.sort(key=lambda t: t[0])
    top = items[: max(1, topk)]
    union = set()
    acq_union = set()
    per = []
    for h, lid, seq in top:
        hits = muts_present(seq, voc["signature"])
        acq = muts_vs_root(seq, root_seq, voc["signature"])
        union.update(hits)
        acq_union.update(acq)
        per.append(
            {
                "leaf": lid,
                "rbd_or_full_hamming": h,
                "sig_hits": hits,
                "acquired": acq,
                "n_sig": len(hits),
            }
        )
    n_sig = len(voc["signature"])
    return {
        "topk": len(top),
        "hamming_range": [top[0][0], top[-1][0]] if top else None,
        "union_sig": sorted(union),
        "union_acquired": sorted(acq_union),
        "union_exact_recall": len(union) / n_sig,
        "union_acquired_recall": len(acq_union) / n_sig,
        "leaves": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=None)
    ap.add_argument("--voc", required=True, help="VOC id from panel")
    ap.add_argument("--gen-fasta", type=Path, required=True)
    ap.add_argument("--obs-fasta", type=Path, default=None, help="Observed anc_aa for root/ref")
    ap.add_argument("--gen-nwk", type=Path, default=None)
    ap.add_argument("--obs-nwk", type=Path, default=None)
    ap.add_argument("--ref-tip", type=str, default=None, help="Observed tip id for top-k ranking")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--evescape", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    panel = load_panel(args.panel)
    voc = voc_by_id(panel, args.voc)
    wt = load_wt_seq()

    gen_seqs = load_fasta_seqs(args.gen_fasta)
    gen_root_id, gen_root = find_root_seq(
        gen_seqs, args.gen_nwk if args.gen_nwk else None
    )
    gen_leaves = leaf_seqs(gen_seqs)
    gen_leaves = {k: v for k, v in gen_leaves.items() if k != gen_root_id}

    obs_stats = None
    ref_seq = None
    obs_root = None
    if args.obs_fasta and args.obs_fasta.exists():
        obs_seqs = load_fasta_seqs(args.obs_fasta)
        obs_root_id, obs_root = find_root_seq(
            obs_seqs, args.obs_nwk if args.obs_nwk else None
        )
        obs_leaves = leaf_seqs(obs_seqs)
        obs_leaves = {k: v for k, v in obs_leaves.items() if k != obs_root_id}
        obs_stats = score_group_against_voc(obs_root, obs_leaves, voc)
        if args.ref_tip and args.ref_tip in obs_seqs:
            ref_seq = obs_seqs[args.ref_tip]
        else:
            # consensus-like: leaf with most signature hits
            best = None
            best_n = -1
            for lid, seq in obs_leaves.items():
                n = len(muts_present(seq, voc["signature"]))
                if n > best_n:
                    best_n = n
                    best = seq
            ref_seq = best

    # Prefer scoring acquisitions vs observed root when available (shared forecast setup)
    score_root = obs_root if obs_root is not None else gen_root
    gen_stats = score_group_against_voc(score_root, gen_leaves, voc)

    # Also report vs gen's own root
    gen_stats_own_root = score_group_against_voc(gen_root, gen_leaves, voc)

    topk = topk_union_stats(gen_leaves, score_root, voc, ref_seq, args.topk)

    scores, ev_meta = load_evescape(args.evescape)
    ev_sig = annotate_muts_evescape(voc["signature"], scores)
    recovered = [
        m for m, c in gen_stats["leaf_exact_counts"].items() if c > 0
    ]
    missing = [m for m in voc["signature"] if m not in recovered]
    ev_recovered = {m: ev_sig[m] for m in recovered}
    ev_missing = {m: ev_sig[m] for m in missing}

    # site recall vs Wuhan (any change at signature sites on ≥1 leaf)
    site_hits = 0
    for mut in voc["signature"]:
        _, pos, _ = parse_mut(mut)
        wt_aa = wt[pos - 1] if pos <= len(wt) else None
        hit = False
        for seq in gen_leaves.values():
            if pos <= len(seq) and wt_aa and seq[pos - 1] != wt_aa and seq[pos - 1] in AA_TO_IDX:
                hit = True
                break
        if hit:
            site_hits += 1

    out = {
        "voc_id": voc["id"],
        "pango": voc.get("pango"),
        "signature": voc["signature"],
        "defining_bundle": voc["defining_bundle"],
        "gen_fasta": str(args.gen_fasta),
        "obs_fasta": str(args.obs_fasta) if args.obs_fasta else None,
        "n_gen_leaves": len(gen_leaves),
        "score_root_source": "obs" if obs_root is not None else "gen",
        "gen_vs_score_root": gen_stats,
        "gen_vs_own_root": gen_stats_own_root,
        "obs": obs_stats,
        "voc_site_recall_vs_wuhan": site_hits / len(voc["signature"]),
        "topk": topk,
        "evescape_signature": ev_sig,
        "evescape_recovered": ev_recovered,
        "evescape_missing": ev_missing,
        "evescape_meta": {
            "standardized": ev_meta.get("standardized"),
            "score_col": ev_meta.get("score_col"),
            "pathogen": ev_meta.get("pathogen"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(
        f"{voc['id']}: exact_recall={gen_stats['voc_exact_mut_recall']:.3f} "
        f"acquired={gen_stats['voc_acquired_mut_recall']:.3f} "
        f"bundle_any={gen_stats['voc_bundle_any']} "
        f"topk{args.topk}_union={topk['union_exact_recall']:.3f} "
        f"({len(topk['union_sig'])}/{len(voc['signature'])})"
    )
    print("topk union:", ",".join(topk["union_sig"]) or "(none)")
    print("missing:", ",".join(missing) or "(none)")


if __name__ == "__main__":
    main()
