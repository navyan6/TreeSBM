#!/usr/bin/env python3
"""Audit whether gap-aware ancestral retranslation is needed per dataset."""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

from Bio.Seq import Seq
from Bio import SeqIO

CODON_STOP = "*"


def is_leaf(name: str) -> bool:
    if "|" in name:
        return name.rsplit("|", 1)[1] == "leaf"
    return not name.startswith("NODE_") and name != "root"


def translate_at_offset(aln: str, offset: int) -> str:
    """Translate in ALIGNMENT coordinates starting at `offset`.

    Partial-gap codons become X, all-gap codons are dropped. This mirrors the
    gap-aware translator so the stop-codon counts below describe what the
    repair would actually produce, not an idealised version of it.
    """
    out = []
    for i in range(offset, len(aln) - 2, 3):
        cod = aln[i:i + 3]
        if cod == "---":
            continue
        if "-" in cod or "N" in cod:
            out.append("X")
            continue
        try:
            out.append(str(Seq(cod).translate()))
        except Exception:  # noqa: BLE001
            out.append("X")
    return "".join(out)


def internal_stops(prot: str) -> int:
    """Stops that are not the single expected terminal one."""
    p = prot.rstrip("X")
    if p.endswith(CODON_STOP):
        p = p[:-1]
    return p.count(CODON_STOP)


def gap_topology(seq: str) -> tuple[int, int, int, int]:
    """(terminal runs, terminal frameshifting, internal runs, internal frameshifting)"""
    n = len(seq)
    t = tfs = i = ifs = 0
    for m in re.finditer(r"-+", seq):
        terminal = m.start() == 0 or m.end() == n
        bad = len(m.group(0)) % 3 != 0
        if terminal:
            t += 1
            tfs += bad
        else:
            i += 1
            ifs += bad
    return t, tfs, i, ifs


def audit(d: Path, max_trees: int) -> dict | None:
    files = sorted(d.glob("group_*_anc_nt.fasta"))[:max_trees]
    if not files:
        return None

    stops = {0: 0, 1: 0, 2: 0}
    lens = {0: [], 1: [], 2: []}
    n_scored = 0
    topo = [0, 0, 0, 0]
    fs_terminal_only = fs_internal = 0
    width = None

    for f in files:
        recs = list(SeqIO.parse(f, "fasta"))
        if not recs:
            continue
        width = width or len(recs[0].seq)
        # score frame on full-coverage leaves only: ragged leaves add spurious X
        cands = [r for r in recs if is_leaf(r.id)]
        cands.sort(key=lambda r: str(r.seq).count("-"))
        for r in cands[:3]:
            s = str(r.seq).upper()
            n_scored += 1
            for off in (0, 1, 2):
                p = translate_at_offset(s, off)
                stops[off] += internal_stops(p)
                lens[off].append(len(p.replace("X", "")))
        for r in recs:
            s = str(r.seq).upper()
            if "-" not in s:
                continue
            t, tfs, i, ifs = gap_topology(s)
            topo[0] += t; topo[1] += tfs; topo[2] += i; topo[3] += ifs
            if ifs:
                fs_internal += 1
            elif tfs:
                fs_terminal_only += 1

    aa_files = sorted(d.glob("group_*_anc_aa.fasta"))[:max_trees]
    aa_lens: collections.Counter = collections.Counter()
    for f in aa_files:
        for r in SeqIO.parse(f, "fasta"):
            if is_leaf(r.id):
                aa_lens[len(str(r.seq).replace("-", ""))] += 1

    best = min(stops, key=lambda o: stops[o])
    return {
        "dir": str(d), "width": width, "n_scored": n_scored,
        "stops": stops, "best": best,
        "median_len": sorted(lens[best])[len(lens[best]) // 2] if lens[best] else 0,
        "aa_modal": aa_lens.most_common(1)[0] if aa_lens else (0, 0),
        "topo": topo, "fs_terminal_only": fs_terminal_only, "fs_internal": fs_internal,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--max-trees", type=int, default=25)
    args = ap.parse_args()

    for d in args.dirs:
        r = audit(d, args.max_trees)
        name = "/".join(str(d).split("/")[-2:])
        if r is None:
            print(f"\n=== {name} === (no anc_nt files)")
            continue
        print(f"\n=== {name} ===  width={r['width']} (width%3={r['width']%3})")
        s = r["stops"]
        for off in (0, 1, 2):
            mark = "  <-- best" if off == r["best"] else ""
            per = s[off] / max(r["n_scored"], 1)
            print(f"  offset {off}: {s[off]:>6} internal stops "
                  f"({per:6.2f}/seq){mark}")
        ratio = s[r["best"]] / max(min(s[o] for o in s if o != r["best"]), 1)
        verdict = ("UNAMBIGUOUS" if ratio < 0.2 else
                   "WEAK -- frames not clearly separated, inspect manually")
        print(f"  frame call: offset {r['best']} [{verdict}]")
        print(f"  translated protein median len {r['median_len']}, "
              f"existing aa modal len {r['aa_modal'][0]} (n={r['aa_modal'][1]})")
        t, tfs, i, ifs = r["topo"]
        print(f"  gap runs: {t} terminal ({tfs} frameshifting), "
              f"{i} internal ({ifs} frameshifting)")
        print(f"  sequences frameshifting from terminal raggedness only: "
              f"{r['fs_terminal_only']}; with internal frameshift: {r['fs_internal']}")


if __name__ == "__main__":
    main()
