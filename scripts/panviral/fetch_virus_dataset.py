#!/usr/bin/env python3
"""Panviral stage 2: download and form tree-ready groups for one virus taxon."""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
# Prefer PATH (conda env activate / TREESBM_PY's bin). Override with MAFFT_BIN.
MAFFT = os.environ.get("MAFFT_BIN") or shutil.which("mafft") or "mafft"

# Antigenic surface proteins first: those are the ones under immune selection,
# which is what TreeSBM is being asked to forecast. "polyprotein" is last and
# only ever a route into mat_peptide.
PRODUCT_PRIORITY = [
    "spike glycoprotein", "spike protein", "surface glycoprotein", "spike",
    "hemagglutinin-neuraminidase", "hemagglutinin", "haemagglutinin",
    "envelope glycoprotein", "fusion glycoprotein", "fusion protein",
    "glycoprotein precursor", "attachment glycoprotein",
    "major surface glycoprotein", "envelope protein", "glycoprotein",
    "capsid protein", "envelope", "polyprotein",
]

# Within a flavivirus-style polyprotein, the envelope protein is the antigenic
# target and the direct analogue of Spike / HA.
MATPEP_PRIORITY = [
    "envelope protein e", "envelope protein", "protein e", "glycoprotein e",
    "e protein", "envelope",
]

YEAR_RE = re.compile(r"\b(\d{4})\b")
AMBIG_OK = set("ACGT")


def _api_key() -> str | None:
    """
    NCBI key from the environment, or from a 0600 file the user writes once.

    Kept out of the command line on purpose: anything passed as an argument is
    visible in `ps` and lands in SLURM logs. Read straight from disk instead,
    and never echoed.
    """
    key = os.environ.get("NCBI_API_KEY")
    if key:
        return key.strip()
    path = Path(os.environ.get("NCBI_API_KEY_FILE", Path.home() / ".ncbi_api_key"))
    try:
        if path.exists():
            return path.read_text().strip() or None
    except Exception:  # noqa: BLE001 - unreadable key is not fatal
        pass
    return None


def eget(endpoint: str, params: dict, timeout: int = 300, tries: int = 5) -> str:
    key = _api_key()
    if key:
        params = {**params, "api_key": key}
    url = EUTILS + endpoint + "?" + urllib.parse.urlencode(params)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as fh:
                return fh.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - transient network / rate limit
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


# ── reference and protein of interest ───────────────────────────────────────

def _rank(product: str, table: list[str]) -> int:
    p = product.lower().strip()
    for i, kw in enumerate(table):
        if kw in p:
            return i
    return len(table)


def _target_from_record(rec) -> dict | None:
    """Choose the antigenic locus in one annotated record."""
    cds = [f for f in rec.features if f.type == "CDS"]
    if not cds:
        return None
    best = min(cds, key=lambda f: (_rank(f.qualifiers.get("product", ["?"])[0],
                                         PRODUCT_PRIORITY),
                                   -len(f)))
    product = best.qualifiers.get("product", ["?"])[0]
    feat = best
    polyprotein = "polyprotein" in product.lower()

    # Flavivirus-style: the CDS is the entire polyprotein, so step into the
    # mature peptides and take the envelope protein instead. Not every RefSeq
    # annotates them -- Zika's NC_012532.1 has a lone CDS and no mat_peptide,
    # while NC_035889.1 and the dengue references carry the full set -- which is
    # why pick_reference scores several candidates rather than trusting the first.
    if polyprotein:
        mats = [f for f in rec.features if f.type == "mat_peptide"]
        if mats:
            m = min(mats, key=lambda f: _rank(f.qualifiers.get("product", ["?"])[0],
                                              MATPEP_PRIORITY))
            if _rank(m.qualifiers.get("product", ["?"])[0], MATPEP_PRIORITY) < len(MATPEP_PRIORITY):
                feat = m
                product = m.qualifiers.get("product", ["?"])[0]
                polyprotein = False

    start, end = int(feat.location.start), int(feat.location.end)
    return {
        "accession": rec.id,
        "genome": str(rec.seq).upper(),
        "genome_len": len(rec.seq),
        "product": product,
        "cds_start": start,
        "cds_end": end,
        "cds_len": end - start,
        "unresolved_polyprotein": polyprotein,
    }


def pick_reference(taxid: int, sleep: float, candidates: int = 5) -> dict | None:
    """Find the annotated RefSeq whose antigenic protein is best resolved."""
    res = json.loads(eget("esearch.fcgi", {
        "db": "nuccore", "term": f"txid{taxid}[Organism:exp] AND srcdb_refseq[PROP]",
        "retmax": candidates, "retmode": "json",
    }))["esearchresult"]
    if not res.get("idlist"):
        return None
    time.sleep(sleep)

    gb = eget("efetch.fcgi", {"db": "nuccore", "id": ",".join(res["idlist"]),
                              "rettype": "gb", "retmode": "text"})
    time.sleep(sleep)

    best = None
    for rec in SeqIO.parse(io.StringIO(gb), "genbank"):
        cand = _target_from_record(rec)
        if cand is None:
            continue
        # Prefer a resolved single protein over a whole polyprotein, then the
        # highest-priority product name.
        key = (cand["unresolved_polyprotein"],
               _rank(cand["product"], PRODUCT_PRIORITY))
        if best is None or key < best[0]:
            best = (key, cand)
    return best[1] if best else None


# ── sequence retrieval ──────────────────────────────────────────────────────

def list_accessions(taxid: int, lo: int, hi: int, want: int, sleep: float,
                    hard_cap: int = 200000) -> tuple[list[str], int]:
    """
    Return a date-stratified sample of accessions, plus the true total.

    esearch hands back nuccore UIDs newest-first, so simply asking for the first
    N is a recency sample, not a sample of the virus. That silently wrecks a
    train/test split by collection year: capping measles at 600 returned 527
    post-2024 records and only 36 before, which is an artefact of the ordering
    rather than anything about measles. Pulling the whole (cheap) UID list and
    then striding through it evenly spreads the sample across the date range.
    """
    res = json.loads(eget("esearch.fcgi", {
        "db": "nuccore", "term": f"txid{taxid}[Organism:exp] AND {lo}:{hi}[SLEN]",
        "retmax": hard_cap, "retmode": "json",
    }))["esearchresult"]
    time.sleep(sleep)
    uids = res.get("idlist", [])
    total = int(res.get("count", len(uids)))
    if len(uids) <= want:
        return uids, total
    step = len(uids) / want
    return [uids[int(i * step)] for i in range(want)], total


def parse_records(gb_text: str) -> list[dict]:
    out = []
    for rec in SeqIO.parse(io.StringIO(gb_text), "genbank"):
        src = next((f for f in rec.features if f.type == "source"), None)
        q = src.qualifiers if src else {}
        date = (q.get("collection_date") or [None])[0]
        geo = (q.get("geo_loc_name") or q.get("country") or [None])[0]
        if not date or not geo:
            continue
        m = YEAR_RE.search(date)
        if not m:
            continue
        year = int(m.group(1))
        if not (1900 < year <= 2026):
            continue
        out.append({
            "accession": rec.id,
            "seq": str(rec.seq).upper(),
            "raw_date": date,
            "year": year,
            "country": geo.split(":")[0].strip(),
            "host": (q.get("host") or [""])[0],
        })
    return out


def fetch_records(uids: list[str], batch: int, sleep: float,
                  cache: Path | None) -> list[dict]:
    if cache and cache.exists():
        recs = [json.loads(l) for l in cache.open()]
        print(f"  cache hit: {len(recs):,} records", flush=True)
        return recs

    recs: list[dict] = []
    for i in range(0, len(uids), batch):
        chunk = uids[i:i + batch]
        try:
            gb = eget("efetch.fcgi", {"db": "nuccore", "id": ",".join(chunk),
                                      "rettype": "gb", "retmode": "text"})
            recs.extend(parse_records(gb))
        except Exception as e:  # noqa: BLE001 - skip a bad batch, keep going
            print(f"  ! batch {i}: {e}", flush=True)
        time.sleep(sleep)
        if (i // batch) % 10 == 0:
            print(f"  fetched {min(i + batch, len(uids)):,}/{len(uids):,} "
                  f"-> {len(recs):,} usable", flush=True)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
    return recs


# ── reference-frame extraction ──────────────────────────────────────────────

def extract_region(records: list[dict], ref: dict, chunk: int,
                   threads: int, max_ambig: float) -> list[dict]:
    """
    Align genomes into reference coordinates and slice the target CDS.

    `--keeplength` forces the output to the reference's column count, so
    insertions relative to the reference are dropped and a fixed column slice
    recovers the same locus in every sequence. This is the same coordinate
    discipline as the Spike extraction; getting it wrong is what produced the
    frame bug, so the slice stays in alignment coordinates throughout.
    """
    kept: list[dict] = []
    start, end = ref["cds_start"], ref["cds_end"]
    expected = end - start

    for i in range(0, len(records), chunk):
        part = records[i:i + chunk]
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            ref_fa, q_fa = tdp / "ref.fasta", tdp / "q.fasta"
            SeqIO.write([SeqRecord(Seq(ref["genome"]), id="__REF__",
                                   description="")], ref_fa, "fasta")
            SeqIO.write([SeqRecord(Seq(r["seq"]), id=r["accession"], description="")
                         for r in part], q_fa, "fasta")
            try:
                res = subprocess.run(
                    [MAFFT, "--6merpair", "--keeplength", "--thread", str(threads),
                     "--addfragments", str(q_fa), str(ref_fa)],
                    capture_output=True, text=True, timeout=7200,
                )
            except subprocess.TimeoutExpired:
                print(f"  ! mafft timeout on chunk {i}", flush=True)
                continue
            if res.returncode != 0:
                print(f"  ! mafft failed on chunk {i}: {res.stderr[-200:]}", flush=True)
                continue

            by_id = {r["accession"]: r for r in part}
            for rec in SeqIO.parse(io.StringIO(res.stdout), "fasta"):
                if rec.id == "__REF__":
                    continue
                src = by_id.get(rec.id)
                if src is None:
                    continue
                sub = str(rec.seq)[start:end].upper()
                nogap = sub.replace("-", "")
                # Tolerate real indels but reject fragments and junk.
                if not (0.8 * expected <= len(nogap) <= 1.05 * expected):
                    continue
                if sum(c not in AMBIG_OK for c in nogap) / max(len(nogap), 1) > max_ambig:
                    continue
                kept.append({**src, "cds": nogap})
        print(f"  extracted {min(i + chunk, len(records)):,}/{len(records):,} "
              f"-> {len(kept):,} passing QC", flush=True)
    return kept


# ── grouping ────────────────────────────────────────────────────────────────

def build_groups(records: list[dict], window: int, min_leaves: int,
                 max_leaves: int) -> dict[tuple[str, int], list[dict]]:
    buckets: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for r in records:
        buckets[(r["country"], r["year"] - r["year"] % window)].append(r)
    out = {}
    for key, rs in buckets.items():
        if len(rs) < min_leaves:
            continue
        rs.sort(key=lambda r: r["raw_date"])
        if len(rs) > max_leaves:
            step = len(rs) / max_leaves
            rs = [rs[int(i * step)] for i in range(max_leaves)]
        out[key] = rs
    return out


def snap_to_window(cutoff: int, window: int) -> int:
    """
    Move the cutoff onto a window boundary so no group straddles the split.

    Groups are `window`-year bins, but the cutoff is a single year, so an
    unsnapped cutoff can cut a bin in half: with cutoff 2018 and 2-year bins,
    the 2018-2019 bin sends its 2018 sequences to train and its 2019 sequences
    to test. For an outbreak still running across that boundary those are close
    relatives, which quietly undermines the claim that the test trees are
    unseen. Snapping down puts the whole contested bin in test, which keeps the
    holdout honest and is the conservative direction.
    """
    if window <= 1 or cutoff % window == window - 1:
        return cutoff
    return cutoff - (cutoff % window) - 1


def choose_cutoff(records: list[dict], fixed: int | None,
                  holdout_frac: float, window: int) -> int:
    """
    Pick the year that separates train from test.

    A single global cutoff does not suit every virus. SARS-CoV-2 and influenza
    are sequenced continuously, so 2024 leaves a healthy test band; Ebola and
    Zika are outbreak-driven and have almost nothing after 2024, so the same
    cutoff yields an empty test split. When no cutoff is given, hold out the
    most recent `holdout_frac` of sequences instead, which keeps the split
    strictly forward in time while adapting to each virus's sampling history.
    """
    years = sorted(r["year"] for r in records)
    if fixed is not None:
        return snap_to_window(fixed, window)
    if not years:
        return 2024
    idx = max(0, min(len(years) - 1, int(len(years) * (1 - holdout_frac))))
    cutoff = snap_to_window(max(years[idx] - 1, years[0]), window)
    # Snapping down can strand every sequence in test for a virus whose data
    # sits inside one bin; in that case take the next boundary up instead.
    if not any(y <= cutoff for y in years):
        cutoff = snap_to_window(cutoff + window, window)
    return cutoff


def norm_date(raw: str, year: int) -> str:
    """TreeTime accepts ambiguous dates as YYYY-XX-XX; only the year is certain."""
    try:
        from datetime import datetime
        for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%b-%Y", "%Y-%m", "%Y"):
            try:
                dt = datetime.strptime(raw, fmt)
            except ValueError:
                continue
            if fmt in ("%d-%b-%Y", "%Y-%m-%d"):
                return dt.strftime("%Y-%m-%d")
            if fmt in ("%b-%Y", "%Y-%m"):
                return dt.strftime("%Y-%m-XX")
            return f"{year}-XX-XX"
    except Exception:  # noqa: BLE001 - fall through to year-only
        pass
    return f"{year}-XX-XX"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taxid", type=int, required=True)
    ap.add_argument("--name", required=True, help="short slug, e.g. ebola")
    ap.add_argument("--out-root", type=Path, default=Path("data/panviral"))
    ap.add_argument("--train-cutoff", type=int, default=None,
                    help="collection year <= cutoff is train; later is test. "
                         "Omit to choose per virus from --holdout-frac.")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="fraction of the most recent sequences held out when "
                         "--train-cutoff is not given")
    ap.add_argument("--window", type=int, default=2, help="years per group")
    ap.add_argument("--min-leaves", type=int, default=30)
    ap.add_argument("--max-leaves", type=int, default=600)
    ap.add_argument("--max-records", type=int, default=20000)
    ap.add_argument("--max-ambig", type=float, default=0.02)
    ap.add_argument("--len-tol", type=float, default=0.3,
                    help="genome length window around the reference")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    print(f"=== {args.name} (taxid {args.taxid}) ===", flush=True)

    ref = pick_reference(args.taxid, args.sleep)
    if ref is None:
        sys.exit(f"no annotated RefSeq for taxid {args.taxid}")
    print(f"reference {ref['accession']} ({ref['genome_len']:,} nt)\n"
          f"target    {ref['product']!r} at {ref['cds_start']}-{ref['cds_end']} "
          f"({ref['cds_len']} nt, {ref['cds_len'] // 3} aa)", flush=True)

    lo = int(ref["genome_len"] * (1 - args.len_tol))
    hi = int(ref["genome_len"] * (1 + args.len_tol))
    uids, total = list_accessions(args.taxid, lo, hi, args.max_records, args.sleep)
    print(f"accessions in {lo:,}-{hi:,} nt: {total:,}"
          + (f" (date-stratified sample of {len(uids):,})" if len(uids) < total else ""),
          flush=True)
    if not uids:
        sys.exit("no genomes in length range")

    cache = args.out_root / "cache" / f"{args.name}_records.jsonl"
    records = fetch_records(uids, args.batch, args.sleep, cache)
    print(f"records with date + country: {len(records):,}", flush=True)
    if not records:
        sys.exit("no records carried both a collection date and a country")

    kept = extract_region(records, ref, args.chunk, args.threads, args.max_ambig)
    print(f"passing extraction QC: {len(kept):,}", flush=True)
    if not kept:
        sys.exit("no sequences survived extraction")

    cutoff = choose_cutoff(kept, args.train_cutoff, args.holdout_frac, args.window)
    train = [r for r in kept if r["year"] <= cutoff]
    test = [r for r in kept if r["year"] > cutoff]
    print(f"cutoff {cutoff} -> train {len(train):,}  test {len(test):,}"
          + ("" if args.train_cutoff else "  (chosen automatically)"), flush=True)

    manifest = {"virus": args.name, "taxid": args.taxid, "reference": ref["accession"],
                "product": ref["product"], "cds_len": ref["cds_len"],
                "train_cutoff": cutoff, "splits": {}}

    for split, rows in (("train", train), ("test", test)):
        groups = build_groups(rows, args.window, args.min_leaves, args.max_leaves)
        d = args.out_root / args.name / split
        d.mkdir(parents=True, exist_ok=True)
        info = []
        for gi, (key, rs) in enumerate(sorted(groups.items(),
                                              key=lambda kv: -len(kv[1])), 1):
            country, win = key
            SeqIO.write([SeqRecord(Seq(r["cds"]), id=r["accession"], description="")
                         for r in rs], d / f"{args.name}_group_{gi:03d}.fasta", "fasta")
            with (d / f"{args.name}_group_{gi:03d}.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["name", "date"])
                w.writeheader()
                for r in rs:
                    w.writerow({"name": r["accession"],
                                "date": norm_date(r["raw_date"], r["year"])})
            info.append({"group": gi, "country": country,
                         "window": f"{win}-{win + args.window - 1}",
                         "n": len(rs)})
        manifest["splits"][split] = info
        print(f"{split}: {len(info)} groups, {sum(i['n'] for i in info):,} leaves"
              + (f"  (largest: {info[0]['country']} {info[0]['window']}, "
                 f"{info[0]['n']})" if info else ""), flush=True)

    mpath = args.out_root / args.name / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {mpath}")


if __name__ == "__main__":
    main()
