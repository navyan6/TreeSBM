#!/usr/bin/env python3
"""
Stage 1 of the pan-viral pipeline: what eukaryotic viruses have enough data?

Builds the full NCBI virus taxonomy offline from the taxdump (one 79 MB
download, no per-taxon taxonomy calls), keeps eukaryote-infecting species, then
asks NCBI Datasets how many genomes exist for each.

Output is an inventory that stage 2 draws from. Resumable: every answered taxon
is appended to a JSONL cache and skipped on re-run, so an interrupted job picks
up where it stopped.

    python scripts/panviral/build_virus_inventory.py --min-count 150
    python scripts/panviral/build_virus_inventory.py --limit 200   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "data" / "panviral"

VIRUS_ROOT_TAXID = 10239
TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
API = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/virus/taxon/{}/dataset_report"

# Host range is not encoded in the taxonomy, so we exclude the clades that are
# unambiguously prokaryote- or archaea-infecting and keep the rest. Names are
# matched against the full lineage. This is deliberately conservative: a stray
# phage costs one wasted tree, a wrongly excluded eukaryotic virus costs a
# dataset we wanted.
NON_EUKARYOTIC = {
    # bacteriophage classes / families
    "Caudoviricetes", "Leviviricetes", "Microviridae", "Inoviridae",
    "Tectiviridae", "Corticoviridae", "Plasmaviridae", "Cystoviridae",
    "Sphaerolipoviridae", "Finnlakeviridae", "Autolykiviridae",
    "Faserviricetes", "Tubulavirales", "Vinavirales", "Norzivirales",
    "Timlovirales", "Petitvirales", "Malgrandaviricetes",
    # archaeal virus realms / families
    "Adnaviria", "Fuselloviridae", "Lipothrixviridae", "Rudiviridae",
    "Bicaudaviridae", "Ampullaviridae", "Globuloviridae", "Guttaviridae",
    "Clavaviridae", "Spiraviridae", "Portogloboviridae", "Ovaliviridae",
    "Thaspiviridae", "Turriviridae", "Halopanivirales", "Simuloviridae",
    "Matshushitaviridae", "Sphaerolipoviricetes",
    # prokaryote-infecting branches of Monodnaviria / Varidnaviria
    "Sangervirae", "Loebvirae", "Trapavirae", "Helvetiavirae",
    "Laserviricetes", "Tectiliviricetes"
}
COVID_FLU = {
    # explicitly exclude the two broad mammalian-virus clades we do not want in
    # the pan-viral benchmark inventory: influenza and SARS-/MERS-like coronaviruses.
    "Orthomyxovirales", "Orthomyxoviridae", "Influenza A virus", "Influenza B virus",
    "Influenza C virus", "Influenza D virus", "Coronaviridae", "Nidovirales",
    "Severe acute respiratory syndrome-related coronavirus",
    "Middle East respiratory syndrome-related coronavirus",
}


# ── taxonomy ────────────────────────────────────────────────────────────────

def ensure_taxdump(cache: Path) -> Path:
    d = cache / "taxdump"
    if (d / "nodes.dmp").exists() and (d / "names.dmp").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    tgz = cache / "taxdump.tar.gz"
    if not tgz.exists():
        print(f"downloading {TAXDUMP_URL} ...", flush=True)
        urllib.request.urlretrieve(TAXDUMP_URL, tgz)
    print("extracting nodes.dmp / names.dmp ...", flush=True)
    with tarfile.open(tgz) as tf:
        for member in ("nodes.dmp", "names.dmp"):
            tf.extract(member, path=d)
    return d


def load_taxonomy(d: Path) -> tuple[dict[int, int], dict[int, str], dict[int, str]]:
    parent: dict[int, int] = {}
    rank: dict[int, str] = {}
    with (d / "nodes.dmp").open() as fh:
        for line in fh:
            f = [x.strip() for x in line.split("|")]
            tid, pid = int(f[0]), int(f[1])
            parent[tid] = pid
            rank[tid] = f[2]
    name: dict[int, str] = {}
    with (d / "names.dmp").open() as fh:
        for line in fh:
            f = [x.strip() for x in line.split("|")]
            if f[3] == "scientific name":
                name[int(f[0])] = f[1]
    return parent, rank, name


def virus_descendants(parent: dict[int, int]) -> set[int]:
    """Every taxon under Viruses, found by walking each node up to a root."""
    children: dict[int, list[int]] = {}
    for tid, pid in parent.items():
        if tid != pid:
            children.setdefault(pid, []).append(tid)
    seen, stack = set(), [VIRUS_ROOT_TAXID]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        stack.extend(children.get(t, ()))
    return seen


def lineage_names(tid: int, parent: dict[int, int], name: dict[int, str]) -> list[str]:
    out, cur, guard = [], tid, 0
    while cur and guard < 100:
        out.append(name.get(cur, ""))
        nxt = parent.get(cur, cur)
        if nxt == cur:
            break
        cur = nxt
        guard += 1
    return out


def is_eukaryotic(tid: int, parent: dict[int, int], name: dict[int, str]) -> bool:
    lin = lineage_names(tid, parent, name)
    lin_norm = {str(x).lower() for x in lin if x}
    if lin_norm & {str(x).lower() for x in NON_EUKARYOTIC}:
        return False
    label = (lin[0] if lin else "").lower()
    if "phage" in label:
        return False
    # Exclude influenza and coronavirus taxa even when they are not labeled as
    # phages or obvious prokaryote-infecting lineages.
    if any(tok in label for tok in ("influenza", "coronavirus", "sars", "mers")):
        return False
    return True


# ── counts ──────────────────────────────────────────────────────────────────

def api_get(url: str, tries: int = 5) -> dict:
    key = os.environ.get("NCBI_API_KEY")
    if not key:
        kf = Path(os.environ.get("NCBI_API_KEY_FILE",
                                 Path.home() / ".ncbi_api_key"))
        key = kf.read_text().strip() if kf.exists() else None
    if key:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode({"api_key": key})
    delay = 1.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as fh:
                return json.loads(fh.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except Exception:  # noqa: BLE001 - transient network
            if attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return {}


def count_for_taxon(tid: int) -> int:
    data = api_get(API.format(tid) + "?page_size=1")
    return int(data.get("total_count", 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=OUT_DIR / "cache")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "virus_inventory.json")
    ap.add_argument("--counts-cache", type=Path, default=OUT_DIR / "cache" / "counts.jsonl")
    ap.add_argument("--min-count", type=int, default=150,
                    help="keep species with at least this many genomes")
    ap.add_argument("--rate", type=float, default=None,
                    help="requests per second (default 9 with NCBI_API_KEY, else 2.5)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only query this many species (smoke test)")
    ap.add_argument("--ranks", nargs="+", default=["species"])
    ap.add_argument("--exclude-covid-flu", action="store_true",
                    help="exclude influenza and SARS-/MERS-like coronaviruses?")
    args = ap.parse_args()

    if args.exclude_covid_flu:
        global NON_EUKARYOTIC
        NON_EUKARYOTIC |= COVID_FLU

    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rate = args.rate or (9.0 if os.environ.get("NCBI_API_KEY") else 2.5)
    interval = 1.0 / rate

    d = ensure_taxdump(args.cache)
    parent, rank, name = load_taxonomy(d)
    print(f"taxonomy: {len(parent):,} nodes", flush=True)

    viruses = virus_descendants(parent)
    print(f"virus taxa: {len(viruses):,}", flush=True)

    candidates = [
        t for t in viruses
        if rank.get(t) in args.ranks and is_eukaryotic(t, parent, name)
    ]
    candidates.sort(key=lambda t: name.get(t, ""))
    print(f"eukaryotic virus taxa at rank {args.ranks}: {len(candidates):,}", flush=True)

    done: dict[int, int] = {}
    if args.counts_cache.exists():
        with args.counts_cache.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    done[int(rec["tax_id"])] = int(rec["count"])
                except Exception:  # noqa: BLE001 - partial final line
                    continue
        print(f"resuming: {len(done):,} taxa already counted", flush=True)

    todo = [t for t in candidates if t not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"querying {len(todo):,} taxa at {rate:g} req/s "
          f"(~{len(todo) * interval / 60:.0f} min)", flush=True)

    t_start = time.time()
    with args.counts_cache.open("a") as fh:
        for i, tid in enumerate(todo, 1):
            try:
                n = count_for_taxon(tid)
            except Exception as e:  # noqa: BLE001 - record and move on
                print(f"  ! {tid} {name.get(tid,'')}: {e}", flush=True)
                n = -1
            done[tid] = n
            fh.write(json.dumps({"tax_id": tid, "name": name.get(tid, ""),
                                 "count": n}) + "\n")
            if i % 250 == 0:
                el = time.time() - t_start
                fh.flush()
                print(f"  {i:,}/{len(todo):,}  ({el/60:.1f} min elapsed, "
                      f"{len([v for v in done.values() if v >= args.min_count]):,} "
                      f"over threshold)", flush=True)
            time.sleep(interval)

    keep = [
        {"tax_id": t, "name": name.get(t, ""), "count": c,
         "lineage": lineage_names(t, parent, name)[1:6][::-1]}
        for t, c in done.items() if c >= args.min_count
    ]
    keep.sort(key=lambda r: -r["count"])

    args.out.write_text(json.dumps({
        "min_count": args.min_count,
        "n_candidates": len(candidates),
        "n_counted": len(done),
        "n_kept": len(keep),
        "total_genomes": sum(r["count"] for r in keep),
        "viruses": keep,
    }, indent=2) + "\n")

    print(f"\n{len(keep):,} species with >= {args.min_count} genomes "
          f"({sum(r['count'] for r in keep):,} sequences total)")
    print(f"\n{'rank':>5}  {'count':>9}  species")
    for i, r in enumerate(keep[:40], 1):
        print(f"{i:>5}  {r['count']:>9,}  {r['name']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
