#!/usr/bin/env python3
"""
Inventory NCBI Virus / InfluenzaDB counts for the M3 Tier-1 protein panel.

Writes data/m3_panel/PANEL_COUNTS.json with per-gene query specs, length
windows, and (when network is available) Entrez hit counts.

Usage:
  python scripts/audit_m3_panel_ncbi.py              # dry counts via esearch
  python scripts/audit_m3_panel_ncbi.py --dry-run    # write specs only, no NCBI
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "m3_panel" / "PANEL_COUNTS.json"

# Tier-1 panel (see M3 plan). HIV PR excluded (L~99). CoV Mpro included.
PANEL = [
    {
        "gene_id": "h3n2_ha",
        "class": "glycoprotein",
        "organism": "Influenza A H3N2",
        "max_seq_len": 566,
        "source": "ncbi_influenza",
        "query": '"Influenza A virus"[Organism] AND H3N2[All Fields] AND HA[Gene Name] AND "Homo sapiens"[Host] AND 1924:2026[PDAT]',
        "status": "have_local",
    },
    {
        "gene_id": "h1n1_ha",
        "class": "glycoprotein",
        "organism": "Influenza A H1N1",
        "max_seq_len": 566,
        "source": "ncbi_influenza",
        "query": '"Influenza A virus"[Organism] AND H1N1[All Fields] AND HA[Gene Name] AND "Homo sapiens"[Host] AND 1924:2026[PDAT]',
        "status": "have_local",
    },
    {
        "gene_id": "flub_ha",
        "class": "glycoprotein",
        "organism": "Influenza B",
        "max_seq_len": 566,
        "source": "ncbi",
        "query": '"Influenza B virus"[Organism] AND HA[Title] AND 1400:2000[SLEN] AND 2009:2026[PDAT]',
        "status": "ingested",
    },
    {
        "gene_id": "sarscov2_spike",
        "class": "glycoprotein",
        "organism": "SARS-CoV-2",
        "max_seq_len": 1280,
        "source": "ncbi_virus",
        "query": 'txid2697049[Organism:exp] AND spike[Protein Name] AND "Homo sapiens"[Host]',
        "status": "have_local",
    },
    {
        "gene_id": "hiv1_env",
        "class": "glycoprotein",
        "organism": "HIV-1",
        "max_seq_len": 900,
        "source": "ncbi_virus",
        "query": 'txid11676[Organism:exp] AND env[Gene Name] AND "Homo sapiens"[Host]',
        "status": "have_local",
    },
    {
        "gene_id": "h3n2_pb2",
        "class": "polymerase",
        "organism": "Influenza A H3N2",
        "max_seq_len": 770,
        "source": "ncbi_influenza",
        "query": '"Influenza A virus"[Organism] AND H3N2[All Fields] AND PB2[Gene Name] AND "Homo sapiens"[Host] AND 1924:2026[PDAT]',
        "status": "to_ingest",
    },
    {
        "gene_id": "h1n1_pb2",
        "class": "polymerase",
        "organism": "Influenza A H1N1",
        "max_seq_len": 770,
        "source": "ncbi_influenza",
        "query": '"Influenza A virus"[Organism] AND H1N1[All Fields] AND PB2[Gene Name] AND "Homo sapiens"[Host] AND 1924:2026[PDAT]',
        "status": "to_ingest",
    },
    {
        "gene_id": "sarscov2_nsp12",
        "class": "polymerase",
        "organism": "SARS-CoV-2",
        "max_seq_len": 940,
        "source": "ncbi_virus",
        "query": 'txid2697049[Organism:exp] AND (RdRp[Protein Name] OR nsp12[Protein Name] OR "RNA-dependent RNA polymerase"[Protein Name]) AND "Homo sapiens"[Host]',
        "status": "to_ingest",
        "note": "Prefer mature nsp12 / RdRp; else slice ORF1ab at Wuhan Hu-1 coords",
    },
    {
        "gene_id": "hiv1_rt",
        "class": "polymerase",
        "organism": "HIV-1",
        "max_seq_len": 560,
        "source": "ncbi_virus",
        "query": 'txid11676[Organism:exp] AND pol[Gene Name] AND "Homo sapiens"[Host]',
        "status": "to_ingest",
        "note": "Extract RT window (HXB2 p51) from Pol polyprotein",
    },
    {
        "gene_id": "filo_l",
        "class": "polymerase",
        "organism": "Filoviridae",
        "max_seq_len": 900,
        "source": "ncbi+nextstrain",
        "query": '("Ebolavirus"[Organism] OR "Marburgvirus"[Organism]) AND L[Gene Name] AND 2000:2026[PDAT]',
        "status": "paused_ingest",
        "note": "900-aa conserved window; Pathoplexus 2026 test-only",
    },
    {
        "gene_id": "h3n2_np",
        "class": "nucleoprotein",
        "organism": "Influenza A H3N2",
        "max_seq_len": 510,
        "source": "ncbi_influenza",
        "query": '"Influenza A virus"[Organism] AND H3N2[All Fields] AND NP[Gene Name] AND "Homo sapiens"[Host] AND 1924:2026[PDAT]',
        "status": "to_ingest",
    },
    {
        "gene_id": "sarscov2_n",
        "class": "nucleoprotein",
        "organism": "SARS-CoV-2",
        "max_seq_len": 440,
        "source": "ncbi_virus",
        "query": 'txid2697049[Organism:exp] AND (nucleocapsid[Protein Name] OR "N protein"[Protein Name]) AND "Homo sapiens"[Host]',
        "status": "to_ingest",
    },
    {
        "gene_id": "sarscov2_mpro",
        "class": "protease",
        "organism": "SARS-CoV-2",
        "max_seq_len": 320,
        "source": "ncbi_virus",
        "query": 'txid2697049[Organism:exp] AND (Mpro[Protein Name] OR "3C-like proteinase"[Protein Name] OR nsp5[Protein Name]) AND "Homo sapiens"[Host]',
        "status": "to_ingest",
        "note": "Core panel; slice nsp5 from ORF1ab if needed. HIV PR (~99 AA) excluded.",
    },
]


def esearch_count(query: str, db: str = "protein") -> int | None:
    """Hit NCBI E-utilities via urllib (no Biopython required).

    On macOS Python.org installs, SSL often fails with CERTIFICATE_VERIFY_FAILED
    until `Install Certificates.command` is run (or certifi is used). Prefer
    running this on a machine with a working system CA store.
    """
    import ssl
    import urllib.error
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    params = urllib.parse.urlencode(
        {
            "db": db,
            "term": query,
            "retmax": 0,
            "retmode": "xml",
            "email": "nnori@upenn.edu",
            "tool": "DiscreteTreeFlows_m3_audit",
        }
    )
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}"
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=60) as resp:
            xml_text = resp.read()
        root = ET.fromstring(xml_text)
        count = root.findtext("Count")
        return int(count) if count is not None else None
    except Exception as e:  # noqa: BLE001
        print(f"  esearch failed: {type(e).__name__}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip NCBI; write query specs only")
    ap.add_argument("--db", default="protein", help="Entrez db for counts (protein|nucleotide)")
    args = ap.parse_args()

    rows = []
    for gene in PANEL:
        entry = dict(gene)
        entry["esearch_db"] = args.db
        if args.dry_run:
            entry["n_hits"] = None
            entry["queried_at"] = None
        else:
            print(f"=== {gene['gene_id']} ===")
            n = esearch_count(gene["query"], db=args.db)
            entry["n_hits"] = n
            entry["queried_at"] = datetime.now(timezone.utc).isoformat()
            print(f"  n_hits={n}")
            time.sleep(0.4)
        rows.append(entry)

    payload = {
        "panel": "m3_tier1",
        "length_gate_aa": [250, 1400],
        "host": "Homo sapiens",
        "date_universe": "1924-present (flu); organism-specific for CoV/HIV/filo",
        "exclude": ["hiv1_pr (~99 AA)"],
        "genes": rows,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
