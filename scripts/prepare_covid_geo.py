#!/usr/bin/env python3
"""Prepare a geographically split COVID Spike dataset.

Groups sequences by country (date-ordered chunks). Output: ``data/covid``.
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from Bio import SeqIO

VAL_COUNTRIES = {"Australia"}
TEST_COUNTRIES = {"Brazil"}
assert VAL_COUNTRIES.isdisjoint(TEST_COUNTRIES)

RAW_DIR = "data/covid/train"          # raw *_covid_seqs.fasta always land here;
                                       # split is decided per-country below, not by folder
MIN_GROUP = 20                         # drop country/remainder chunks smaller than this


def parse_date(raw: str) -> tuple[str, tuple[int, int, int]]:
    """'2020' / '2020-03' / '2020-03-15' -> (augur-style date string w/ XX, sort key)."""
    parts = raw.split("-")
    y = int(parts[0])
    if len(parts) == 1:
        return f"{y:04d}-XX-XX", (y, 7, 15)
    if len(parts) == 2:
        m = int(parts[1])
        return f"{y:04d}-{m:02d}-XX", (y, m, 15)
    m, d = int(parts[1]), int(parts[2])
    return f"{y:04d}-{m:02d}-{d:02d}", (y, m, d)


def parse_header(desc: str) -> tuple[str, str, str]:
    """
    >ACCESSION |description|date|length|country[|subregion] -> (accession, date, country).
    Country is always field index 4 -- some regions (e.g. Europe/Oceania/S.America)
    carry a trailing subregion field ("Germany|Germany: Bayern"), so it must NOT be
    read as the last field (that only worked by coincidence on the 5-field files).
    """
    parts = desc.split("|")
    return parts[0].strip(), parts[2].strip(), parts[4].strip()


def load_spike_records(raw_fasta: Path, spike_fasta: Path):
    headers = {}
    for rec in SeqIO.parse(raw_fasta, "fasta"):
        acc, date_raw, country = parse_header(rec.description)
        headers[acc] = (date_raw, country)

    out = []
    for rec in SeqIO.parse(spike_fasta, "fasta"):
        acc = rec.id.split()[0]
        if acc not in headers:
            continue
        date_raw, country = headers[acc]
        out.append((acc, date_raw, country, str(rec.seq)))
    return out


def assign_split(country: str) -> str:
    if country in VAL_COUNTRIES:
        return "val"
    if country in TEST_COUNTRIES:
        return "test"
    return "train"


def write_groups(records, out_dir: Path, prefix: str, group_size: int, start_group: int) -> int:
    """records: (acc, augur_date_str, sort_key, seq), single country, already date-ordered."""
    out_dir.mkdir(parents=True, exist_ok=True)
    g = start_group
    for i in range(0, len(records), group_size):
        chunk = records[i:i + group_size]
        if len(chunk) < MIN_GROUP:
            print(f"    dropping remainder of {len(chunk)} seqs (< MIN_GROUP={MIN_GROUP})")
            continue
        with open(out_dir / f"{prefix}_group_{g:03d}.fasta", "w") as f:
            for acc, date_str, _, seq in chunk:
                f.write(f">{acc},{date_str}\n{seq}\n")
        with open(out_dir / f"{prefix}_group_{g:03d}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "date"])
            for acc, date_str, _, seq in chunk:
                w.writerow([acc, date_str])
        g += 1
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", default="data/covid")
    ap.add_argument("--group-size", type=int, default=300)
    args = ap.parse_args()

    base = ROOT / args.out_base
    raw_dir = ROOT / RAW_DIR
    by_split_country: dict[str, dict[str, list]] = {"train": {}, "val": {}, "test": {}}
    seen_acc: set[str] = set()

    raw_sources = sorted(raw_dir.glob("*_covid_seqs.fasta"))
    if not raw_sources:
        print(f"No *_covid_seqs.fasta found in {raw_dir}"); return

    for raw_path in raw_sources:
        spike_path = raw_path.with_name(raw_path.stem.replace("_covid_seqs", "") + "_spike.fasta")
        if not spike_path.exists():
            print(f"WARNING: missing {spike_path.name} -- run "
                  f"covid_extract_spike.py on {raw_path.name} first. Skipping.")
            continue
        recs = load_spike_records(raw_path, spike_path)
        dup = bad_date = 0
        for acc, date_raw, country, seq in recs:
            if acc in seen_acc:
                dup += 1
                continue
            seen_acc.add(acc)
            try:
                date_str, sort_key = parse_date(date_raw)
            except (ValueError, IndexError):
                bad_date += 1
                continue
            split = assign_split(country)
            by_split_country[split].setdefault(country, []).append((acc, date_str, sort_key, seq))
        print(f"{raw_path.name}: {len(recs)} spike seqs "
              f"({dup} cross-file dup accessions, {bad_date} unparseable dates dropped)")

    for split, prefix in [("train", "covidtrain"), ("val", "covidval"), ("test", "covidtest")]:
        out_dir = base / split
        g, total = 1, 0
        for country, recs in sorted(by_split_country[split].items()):
            recs.sort(key=lambda r: r[2])
            g_before = g
            g = write_groups(recs, out_dir, prefix, args.group_size, g)
            total += len(recs)
            print(f"  [{split}] {country}: {len(recs)} seqs -> groups {g_before:03d}-{g - 1:03d}")
        print(f"=== {split}: {total} seqs, {g - 1} groups written to {out_dir} ===\n")

    print("Next: run_all_groups.py --data-dir data/covid/{split} --prefix covid{split}")


if __name__ == "__main__":
    main()
