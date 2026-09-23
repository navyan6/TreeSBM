#!/usr/bin/env python3
"""Shared helpers for HIV Env prep (region detection, frame notes, dating)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from Bio.Seq import Seq

# Near-complete HIV-1 Env CDS (HXB2 Env = 2568 nt / 856 AA).
MIN_ENV_NT = 2200
MAX_ENV_NT = 2800
MIN_ENV_AA = 700
MAX_ENV_AA = 950

# HXB2 Env variable loops (1-based inclusive).
HXB2_V_REGIONS: dict[str, tuple[int, int]] = {
    "V1": (131, 157),
    "V2": (158, 196),
    "V3": (296, 331),
    "V4": (385, 418),
    "V5": (461, 471),
}

_ENV_NTERM_OK = re.compile(
    r"^(MRV|MKV|MRG|MQP|MRA|MKA|MRW|MKW|MRI|MKI)"
)
_MATURE_GP120 = "VENLWVTVYY"


@dataclass
class HivRecord:
    acc: str
    desc: str
    location_raw: str
    country: str
    date_raw: str
    date_str: str
    sort_key: tuple[int, int, int]
    unit: str
    has_state: bool
    nt: str
    aa: str
    frame_note: str
    source_file: str


def normalize_location(field: str) -> str:
    s = field.strip()
    s = re.sub(r"\s+state\b", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def geo_unit(location: str, country: str) -> tuple[str, bool]:
    """
    Finest unit with enough signal: state/admin when LOCATION has 'Country: X,...';
    else country. Skip free-text regions without a clear geo token.
    """
    loc = normalize_location(location)
    country = country.strip() or loc
    if ":" in loc:
        right = loc.split(":", 1)[1].strip()
        state = right.split(",")[0].strip()
        if state and state.lower() not in {country.lower(), ""}:
            return f"{country}: {state}", True
    return country, False


def parse_date(raw: str) -> tuple[str, tuple[int, int, int]]:
    """
    Parse GenBank collection dates. Raises ValueError on empty / range / junk.
    Accepts YYYY, YYYY-MM, YYYY-MM-DD only (no 2015/2018 ranges).
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty date")
    if "/" in raw:
        raise ValueError(f"range/ambiguous date: {raw!r}")
    parts = raw.split("-")
    y = int(parts[0])
    if y < 1980 or y > 2030:
        raise ValueError(f"year out of range: {raw!r}")
    if len(parts) == 1:
        return f"{y:04d}-XX-XX", (y, 7, 15)
    if len(parts) == 2:
        m = int(parts[1])
        return f"{y:04d}-{m:02d}-XX", (y, m, 15)
    m, d = int(parts[1]), int(parts[2])
    return f"{y:04d}-{m:02d}-{d:02d}", (y, m, d)


def parse_header(desc: str) -> tuple[str, str, str, str, str]:
    """
    -> (acc, description, location, country, date_raw)
    Field layout: ACC |desc|organism|LOCATION|COUNTRY|DATE|LENGTH
    Some descriptions embed extra '|' (africa 8-field); LOCATION/COUNTRY/DATE
    are always the last three fields before LENGTH.
    """
    parts = [p.strip() for p in desc.split("|")]
    if len(parts) < 7:
        raise ValueError(f"expected >=7 pipe fields, got {len(parts)}")
    acc = parts[0].split()[0].lstrip(">")
    # last field = LENGTH; date=[-2], country=[-3], location=[-4]
    location = parts[-4]
    country = parts[-3]
    date_raw = parts[-2]
    description = "|".join(parts[1:-4]) if len(parts) > 7 else parts[1]
    return acc, description, location, country, date_raw


def is_env_description(desc: str) -> bool:
    d = desc.lower()
    return ("envelope" in d) or ("(env)" in d) or bool(re.search(r"\benv\b", d))


def is_genome_description(desc: str) -> bool:
    d = desc.lower()
    return "genome" in d and "envelope" not in d


def _clean_nt(nt: str) -> str:
    return re.sub(r"[^ACGTN]", "", nt.upper())


def _translate(nt: str) -> str:
    cds = nt[: len(nt) - len(nt) % 3]
    return str(Seq(cds).translate())


def score_env_aa(aa: str) -> tuple[float, str]:
    """Return (score, trunc_aa). score < 0 => reject."""
    trunc = aa.split("*")[0]
    if not (MIN_ENV_AA <= len(trunc) <= MAX_ENV_AA):
        return -1.0, trunc
    early_stops = aa[: min(len(aa), MAX_ENV_AA)].count("*")
    if early_stops > 2:
        return -1.0, trunc
    if _ENV_NTERM_OK.match(trunc):
        return float(len(trunc)) - 2.0 * early_stops, trunc
    if trunc.startswith(_MATURE_GP120) or trunc.startswith("VEKLWVTVYY"):
        # Mature gp120 without signal — usable but note weaker for nt_to_aa ATG.
        return float(len(trunc)) * 0.5 - 2.0 * early_stops, trunc
    return -1.0, trunc


def extract_env_cds(nt: str) -> tuple[str, str, str] | None:
    """
    Find best in-frame Env CDS in a nucleotide string.
    Returns (env_nt, env_aa, frame_note) or None if no confident hit.
    """
    s = _clean_nt(nt)
    if len(s) < MIN_ENV_NT:
        return None

    candidates: list[tuple[float, int, str, str, str]] = []

    # Direct frame-0 / ±1 on full string (deposited CDS).
    for fr in range(3):
        sub = s[fr:]
        aa = _translate(sub)
        sc, trunc = score_env_aa(aa)
        if sc > 0:
            # Prefer ATG-start slice for downstream nt_to_aa.
            atg_rel = sub.find("ATG")
            if atg_rel >= 0 and atg_rel < 60:
                env_nt = sub[atg_rel:]
                aa2 = _translate(env_nt)
                sc2, trunc2 = score_env_aa(aa2)
                if sc2 > 0:
                    candidates.append((sc2 + 50.0, fr, env_nt, trunc2, f"frame{fr}+ATG@{atg_rel}"))
                    continue
            candidates.append((sc, fr, sub, trunc, f"frame{fr}"))

    # Scan ATGs (genomes / long contigs).
    if len(s) >= 8000 or not candidates:
        for m in re.finditer("ATG", s):
            i = m.start()
            if i + MIN_ENV_NT > len(s):
                break
            env_nt = s[i : i + MAX_ENV_NT + 60]
            aa = _translate(env_nt)
            sc, trunc = score_env_aa(aa)
            if sc > 0:
                # Trim to AA ORF length * 3
                env_nt = env_nt[: len(trunc) * 3]
                candidates.append((sc, i, env_nt, trunc, f"ATG@{i}"))

    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    _, _, env_nt, trunc, note = candidates[0]
    # Require signal-like start for pipeline (nt_to_aa needs ATG).
    if not env_nt.startswith("ATG") or not _ENV_NTERM_OK.match(trunc):
        # Retry: only ATG+signal candidates
        for sc, _, ent, tr, ntnote in candidates:
            if ent.startswith("ATG") and _ENV_NTERM_OK.match(tr):
                return ent[: len(tr) * 3], tr, ntnote
        return None
    if not (MIN_ENV_NT <= len(env_nt) <= MAX_ENV_NT + 30):
        # Clip/pad length window: keep if AA OK even if NT slightly long
        if not (MIN_ENV_AA <= len(trunc) <= MAX_ENV_AA):
            return None
    return env_nt[: len(trunc) * 3], trunc, note


def iter_fasta(path) -> Iterator[tuple[str, str]]:
    header = None
    parts: list[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header = line[1:].strip()
                parts = []
            else:
                parts.append(line.strip())
        if header is not None:
            yield header, "".join(parts)


def load_eligible_env_records(
    fasta_paths: list,
    *,
    min_year: int = 1985,
    max_year: int = 2026,
    include_genomes: bool = True,
) -> tuple[list[HivRecord], dict]:
    """
    Load and QC Env CDS records from regional FASTAs.
    Returns (records, stats).
    """
    stats = {
        "n_raw": 0,
        "bad_header": 0,
        "empty_date": 0,
        "range_or_bad_date": 0,
        "year_oob": 0,
        "not_env_or_genome": 0,
        "extract_fail": 0,
        "dup": 0,
        "kept": 0,
        "kept_from_genome": 0,
        "kept_from_env_cds": 0,
    }
    seen: set[str] = set()
    out: list[HivRecord] = []

    for fp in fasta_paths:
        for hdr, seq in iter_fasta(fp):
            stats["n_raw"] += 1
            try:
                acc, desc, loc, country, date_raw = parse_header(hdr)
            except ValueError:
                stats["bad_header"] += 1
                continue
            if acc in seen:
                stats["dup"] += 1
                continue
            try:
                date_str, sort_key = parse_date(date_raw)
            except ValueError as e:
                msg = str(e)
                if "empty" in msg:
                    stats["empty_date"] += 1
                else:
                    stats["range_or_bad_date"] += 1
                continue
            y = sort_key[0]
            if y < min_year or y > max_year:
                stats["year_oob"] += 1
                continue

            is_env = is_env_description(desc)
            is_genome = is_genome_description(desc)
            nt_raw = _clean_nt(seq)
            extracted = None
            src = "env_cds"

            if is_env and MIN_ENV_NT <= len(nt_raw) <= MAX_ENV_NT:
                extracted = extract_env_cds(nt_raw)
                src = "env_cds"
            elif include_genomes and is_genome and len(nt_raw) >= 8000:
                extracted = extract_env_cds(nt_raw)
                src = "genome"
            else:
                stats["not_env_or_genome"] += 1
                continue

            if extracted is None:
                stats["extract_fail"] += 1
                continue
            env_nt, env_aa, note = extracted
            unit, has_state = geo_unit(loc, country)
            seen.add(acc)
            out.append(
                HivRecord(
                    acc=acc,
                    desc=desc,
                    location_raw=loc,
                    country=country,
                    date_raw=date_raw,
                    date_str=date_str,
                    sort_key=sort_key,
                    unit=unit,
                    has_state=has_state,
                    nt=env_nt,
                    aa=env_aa,
                    frame_note=f"{src}:{note}",
                    source_file=str(fp),
                )
            )
            stats["kept"] += 1
            if src == "genome":
                stats["kept_from_genome"] += 1
            else:
                stats["kept_from_env_cds"] += 1

    return out, stats


def greedy_geo_split(unit_counts: dict[str, int]) -> dict[str, str]:
    """Whole units → train/val/test targeting 80/10/10 by sequence count."""
    total = sum(unit_counts.values())
    target = {"train": 0.80 * total, "val": 0.10 * total, "test": 0.10 * total}
    fill = {"train": 0.0, "val": 0.0, "test": 0.0}
    assign: dict[str, str] = {}
    for unit, c in sorted(unit_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        split = max(("train", "val", "test"), key=lambda s: target[s] - fill[s])
        assign[unit] = split
        fill[split] += c
    return assign


def write_date_contiguous_groups(
    records: list[tuple[str, str, tuple[int, int, int], str]],
    out_dir,
    prefix: str,
    start_group: int,
    group_size: int,
    min_group: int,
    max_span_years: int,
    meta_extra: dict | None = None,
) -> tuple[int, list[dict]]:
    """
    records: (acc, date_str, sort_key, nt), already filtered to one geo unit.
    Returns (next_group_index, group_summaries).
    """
    import csv
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = start_group
    chunk: list = []
    summaries: list[dict] = []

    def emit(ch):
        nonlocal g
        if len(ch) < min_group:
            return
        fasta = out_dir / f"{prefix}_group_{g:03d}.fasta"
        csv_path = out_dir / f"{prefix}_group_{g:03d}.csv"
        with open(fasta, "w") as ff, open(csv_path, "w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["name", "date"])
            for acc, date_str, _, seq in ch:
                ff.write(f">{acc},{date_str}\n{seq}\n")
                w.writerow([acc, date_str])
        years = [r[2][0] for r in ch]
        info = {
            "group": g,
            "prefix": prefix,
            "n_leaves": len(ch),
            "year_min": min(years),
            "year_max": max(years),
        }
        if meta_extra:
            info.update(meta_extra)
        summaries.append(info)
        g += 1

    for rec in records:
        year = rec[2][0]
        if chunk and (
            len(chunk) >= group_size
            or (year - chunk[0][2][0]) > max_span_years
        ):
            emit(chunk)
            chunk = []
        chunk.append(rec)
    emit(chunk)
    return g, summaries
