"""SARS-CoV-2 Spike domain bands in Wuhan-Hu-1 / UniProt P0DTC2 coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch

# Inclusive 1-based Wuhan / P0DTC2 residue ranges.
SPIKE_DOMAINS_1BASED: dict[str, tuple[int, int]] = {
    "NTD": (13, 305),
    "RBD": (319, 541),
    "RBM": (438, 506),
    "S1": (13, 685),
    "S2": (686, 1273),
    "furin": (680, 685),  # P681 cleavage motif neighborhood (S1/S2 boundary)
}

# Primary mut-frac bands (emit by default in enrichment JSON).
PRIMARY_DOMAIN_KEYS = ("NTD", "RBD", "RBM", "S1", "S2", "furin")

# Broader region KPIs (site_recall / any_mut) — antigenic + cleavage focus.
REGION_RECALL_KEYS = ("NTD", "RBD", "furin")

# Landmark residues for viz / coord sanity (1-based spike).
SPIKE_LANDMARKS_1BASED: dict[str, list[dict]] = {
    "NTD": [
        {"name": "L212", "pos": 212, "note": "PMC primary NTD"},
        {"name": "A222", "pos": 222, "note": "PMC primary NTD"},
    ],
    "RBD": [
        {"name": "K417", "pos": 417, "note": "ACE2 / escape"},
        {"name": "L452", "pos": 452, "note": "Delta escape"},
        {"name": "E484", "pos": 484, "note": "escape"},
        {"name": "N501", "pos": 501, "note": "ACE2 affinity"},
    ],
    "RBM": [
        {"name": "S477", "pos": 477, "note": "flexible loop start"},
        {"name": "G485", "pos": 485, "note": "flexible loop end"},
    ],
    "furin": [
        {"name": "P681", "pos": 681, "note": "PRRAR↓S cleavage motif"},
        {"name": "S1_S2_boundary", "pos": 685, "note": "S1 ends; S2 starts 686"},
    ],
    "S1": [
        {"name": "D614", "pos": 614, "note": "D614G fitness landmark"},
        {"name": "T547", "pos": 547, "note": "PMC S1"},
    ],
    "S2": [
        {"name": "N856", "pos": 856, "note": "PMC S2"},
        {"name": "L981", "pos": 981, "note": "PMC S2"},
    ],
}

CANONICAL_SPIKE_LEN = 1273


def domain_cols_0based(lo_1based: int, hi_1based: int, L: int) -> range:
    """Convert inclusive 1-based [lo, hi] to 0-based columns clipped to [0, L)."""
    lo = max(0, lo_1based - 1)
    hi = min(L - 1, hi_1based - 1)
    if hi < lo:
        return range(0, 0)
    return range(lo, hi + 1)


def domain_mask(L: int, lo_1based: int, hi_1based: int) -> torch.Tensor:
    """Bool [L] mask for one domain band."""
    m = torch.zeros(L, dtype=torch.bool)
    for c in domain_cols_0based(lo_1based, hi_1based, L):
        m[c] = True
    return m


def all_domain_masks(
    L: int,
    domains: dict[str, tuple[int, int]] | None = None,
) -> dict[str, torch.Tensor]:
    domains = domains or SPIKE_DOMAINS_1BASED
    return {name: domain_mask(L, lo, hi) for name, (lo, hi) in domains.items()}


def domain_mut_frac(
    mut_cols: Iterable[int],
    mask: torch.Tensor,
) -> float:
    """
    Fraction of mutation columns that fall in ``mask``.
    NaN if there are no mutations.
    """
    cols = list(mut_cols)
    if not cols:
        return float("nan")
    hot = mask.bool()
    n_hot = sum(1 for c in cols if 0 <= c < hot.numel() and bool(hot[c].item()))
    return n_hot / len(cols)


def region_site_recall(
    root: str,
    gt: str,
    gen: str,
    mask: torch.Tensor,
) -> float:
    """
    Among GT mutations inside ``mask``, fraction where gen also mutated.

    Same factorization as global site_recall but restricted to region columns:
      region_site_recall = P(gen!=root | root!=GT, col in R)
    NaN if GT has no mutations in the region.
    """
    hot = mask.bool()
    L = min(len(root), len(gt), len(gen), hot.numel())
    mut_total = site_hits = 0
    for i in range(L):
        if not bool(hot[i].item()):
            continue
        if root[i] == gt[i]:
            continue
        mut_total += 1
        if gen[i] != root[i]:
            site_hits += 1
    if mut_total == 0:
        return float("nan")
    return site_hits / mut_total


def region_any_mut(
    root: str,
    leaf: str,
    mask: torch.Tensor,
) -> float:
    """
    Binary: 1.0 if leaf has ≥1 mutation (≠root) inside ``mask``, else 0.0.
    """
    hot = mask.bool()
    L = min(len(root), len(leaf), hot.numel())
    for i in range(L):
        if bool(hot[i].item()) and leaf[i] != root[i]:
            return 1.0
    return 0.0


def spike_region_annotations(L: int = CANONICAL_SPIKE_LEN) -> dict:
    """Serializable COVID region annotation artifact for viz / eval metadata."""
    regions = {}
    for name, (lo, hi) in SPIKE_DOMAINS_1BASED.items():
        cols = list(domain_cols_0based(lo, hi, L))
        regions[name] = {
            "lo_1based": lo,
            "hi_1based": hi,
            "cols_0based": [cols[0], cols[-1]] if cols else [],
            "n_cols": len(cols),
            "landmarks_1based": SPIKE_LANDMARKS_1BASED.get(name, []),
        }
    return {
        "pathogen": "covid_spike",
        "reference": "Wuhan-Hu-1 / UniProt P0DTC2",
        "indexing": {
            "residue": "1-based spike",
            "column": "0-based; col = spike_pos - 1 on ungapped full-length L=1273",
            "max_seq_len": L,
            "canonical_len": CANONICAL_SPIKE_LEN,
        },
        "primary_domain_keys": list(PRIMARY_DOMAIN_KEYS),
        "region_recall_keys": list(REGION_RECALL_KEYS),
        "regions": regions,
        "notes": (
            "Furin band is the PRRAR↓S neighborhood (680–685), not the entire "
            "S1/S2 ectodomain. Absolute columns are homologous only for "
            "ungapped Wuhan-frame full-length sequences."
        ),
    }


def write_spike_region_annotations(
    path: str | Path,
    L: int = CANONICAL_SPIKE_LEN,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spike_region_annotations(L), indent=2) + "\n")
    return path
