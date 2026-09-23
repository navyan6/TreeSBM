"""Literature antigenic HA site lists (Li et al., Nat Microbiol 2016).

Flu HA only. Converts mature H3/H1 numbering to full-ORF column indices
(including signal peptide).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Full-length ORF signal peptide length for human H3 HA (standard).
H3_SIGNAL_LEN = 16
# Common pdm09 H1 signal length (validate before applying H1 masks).
H1_SIGNAL_LEN_DEFAULT = 17

# ---------------------------------------------------------------------------
# H3N2 (H3 numbering) — primary train / eval lit mask
# ---------------------------------------------------------------------------
# Body-text cluster transitions + contemporary escape (Cambridge author MS).
# Fig. 5 also colors Koel-cited H3 positions 145/155/156/158/189/193.
NMICROBIOL201658_H3_PRIMARY = [
    ("H155T", 155, "HA1/A", "SY97→FU02 cluster transition (with Q156H); body"),
    ("Q156H", 156, "HA1/A", "SY97→FU02 cluster transition (with H155T); body"),
    ("F159S", 159, "HA1/B", "3C.3a vs TX/50 (159F); most freq. escape pos; body"),
    ("F159Y", 159, "HA1/B", "3C.2a vs TX/50; epidemic 2014–15; body"),
    ("H183L", 183, "HA1", "CUHK5250 Q156H+H183L near FU02 edge; body"),
    ("K189E", 189, "HA1/B", "TX/50 escape toward HK/4801 (#21,#27); body+Fig6"),
    ("N225D", 225, "RBS", "TX/50 escape / clade 3C.2a RBS; body+Fig6"),
    ("N145K", 145, "HA1/A", "Fig. 5 coloring (Koel-cited antigenic site)"),
    ("N158D", 158, "HA1/B", "Fig. 5 coloring + TX/50 #10"),
    ("F193S", 193, "HA1/B", "Fig. 5 coloring + TX/50 escape combos"),
]

# Fig. 6 TX/50 escape mutants (unique positions from figure legend genotypes).
NMICROBIOL201658_H3_SECONDARY = [
    ("Q75L", 75, "HA1", "TX/50 #6"),
    ("V88I", 88, "HA1", "TX/50 #15"),
    ("Y94F", 94, "HA1", "TX/50 #8/#14"),
    ("S107T", 107, "HA1", "TX/50 #14"),
    ("N122S", 122, "HA1", "TX/50 #23"),
    ("W127L", 127, "HA1", "TX/50 #13"),
    ("N128D", 128, "HA1", "TX/50 #26"),
    ("N144I", 144, "HA1", "TX/50 #6/#19"),
    ("L157F", 157, "HA1", "TX/50 #18"),
    ("K160E", 160, "HA1", "TX/50 #16"),
    ("E172G", 172, "HA1", "TX/50 #2/#24"),
    ("F174Y", 174, "HA1", "TX/50 #7"),
    ("I192T", 192, "HA1", "TX/50 #6/#16"),
    ("Q197R", 197, "HA1", "TX/50 #11"),
    ("T203A", 203, "HA1", "TX/50 #3"),
    ("K207N", 207, "HA1", "TX/50 #12"),
    ("I217T", 217, "HA1", "TX/50 #5"),
    ("S219F", 219, "HA1", "TX/50 #4"),
    ("R220G", 220, "HA1", "TX/50 #9"),
    ("I242T", 242, "HA1", "TX/50 #16/#22"),
    ("N246H", 246, "HA1", "TX/50 #28/#29 (with F159S)"),
]

# Globular-head library span used for TX/50 / CUHK / Kwangju screens (H3 #).
H3_GLOBULAR_HEAD_RANGE = (63, 252)  # inclusive 1-based H3 numbering

# HA region bands for eval (H3 mature numbering, inclusive).
# Metric prefix ha_head_* maps to the globular-head library span.
HA_DOMAINS_H3_1BASED: dict[str, tuple[int, int]] = {
    "HA_head": H3_GLOBULAR_HEAD_RANGE,
}
# RBS landmark site-set (not a contiguous band) — Y98/W153/H183/Y195/N225.
HA_RBS_SITES_H3_1BASED: tuple[int, ...] = (98, 153, 183, 195, 225)
PRIMARY_HA_DOMAIN_KEYS = ("HA_head", "HA_RBS")
REGION_RECALL_HA_KEYS = ("HA_head", "HA_RBS")

# Landmark residues (H3 #) for viz / coord sanity on full-ORF L=566.
HA_LANDMARKS_H3_1BASED: dict[str, list[dict]] = {
    "HA_head": [
        {"name": "Y98", "h3_pos": 98, "note": "RBS landmark"},
        {"name": "W153", "h3_pos": 153, "note": "RBS landmark"},
        {"name": "H183", "h3_pos": 183, "note": "RBS / lit primary"},
        {"name": "Y195", "h3_pos": 195, "note": "RBS landmark"},
        {"name": "N145", "h3_pos": 145, "note": "Koel antigenic site A"},
        {"name": "F159", "h3_pos": 159, "note": "most frequent TX/50 escape"},
        {"name": "K189", "h3_pos": 189, "note": "site B escape"},
        {"name": "N225", "h3_pos": 225, "note": "RBS escape"},
    ],
}
CANONICAL_HA_ORF_LEN = 566

# ---------------------------------------------------------------------------
# H1N1 pdm09 (H1 numbering) — FLU ONLY; never reuse H3 signal=16 offsets
# ---------------------------------------------------------------------------
# Two numbering schemes appear in the H1 literature:
#   mature_h1 — Caton / nmicrobiol Sa 153–156 (residue 1 = first AA after signal)
#   orf_met   — from first Met incl. signal (Matsuzaki 2014; Frontiers 2017 epitopes;
#               Mountain West Genes 2022 K147N/P154S/S200P)
# TreeSBM columns are full-ORF ungapped HA. Convert with:
#   col = mature_h1 + H1_SIGNAL_LEN - 1
#   col = orf_met - 1
# Empirically on data/h1n1 (geo) train: signal_len=17, mature start DTLCIGYHANNSTDT,
# L=566. Do NOT apply H3 SIGNAL_LEN=16.

# Body text; upload stub supp captions also mention positions 153–156.
# Positions below are **mature_h1**.
NMICROBIOL201658_H1_PRIMARY = [
    ("K153E", 153, "Sa", "4–16× HI drop; field antigenic variants"),
    ("K154E", 154, "Sa", "critical Sa tetramer with 153–156"),
    ("G155E", 155, "Sa", "field antigenic change"),
    ("N156D", 156, "Sa", "high-frequency escape (e.g. N156G)"),
    ("D127E", 127, "Sa-adj", "preliminary + library escape; combined with 153–156"),
]

NMICROBIOL201658_H1_SECONDARY = [
    ("N125D", 125, "Sa", "2–4× HI; Sa site"),
    ("N129D", 129, "Sa", "2–4× HI"),
    ("K141*", 141, "near Sa", "selected from seasonal H1 comparisons; no escape in screen"),
    ("K142*", 142, "Koel", "Koel major-change site (H1 numbering of H3)"),
    ("K152*", 152, "Koel", "Koel major-change site"),
    ("K163E", 163, "Sa", "clinical isolate escape; library screen"),
    ("S183P", 183, "Sb-adj", "2–4× HI adjacent to Sb"),
    ("S186*", 186, "Koel/RBS", "Koel major-change site"),
    ("D187*", 187, "near 186/190", "selected; no escape in screen"),
    ("D190*", 190, "Koel/RBS", "Koel major-change site"),
]

# Field / epitope guidance mutations requested for H1 antigenic retrain.
# scheme: "mature_h1" | "orf_met"
H1_FIELD_GUIDANCE = [
    ("K147N", 147, "orf_met", "Pa/Ca2-adj; Matsuzaki + Mountain West 2015–19"),
    ("P154S", 154, "orf_met", "Ca2; Mountain West; MSA P/S poly"),
    ("K166Q", 163, "mature_h1", "Sa; Sci Rep H3#166≡H1#163; clade 6B marker; MSA K/Q"),
    ("S185I", 185, "mature_h1", "Sb; MSA I/T/S at col 201"),
    ("S200P", 200, "orf_met", "Ca2; Mountain West; MSA S/P poly"),
]

# pdm09 antigenic-site cores in **orf_met** numbering (Frontiers Immunol 2017
# Caton antigenic sites remapped to A/California/07/2009-like frame). Validated
# against data/h1n1 geo consensus (Cb=LSTARS, Ca2a=CPHAGA, Sa_b=VKKGN, …).
H1_ANTIGENIC_SITES_ORF_MET: dict[str, list[int]] = {
    "Cb": list(range(87, 93)),
    "Ca2": list(range(153, 159)) + [237, 238],
    "Sa": [141, 142] + list(range(169, 174)) + list(range(175, 181)),
    "Sb": list(range(200, 212)),
    "Ca1": list(range(182, 187)) + list(range(219, 222)) + list(range(251, 254)),
}

# RBS landmarks in **orf_met** (validated on geo H1 MSA consensus).
H1_RBS_ORF_MET = [
    (115, "Y", "Y98-like (mature≈98); INYEE"),
    (167, "W", "RBS Trp (LIWLV)"),
    (197, "H", "near RBS (GIHHP)"),
    (209, "Y", "Sb/RBS-adjacent Y (SLYQN)"),
]

H1_SA_SITES = [124, 125] + list(range(153, 158)) + list(range(159, 165))
H1_GLOBULAR_HEAD_RANGE = (54, 253)  # inclusive **mature_h1** numbering
H1_ALL_AA_LIBRARY_POSITIONS = [
    125, 127, 129, 141, 142, 152, 153, 154, 155, 156, 163, 183, 186, 187, 190,
]
CANONICAL_H1_ORF_LEN = 566
H1_SIGNAL_MOTIF_PREFIX = "MKAILVV"  # pdm09-like; allow LL/ML / AT/TT variants
H1_MATURE_MOTIF = "DTLCIGYHANNSTDT"


def h3_to_col(h3_pos: int, signal_len: int = H3_SIGNAL_LEN) -> int:
    """0-based full-ORF column for an H3-numbered mature position."""
    return h3_pos + signal_len - 1


def h3_to_orf1(h3_pos: int, signal_len: int = H3_SIGNAL_LEN) -> int:
    """1-based full-ORF index for an H3-numbered mature position."""
    return h3_pos + signal_len


def h1_mature_to_col(mature_pos: int, signal_len: int = H1_SIGNAL_LEN_DEFAULT) -> int:
    """0-based full-ORF column for a mature-H1 position."""
    return mature_pos + signal_len - 1


def h1_orf_to_col(orf_met_pos: int) -> int:
    """0-based full-ORF column for 1-based from-Met numbering."""
    return orf_met_pos - 1


def unique_h3_primary_positions() -> list[int]:
    return sorted({pos for _, pos, _, _ in NMICROBIOL201658_H3_PRIMARY})


def unique_h3_secondary_positions() -> list[int]:
    primary = set(unique_h3_primary_positions())
    return sorted(
        {pos for _, pos, _, _ in NMICROBIOL201658_H3_SECONDARY} - primary
    )


def unique_h1_primary_positions() -> list[int]:
    """nmicrobiol H1 primary positions in **mature_h1** numbering."""
    return sorted({pos for _, pos, _, _ in NMICROBIOL201658_H1_PRIMARY})


def h1_guidance_cols(signal_len: int = H1_SIGNAL_LEN_DEFAULT) -> list[tuple[str, int, str]]:
    """
    Return (label, col_0based, source_tag) for the H1 train lit mask.

    Union of: nmicrobiol Sa 153–156 (+D127E), Sa/Sb/Ca1/Ca2/Cb cores,
    RBS landmarks, and field guidance mutations (K147N, P154S, K166Q, S185I, S200P).
    Does **not** include the full globular-head continuum (see head mask).
    """
    out: dict[int, tuple[str, str]] = {}

    def _add(label: str, col: int, tag: str) -> None:
        if col < 0:
            return
        if col not in out:
            out[col] = (label, tag)

    for mut, pos, site, _note in NMICROBIOL201658_H1_PRIMARY:
        _add(mut, h1_mature_to_col(pos, signal_len), f"nmicrobiol_mature:{site}")

    for site, positions in H1_ANTIGENIC_SITES_ORF_MET.items():
        for pos in positions:
            _add(f"{site}_{pos}", h1_orf_to_col(pos), f"antigenic_orf_met:{site}")

    for orf_pos, _aa, note in H1_RBS_ORF_MET:
        _add(f"RBS_{orf_pos}", h1_orf_to_col(orf_pos), f"rbs_orf_met:{note}")

    for mut, pos, scheme, _note in H1_FIELD_GUIDANCE:
        if scheme == "mature_h1":
            col = h1_mature_to_col(pos, signal_len)
        elif scheme == "orf_met":
            col = h1_orf_to_col(pos)
        else:
            raise ValueError(f"Unknown H1 numbering scheme: {scheme}")
        _add(mut, col, f"field:{scheme}")

    return [(label, col, tag) for col, (label, tag) in sorted(out.items(), key=lambda x: x[0])]


def unique_h1_lit_cols(signal_len: int = H1_SIGNAL_LEN_DEFAULT) -> list[int]:
    return [col for _, col, _ in h1_guidance_cols(signal_len)]


def ha_domain_cols_0based(
    lo_h3: int,
    hi_h3: int,
    L: int,
    signal_len: int = H3_SIGNAL_LEN,
) -> range:
    """Inclusive H3 [lo, hi] → 0-based full-ORF columns clipped to [0, L)."""
    lo = max(0, h3_to_col(lo_h3, signal_len))
    hi = min(L - 1, h3_to_col(hi_h3, signal_len))
    if hi < lo:
        return range(0, 0)
    return range(lo, hi + 1)


def ha_domain_mask(
    L: int,
    lo_h3: int,
    hi_h3: int,
    signal_len: int = H3_SIGNAL_LEN,
) -> "torch.Tensor":
    """Bool [L] mask for one HA domain band (full-ORF columns)."""
    import torch

    m = torch.zeros(L, dtype=torch.bool)
    for c in ha_domain_cols_0based(lo_h3, hi_h3, L, signal_len):
        m[c] = True
    return m


def ha_rbs_mask(
    L: int,
    signal_len: int = H3_SIGNAL_LEN,
    sites_h3: tuple[int, ...] | None = None,
) -> "torch.Tensor":
    """Bool [L] mask for RBS landmark columns (full-ORF)."""
    import torch

    m = torch.zeros(L, dtype=torch.bool)
    for pos in (sites_h3 or HA_RBS_SITES_H3_1BASED):
        c = h3_to_col(pos, signal_len)
        if 0 <= c < L:
            m[c] = True
    return m


def all_ha_domain_masks(
    L: int,
    signal_len: int = H3_SIGNAL_LEN,
    domains: dict[str, tuple[int, int]] | None = None,
) -> dict[str, "torch.Tensor"]:
    domains = domains or HA_DOMAINS_H3_1BASED
    out = {
        name: ha_domain_mask(L, lo, hi, signal_len)
        for name, (lo, hi) in domains.items()
    }
    # Discrete RBS landmark site-set (wired alongside globular-head band).
    out["HA_RBS"] = ha_rbs_mask(L, signal_len=signal_len)
    return out


def ha_region_annotations(
    L: int = CANONICAL_HA_ORF_LEN,
    signal_len: int = H3_SIGNAL_LEN,
) -> dict:
    """Serializable flu HA region annotation artifact (H3 numbering)."""
    regions = {}
    for name, (lo, hi) in HA_DOMAINS_H3_1BASED.items():
        cols = list(ha_domain_cols_0based(lo, hi, L, signal_len))
        landmarks = []
        for lm in HA_LANDMARKS_H3_1BASED.get(name, []):
            landmarks.append({
                **lm,
                "orf_1based": h3_to_orf1(lm["h3_pos"], signal_len),
                "col_0based": h3_to_col(lm["h3_pos"], signal_len),
            })
        regions[name] = {
            "lo_h3_1based": lo,
            "hi_h3_1based": hi,
            "cols_0based": [cols[0], cols[-1]] if cols else [],
            "n_cols": len(cols),
            "metric_prefix": "ha_head",
            "landmarks_h3_1based": landmarks,
        }
    rbs_cols = [h3_to_col(p, signal_len) for p in HA_RBS_SITES_H3_1BASED
                if 0 <= h3_to_col(p, signal_len) < L]
    regions["HA_RBS"] = {
        "sites_h3_1based": list(HA_RBS_SITES_H3_1BASED),
        "cols_0based": rbs_cols,
        "n_cols": len(rbs_cols),
        "metric_prefix": "ha_rbs",
        "note": "RBS landmark site-set (Y98/W153/H183/Y195/N225), not a band",
    }
    return {
        "pathogen": "flu_h3n2_ha",
        "reference": "H3 mature HA1 numbering + 16-aa signal → full-ORF",
        "indexing": {
            "residue": "1-based H3 mature HA1",
            "column": f"0-based full-ORF; col = h3_pos + {signal_len} - 1",
            "signal_len": signal_len,
            "max_seq_len": L,
            "canonical_len": CANONICAL_HA_ORF_LEN,
        },
        "primary_domain_keys": list(PRIMARY_HA_DOMAIN_KEYS),
        "region_recall_keys": list(REGION_RECALL_HA_KEYS),
        "regions": regions,
        "notes": (
            "HA_head = globular-head library span H3 #63–252 "
            "(mut_hotspot_mask_h3_globular_head.pt). HA_RBS = landmark "
            "site-set Y98/W153/H183/Y195/N225. Distinct from "
            "COVID spike/RBD bands. Do not reuse H3 offsets for H1."
        ),
    }


def write_ha_region_annotations(
    path: str | Path,
    L: int = CANONICAL_HA_ORF_LEN,
    signal_len: int = H3_SIGNAL_LEN,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ha_region_annotations(L, signal_len), indent=2) + "\n"
    )
    return path
