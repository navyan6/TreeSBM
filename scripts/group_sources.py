"""Map global group IDs to source dataset and protein."""

# (source, protein, first_global, last_global)  — inclusive, 1-indexed
SOURCE_RANGES = [
    ("master_h3n2",             "H3N2_HA",   1,  48),
    ("h3n2_swine_all",          "H3N2_HA",  49,  55),
    ("avian_h1n1_2010_2020_HA", "H1N1_HA",  56,  56),
    ("h1n1_human_ha_2010_2017", "H1N1_HA",  57, 106),
    ("human_h1n1_NA_2005_2015", "NA",      107, 156),
    ("human_h1n1_2015_2018",    "H1N1_HA", 157, 206),
    ("fluB_yamagata_alltime",   "FLUB_HA", 207, 238),
    ("fluB_victoria_all",       "FLUB_HA", 239, 284),
]


def group_to_source(g: int) -> str | None:
    """Return the source dataset name for a global group number, or None."""
    for source, _protein, lo, hi in SOURCE_RANGES:
        if lo <= g <= hi:
            return source
    return None


def group_to_protein(g: int) -> str | None:
    """Return the protein label (H3N2_HA / H1N1_HA / NA / FLUB_HA) for a group."""
    for _source, protein, lo, hi in SOURCE_RANGES:
        if lo <= g <= hi:
            return protein
    return None


def groups_for_source(name: str) -> list[int]:
    """All global group numbers belonging to a given source dataset."""
    out: list[int] = []
    for source, _protein, lo, hi in SOURCE_RANGES:
        if source == name:
            out.extend(range(lo, hi + 1))
    return out


def h3n2_ha_groups() -> list[int]:
    """Human H3N2 HA groups (master_h3n2) — the v1 EVEscape target set."""
    return groups_for_source("master_h3n2")


def parse_group_spec(spec: str) -> list[int]:
    """
    Parse a CLI group spec into a sorted list of ints.

    Accepts comma-separated ranges/values, e.g. "1-48", "1-48,157-206", "3,7,9".
    """
    groups: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            groups.update(range(int(lo), int(hi) + 1))
        else:
            groups.add(int(part))
    return sorted(groups)


if __name__ == "__main__":
    # Sanity: ranges are contiguous, non-overlapping, and cover 1..284.
    covered: list[int] = []
    for source, protein, lo, hi in SOURCE_RANGES:
        covered.extend(range(lo, hi + 1))
        print(f"{source:28s} {protein:8s} groups {lo:3d}-{hi:3d} ({hi - lo + 1})")
    assert covered == sorted(covered), "ranges overlap or are out of order"
    assert covered == list(range(1, covered[-1] + 1)), "ranges have gaps"
    print(f"\nTotal groups: {len(covered)} (1..{covered[-1]})")
    print(f"H3N2 HA (v1 EVEscape) groups: {h3n2_ha_groups()[0]}..{h3n2_ha_groups()[-1]}")
