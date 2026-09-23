#!/usr/bin/env python3
"""
Rank held-out COVID groups by closeness of generated vs observed trees.

Loads Newick + FASTA pairs, size-matches the observed tree to the generated
leaf count (random tip subsample + induced subtree), then scores with
sequence-matched RF / terminal edit / optional quartet + coverage@eps.

Usage:
  python scripts/rank_covid_closest_groups.py \
      --data data/covid/test \
      --gen-dir results/covid_tree_viz_screen \
      --out-dir results/covid_tree_viz \
      --promote-top 3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Bio import SeqIO

from benchmarks.heldout.build_examples import (
    _parse_newick,
    induced_subtree,
    load_tree,
)
from benchmarks.metrics import sequences as S
from benchmarks.metrics.matched import (
    quartet_distance,
    sequence_matched_rf,
    terminal_edit_distance,
)
from benchmarks.metrics import trees as T
from src.tree_state import TreeState


def _read_fasta_ids(path: Path) -> dict[str, str]:
    """Parse FASTA; strip optional '|tag' suffixes from headers."""
    out: dict[str, str] = {}
    for rec in SeqIO.parse(str(path), "fasta"):
        nid = rec.id.split("|", 1)[0]
        out[nid] = str(rec.seq)
    return out


def load_tree_pair(nwk: Path, fasta: Path) -> TreeState:
    root_id, edges, bls = _parse_newick(str(nwk))
    seqs = _read_fasta_ids(fasta)
    node_ids = sorted({root_id} | {n for e in edges for n in e})
    if seqs:
        ref_len = len(next(iter(seqs.values())))
        for n in node_ids:
            seqs.setdefault(n, "-" * ref_len)
    cm = T.children_map(
        TreeState(
            node_ids=node_ids,
            root_id=root_id,
            edges=edges,
            branch_lengths=bls,
            node_seqs={},
            active_leaves=[],
        )
    )
    leaves = [n for n in node_ids if n not in cm]
    return TreeState(
        node_ids=node_ids,
        root_id=root_id,
        edges=edges,
        branch_lengths=bls,
        node_seqs=seqs,
        active_leaves=leaves,
    )


def size_match_obs(obs: TreeState, n_leaves: int, seed: int) -> TreeState:
    leaves = list(T.leaf_labels(obs))
    if len(leaves) == n_leaves:
        return obs
    if len(leaves) < n_leaves:
        raise ValueError(
            f"observed has {len(leaves)} leaves < gen {n_leaves}; cannot upsample"
        )
    keep = sorted(random.Random(seed).sample(leaves, n_leaves))
    return induced_subtree(obs, obs.root_id, keep)


def score_pair(gen: TreeState, obs: TreeState, seed: int = 42) -> dict:
    n_gen = len(T.leaf_labels(gen))
    obs_m = size_match_obs(obs, n_gen, seed=seed)
    n_obs = len(T.leaf_labels(obs_m))
    assert n_gen == n_obs, (n_gen, n_obs)

    rf = float(sequence_matched_rf(gen, obs_m))
    ted = terminal_edit_distance(gen, obs_m)
    term_edit = float(ted["mean"])
    try:
        qd = float(quartet_distance(gen, obs_m))
    except Exception:
        qd = float("nan")

    g_seqs = [gen.node_seqs[n] for n in T.leaf_labels(gen)]
    o_seqs = [obs_m.node_seqs[n] for n in T.leaf_labels(obs_m)]
    cov = float(S.coverage_at_k(o_seqs, g_seqs, eps_frac=0.02))
    # mean best-of-1 identity (obs leaf -> nearest gen leaf)
    ids = [S.best_of_k_identity(t, g_seqs) for t in o_seqs]
    mean_id = float(sum(ids) / len(ids)) if ids else float("nan")

    # Lower is better. Blend topology + sequence distances; reward coverage.
    combo = 0.45 * rf + 0.35 * term_edit + 0.20 * (1.0 - mean_id)
    if qd == qd:  # not NaN
        combo = 0.85 * combo + 0.15 * qd

    return {
        "n_gen_leaves": n_gen,
        "n_obs_full": len(T.leaf_labels(obs)),
        "rf": rf,
        "quartet": qd,
        "terminal_edit": term_edit,
        "coverage_eps2pct": cov,
        "mean_best_identity": mean_id,
        "combo_distance": float(combo),
    }


_GEN_RE = re.compile(
    r"^group_(\d{3})_generated(?:_([a-zA-Z0-9]+))?(?:_s(\d+))?\.nwk$"
)


def discover_gens(gen_dir: Path) -> list[tuple[int, str, Path, Path]]:
    """Return (group, tag, nwk, fasta) for each generated *.nwk with fasta."""
    found = []
    for nwk in sorted(gen_dir.glob("group_*_generated*.nwk")):
        m = _GEN_RE.match(nwk.name)
        if not m:
            continue
        gid = int(m.group(1))
        tag = m.group(2) or "default"
        fasta = nwk.with_suffix(".fasta")
        if not fasta.exists():
            # allow sidecars named without seed when nwk has seed
            alt = gen_dir / f"group_{gid:03d}_generated" + (
                f"_{tag}" if tag != "default" else ""
            ) + ".fasta"
            if alt.exists():
                fasta = alt
            else:
                print(f"WARN: skip {nwk.name} (no fasta)", file=sys.stderr)
                continue
        found.append((gid, tag, nwk, fasta))
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/covid/test")
    ap.add_argument("--gen-dir", required=True, help="Directory with generated nwk/fasta")
    ap.add_argument("--out-dir", default="results/covid_tree_viz")
    ap.add_argument("--obs-subsample-seed", type=int, default=42)
    ap.add_argument("--promote-top", type=int, default=0,
                    help="Copy top-K groups' best gen + observed into out-dir")
    ap.add_argument("--exclude-groups", type=int, nargs="*", default=[],
                    help="Groups to exclude from promotion (e.g. 40 already kept)")
    ap.add_argument("--tag-filter", default="",
                    help="Only score gens with this tag (e.g. screen / matched)")
    ap.add_argument("--write-md", default="",
                    help="Path for CLOSEST_GROUPS.md (default: out-dir/CLOSEST_GROUPS.md)")
    args = ap.parse_args()

    data = Path(args.data)
    gen_dir = Path(args.gen_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gens = discover_gens(gen_dir)
    if args.tag_filter:
        gens = [g for g in gens if g[1] == args.tag_filter]
    if not gens:
        raise SystemExit(f"No generated trees found in {gen_dir}")

    rows = []
    for gid, tag, nwk, fasta in gens:
        obs = load_tree(data, gid)
        gen = load_tree_pair(nwk, fasta)
        metrics = score_pair(gen, obs, seed=args.obs_subsample_seed)
        rows.append({
            "group": gid,
            "tag": tag,
            "gen_nwk": str(nwk),
            "gen_fasta": str(fasta),
            **metrics,
        })
        print(
            f"group={gid:03d} tag={tag:10s} leaves={metrics['n_gen_leaves']:3d} "
            f"rf={metrics['rf']:.4f} ted={metrics['terminal_edit']:.4f} "
            f"qd={metrics['quartet']:.4f} id={metrics['mean_best_identity']:.4f} "
            f"combo={metrics['combo_distance']:.4f}"
        )

    # Best gen per group (min combo), then rank groups
    best_by_group: dict[int, dict] = {}
    for r in rows:
        g = r["group"]
        if g not in best_by_group or r["combo_distance"] < best_by_group[g]["combo_distance"]:
            best_by_group[g] = r
    ranked = sorted(best_by_group.values(), key=lambda r: r["combo_distance"])

    csv_path = out_dir / "closest_groups_ranking.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["combo_distance"], r["group"])))
    print(f"Wrote {csv_path}")

    json_path = out_dir / "closest_groups_ranking.json"
    json_path.write_text(json.dumps({"all": rows, "best_per_group": ranked}, indent=2))
    print(f"Wrote {json_path}")

    md_path = Path(args.write_md) if args.write_md else out_dir / "CLOSEST_GROUPS.md"
    lines = [
        "# Closest COVID gen↔obs groups",
        "",
        "Ranked by combo distance (lower = closer):",
        "`0.45·RF + 0.35·terminal_edit + 0.20·(1−mean_best_identity)`",
        "(+ 15% quartet when available). Observed tip-subsampled to gen leaf count",
        f"(seed={args.obs_subsample_seed}) before matching.",
        "",
        "| Rank | Group | Leaves (gen/obs) | RF ↓ | Quartet ↓ | Term-edit ↓ | Mean best-id ↑ | Cov@2% ↑ | Combo ↓ | Tag |",
        "|-----:|------:|-----------------:|-----:|----------:|------------:|---------------:|---------:|--------:|-----|",
    ]
    for i, r in enumerate(ranked, 1):
        qd = r["quartet"]
        qd_s = f"{qd:.4f}" if qd == qd else "—"
        lines.append(
            f"| {i} | {r['group']} | {r['n_gen_leaves']}/{r['n_obs_full']} | "
            f"{r['rf']:.4f} | {qd_s} | {r['terminal_edit']:.4f} | "
            f"{r['mean_best_identity']:.4f} | {r['coverage_eps2pct']:.3f} | "
            f"{r['combo_distance']:.4f} | {r['tag']} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")

    exclude = set(args.exclude_groups)
    if args.promote_top > 0:
        promoted = [r for r in ranked if r["group"] not in exclude][: args.promote_top]
        for r in promoted:
            gid = r["group"]
            g3 = f"{gid:03d}"
            # observed full
            src_obs_nwk = data / f"group_{g3}_rooted.nwk"
            src_obs_fa = data / f"group_{g3}_anc_aa.fasta"
            dst_obs_nwk = out_dir / f"group_{g3}_observed.nwk"
            dst_obs_fa = out_dir / f"group_{g3}_observed_anc_aa.fasta"
            dst_obs_nwk.write_bytes(src_obs_nwk.read_bytes())
            if src_obs_fa.exists():
                dst_obs_fa.write_bytes(src_obs_fa.read_bytes())

            # size-matched observed for viz (same leaf count as best gen)
            obs = load_tree(data, gid)
            gen = load_tree_pair(Path(r["gen_nwk"]), Path(r["gen_fasta"]))
            n_gen = len(T.leaf_labels(gen))
            if len(T.leaf_labels(obs)) != n_gen:
                obs_m = size_match_obs(obs, n_gen, seed=args.obs_subsample_seed)
            else:
                obs_m = obs
            (out_dir / f"group_{g3}_observed_matched.nwk").write_text(
                T.to_newick(obs_m) + "\n"
            )

            src_nwk = Path(r["gen_nwk"])
            src_fa = Path(r["gen_fasta"])
            dst_nwk = out_dir / f"group_{g3}_generated_matched.nwk"
            dst_fa = out_dir / f"group_{g3}_generated_matched.fasta"
            dst_nwk.write_bytes(src_nwk.read_bytes())
            dst_fa.write_bytes(src_fa.read_bytes())
            # also keep seed-tagged copy if present
            seed_m = re.search(r"_s(\d+)\.nwk$", src_nwk.name)
            if seed_m:
                dst_nwk.with_name(
                    f"group_{g3}_generated_matched_s{seed_m.group(1)}.nwk"
                ).write_bytes(src_nwk.read_bytes())
                dst_fa.with_name(
                    f"group_{g3}_generated_matched_s{seed_m.group(1)}.fasta"
                ).write_bytes(src_fa.read_bytes())
            print(f"Promoted group {gid} -> {dst_nwk}")

        # append promotion note
        with md_path.open("a") as f:
            f.write("\n## Promoted for viz\n\n")
            for r in promoted:
                g3 = f"{r['group']:03d}"
                f.write(
                    f"- group {r['group']}: "
                    f"`group_{g3}_observed.nwk` ↔ `group_{g3}_generated_matched.nwk` "
                    f"(combo={r['combo_distance']:.4f}, leaves={r['n_gen_leaves']})\n"
                )

        winners_path = out_dir / "closest_groups_winners.json"
        winners_path.write_text(json.dumps(promoted, indent=2))
        print(f"Wrote {winners_path}")


if __name__ == "__main__":
    main()
