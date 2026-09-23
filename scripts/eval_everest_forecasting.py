#!/usr/bin/env python3
"""Clade forecasting eval on priority-virus forecasting datasets."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from src.dataset import TreeDataset
from src.flu_lit_sites import H1_SIGNAL_LEN_DEFAULT, H3_SIGNAL_LEN
from src.treeencoder.plm_embeddings import ESM2Embedder
from scripts.eval_single_tree import (
    AA_VOCAB,
    generate_tree,
    get_leaves,
    load_models,
    positional_recovery,
    seq_identity,
)
from benchmarks.metrics import sequences as S

MUT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY*])(\d+)([ACDEFGHIKLMNPQRSTVWY*])$", re.I)
EVEREST_RAW = (
    "https://github.com/debbiemarkslab/priority-viruses/tree/main/data/forecasting_dataset"
)

VIRUS_FOLDERS = {
    "h3n2": "H3N2",
    "h1n1": "H1N1",
    "covid": "SARS-CoV-2",
    "sars-cov-2": "SARS-CoV-2",
    "hiv": "HIV",
}


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def parse_clade_window(stem: str) -> tuple[str | None, str | None]:
    """Extract YYYY-MM-DD window from filenames like human_h3n2_3c2a1b2a1_2020-01-01_2022-01-01."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$", stem)
    if m:
        return m.group(1), m.group(2)
    return None, None


def parse_first_seen(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw[: len(fmt.replace("%", "0"))], fmt)
        except ValueError:
            continue
    if len(raw) >= 10:
        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d")
        except ValueError:
            pass
    return None


def load_clade_csv(path: Path, virus: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = {c.lower(): c for c in (reader.fieldnames or [])}
        mut_col = fields.get("mutation") or "mutation"
        count_col = fields.get("count") or fields.get("test_count") or "count"
        seen_col = fields.get("first_seen") or "first_seen"
        for row in reader:
            mut = (row.get(mut_col) or "").strip()
            if not mut or mut.lower() == "mutation":
                continue
            m = MUT_RE.match(mut)
            if not m:
                continue
            wt, pos_s, alt = m.group(1).upper(), m.group(2), m.group(3).upper()
            if alt == "X" or wt == "X":
                continue
            try:
                count = float(row.get(count_col) or 1)
            except ValueError:
                count = 1.0
            rows.append({
                "mutation": mut,
                "wt": wt,
                "pos_1based": int(pos_s),
                "alt": alt,
                "count": count,
                "first_seen": (row.get(seen_col) or "").strip(),
            })
    return rows


def mutation_to_col(
    mut: dict,
    virus: str,
    max_seq_len: int,
    root_seq: str,
) -> int | None:
    pos = mut["pos_1based"]
    v = virus.lower()
    if v in ("h3n2", "h1n1"):
        signal = H3_SIGNAL_LEN if v == "h3n2" else H1_SIGNAL_LEN_DEFAULT
        col = pos + signal - 1
    elif v in ("covid", "sars-cov-2"):
        col = pos - 1
    else:
        col = pos - 1
    if col < 0 or col >= max_seq_len or col >= len(root_seq):
        return None
    return col


def resolve_mutations(
    mutations: list[dict],
    virus: str,
    max_seq_len: int,
    root_seq: str,
) -> list[dict]:
    out = []
    for m in mutations:
        col = mutation_to_col(m, virus, max_seq_len, root_seq)
        out.append({**m, "col": col, "valid": col is not None})
    return out


def mutation_hit(root_seq: str, gen_seqs: list[str], col: int, alt: str) -> bool:
    if col is None or col >= len(root_seq):
        return False
    if root_seq[col] == alt:
        return False
    for gs in gen_seqs:
        if col < len(gs) and gs[col] != root_seq[col] and gs[col] == alt:
            return True
    return False


def clade_metrics(
    resolved: list[dict],
    root_seq: str,
    gen_seqs: list[str],
) -> dict:
    valid = [m for m in resolved if m.get("valid")]
    if not valid:
        return {
            "emerging_mut_recall": float("nan"),
            "weighted_recall": float("nan"),
            "n_mutations": 0,
            "n_hit": 0,
        }
    hits = [mutation_hit(root_seq, gen_seqs, m["col"], m["alt"]) for m in valid]
    weights = [m["count"] for m in valid]
    w_sum = sum(weights)
    return {
        "emerging_mut_recall": sum(hits) / len(valid),
        "weighted_recall": sum(h * w for h, w in zip(hits, weights)) / w_sum if w_sum else float("nan"),
        "n_mutations": len(valid),
        "n_hit": sum(hits),
    }


def enrichment_vs_random(
    resolved: list[dict],
    root_seq: str,
    gen_seqs: list[str],
    n_perm: int,
    rng: random.Random,
) -> float:
    valid = [m for m in resolved if m.get("valid")]
    if not valid or not gen_seqs:
        return float("nan")
    obs = clade_metrics(valid, root_seq, gen_seqs)["weighted_recall"]
    L = len(root_seq)
    null_weights = []
    for _ in range(n_perm):
        perm = []
        for m in valid:
            col = rng.randrange(L)
            alt = m["alt"]
            perm.append({**m, "col": col, "valid": True})
        null_weights.append(clade_metrics(perm, root_seq, gen_seqs)["weighted_recall"])
    null_mean = _mean(null_weights)
    if null_mean == null_mean and null_mean > 0:
        return obs / null_mean
    return float("nan")


def root_date_before(root_id: str, batch: dict, cutoff: datetime | None) -> bool:
    if cutoff is None:
        return True
    meta = batch.get("node_dates") or {}
    raw = meta.get(root_id) or root_id.split(",")[-1] if "," in root_id else ""
    if not raw and "," in root_id:
        raw = root_id.split(",", 1)[1]
    dt = parse_first_seen(str(raw))
    if dt is None:
        return True
    return dt <= cutoff


def discover_clade_csvs(everest_root: Path, clade_csv: Path | None) -> list[Path]:
    if clade_csv is not None:
        return [clade_csv]
    if not everest_root.is_dir():
        raise FileNotFoundError(
            f"EVEREST root missing: {everest_root}\n"
            f"Clone: git clone https://github.com/debbiemarkslab/priority-viruses external/priority-viruses\n"
            f"Or pass --clade-csv path/to.csv"
        )
    return sorted(everest_root.glob("*.csv"))


def eval_one_clade(
    clade_path: Path,
    virus: str,
    ds: TreeDataset,
    args,
    node_enc,
    tree_enc,
    rate_heads,
    embedder,
    tokenizer,
    esm_model,
    aa_token_ids,
    device,
    col_entropy,
    rng: random.Random,
) -> dict:
    window_start, window_end = parse_clade_window(clade_path.stem)
    end_dt = parse_first_seen(window_end or "")
    mutations = load_clade_csv(clade_path, virus)
    if window_end:
        mutations = [
            m for m in mutations
            if (dt := parse_first_seen(m["first_seen"])) is None or dt <= end_dt
        ]

    tree_rows = []
    recall_vals = []
    weighted_vals = []
    enrich_vals = []
    mut_rec_vals = []
    cov_e2_vals = []

    n_trees = min(len(ds.groups), args.max_trees)
    for i in range(n_trees):
        random.seed(args.seed + i)
        torch.manual_seed(args.seed + i)
        batch = ds[i]
        root_id = batch["node_ids"][batch["root_index"]]
        if not root_date_before(root_id, batch, end_dt):
            continue
        root_seq = batch["seqs"][root_id]
        resolved = resolve_mutations(mutations, virus, args.max_seq_len, root_seq)
        gt_leaves = [n for n in batch["node_ids"] if batch["is_leaf"][n]]
        if not gt_leaves:
            continue
        try:
            gen = generate_tree(
                root_seq,
                args.n_steps,
                args.max_seq_len,
                args.branch_rate_scale,
                args.max_leaves,
                args.mutation_rate_scale,
                node_enc,
                tree_enc,
                rate_heads,
                embedder,
                tokenizer,
                esm_model,
                aa_token_ids,
                device,
                col_entropy=col_entropy,
            )
        except Exception as e:
            tree_rows.append({"tree": i, "error": str(e)})
            continue

        gen_seqs = [gen.node_seqs[g] for g in get_leaves(gen)][: args.K]
        mets = clade_metrics(resolved, root_seq, gen_seqs)
        enrich = enrichment_vs_random(
            resolved, root_seq, gen_seqs, args.n_perm, rng
        )
        gt = batch["seqs"][rng.choice(gt_leaves)]
        best = max(gen_seqs, key=lambda gs: seq_identity(gt, gs), default=root_seq)
        pr = positional_recovery(root_seq, gt, best)
        cov = S.coverage_at_e([gt], gen_seqs, e=2)

        tree_rows.append({
            "tree": i,
            "root": root_id,
            **mets,
            "enrichment_vs_random": enrich,
            "mut_recovery": pr.get("mut_recovery"),
            "coverage_obs_e2": cov,
        })
        if mets["emerging_mut_recall"] == mets["emerging_mut_recall"]:
            recall_vals.append(mets["emerging_mut_recall"])
        if mets["weighted_recall"] == mets["weighted_recall"]:
            weighted_vals.append(mets["weighted_recall"])
        if enrich == enrich:
            enrich_vals.append(enrich)
        if pr.get("mut_recovery") == pr.get("mut_recovery"):
            mut_rec_vals.append(pr["mut_recovery"])
        if cov == cov:
            cov_e2_vals.append(cov)

    return {
        "clade_csv": str(clade_path),
        "window_start": window_start,
        "window_end": window_end,
        "n_clade_mutations": len(mutations),
        "K": args.K,
        "summary": {
            "emerging_mut_recall": _mean(recall_vals),
            "weighted_recall": _mean(weighted_vals),
            "enrichment_vs_random": _mean(enrich_vals),
            "mut_recovery": _mean(mut_rec_vals),
            "coverage_obs_e2": _mean(cov_e2_vals),
            "n_trees_scored": len(tree_rows),
        },
        "per_tree": tree_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="Test split dir (tree groups)")
    ap.add_argument("--virus", required=True, choices=sorted(VIRUS_FOLDERS.keys()))
    ap.add_argument("--max-seq-len", type=int, required=True)
    ap.add_argument(
        "--everest-root",
        default="",
        help="Path to forecasting_dataset/<Virus> (clone priority-viruses)",
    )
    ap.add_argument("--clade-csv", default="", help="Single clade CSV (overrides --everest-root scan)")
    ap.add_argument("--K", type=int, default=16, help="Max generated leaves scored per tree")
    ap.add_argument("--max-trees", type=int, default=20)
    ap.add_argument("--max-leaves", type=int, default=200)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--mutation-rate-scale", type=float, default=0.5)
    ap.add_argument("--branch-rate-scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-perm", type=int, default=50, help="Random baseline permutations")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    folder = VIRUS_FOLDERS[args.virus.lower()]
    everest_root = Path(args.everest_root) if args.everest_root else ROOT / "external/priority-viruses/data/forecasting_dataset" / folder
    clade_csv = Path(args.clade_csv) if args.clade_csv else None
    clade_paths = discover_clade_csvs(everest_root, clade_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    node_enc, tree_enc, rate_heads, col_entropy = load_models(
        args.checkpoint, device, args.max_seq_len
    )
    embedder = ESM2Embedder(device=device)
    model_id = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    esm_model = EsmForMaskedLM.from_pretrained(model_id).to(device)
    esm_model.eval()
    aa_token_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(aa) for aa in AA_VOCAB], dtype=torch.long
    )

    ds = TreeDataset(str(ROOT / args.data), max_seq_len=args.max_seq_len)
    rng = random.Random(args.seed)

    results = []
    for cp in clade_paths:
        print(f"Clade: {cp.name}")
        results.append(
            eval_one_clade(
                cp, args.virus, ds, args,
                node_enc, tree_enc, rate_heads, embedder,
                tokenizer, esm_model, aa_token_ids, device, col_entropy, rng,
            )
        )

    payload = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "virus": args.virus,
        "everest_source": EVEREST_RAW,
        "K": args.K,
        "clades": results,
        "aggregate": {
            "emerging_mut_recall": _mean([
                r["summary"]["emerging_mut_recall"] for r in results
            ]),
            "weighted_recall": _mean([
                r["summary"]["weighted_recall"] for r in results
            ]),
            "enrichment_vs_random": _mean([
                r["summary"]["enrichment_vs_random"] for r in results
            ]),
            "mut_recovery": _mean([r["summary"]["mut_recovery"] for r in results]),
            "coverage_obs_e2": _mean([r["summary"]["coverage_obs_e2"] for r in results]),
            "n_clades": len(results),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
