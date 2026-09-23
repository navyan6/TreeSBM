#!/usr/bin/env python3
"""Held-out eval: mutation recovery plus EVEscape enrichment.

Generates trees from each test root and reports recovery metrics and mean
per-mutation EVEscape (optionally restricted to a lit-hotspot mask).

Example::

    python scripts/eval_evescape_enrichment.py \
        --checkpoint checkpoints/.../best.pt --data data/covid/test \
        --max-seq-len 1280 --evescape data/covid/evescape_spike_rbd.pt
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from src.dataset import TreeDataset
from src.tree_state import TreeState
from src.treeencoder.plm_embeddings import ESM2Embedder
from src.spike_domains import (
    PRIMARY_DOMAIN_KEYS,
    REGION_RECALL_KEYS,
    SPIKE_DOMAINS_1BASED,
    all_domain_masks,
    domain_mut_frac,
    region_any_mut,
    region_site_recall,
    write_spike_region_annotations,
)
from src.flu_lit_sites import (
    HA_DOMAINS_H3_1BASED,
    HA_RBS_SITES_H3_1BASED,
    H3_SIGNAL_LEN,
    PRIMARY_HA_DOMAIN_KEYS,
    REGION_RECALL_HA_KEYS,
    all_ha_domain_masks,
    write_ha_region_annotations,
)
from benchmarks.metrics.sequences import (
    any_descendant_mut_recovery,
    path_union_mutation_recovery,
)
from scripts.eval_single_tree import (
    load_models, generate_tree, positional_recovery, get_leaves, seq_identity,
    AA_VOCAB,
)

AA_TO_IDX = {a: i for i, a in enumerate(AA_VOCAB)}


def _mean(xs):
    xs = [x for x in xs if x == x]  # drop nan
    return sum(xs) / len(xs) if xs else float("nan")


def mutation_evescape(
    root_seq: str,
    leaf_seq: str,
    evescape: torch.Tensor,
    L: int,
    site_mask: torch.Tensor | None = None,
):
    """Per-mutation EVEscape look-ups for root→leaf AA changes.

    For each column where leaf_aa != root_aa (valid AA), look up
    EVEscape[pos, leaf_aa]. Scores with tensor entry 0.0 are skipped (unscored
    region / missing substitution).

    If ``site_mask`` (bool [L]) is given, only mutations at True sites contribute
    to the returned score list (antigenic / lit-hotspot restriction). ``n_muts``
    is always the total root→leaf mutation count (mask-agnostic denominator for
    frac_scored companions).

    Returns (list_of_scores, n_muts_total, n_muts_in_mask).
    Aggregation elsewhere is a **simple mean** of list_of_scores — not a
    product, log-sum-exp, or strain-level score.
    """
    scores = []
    n_muts = 0
    n_muts_in_mask = 0
    mask = site_mask.bool() if site_mask is not None else None
    for p in range(min(len(root_seq), len(leaf_seq), L)):
        r, g = root_seq[p], leaf_seq[p]
        if g != r and g in AA_TO_IDX:
            n_muts += 1
            in_mask = True if mask is None else bool(mask[p].item())
            if in_mask:
                n_muts_in_mask += 1
            if not in_mask:
                continue
            s = evescape[p, AA_TO_IDX[g]].item()
            if s != 0.0:
                scores.append(s)
    return scores, n_muts, n_muts_in_mask


def random_baseline_evescape(
    evescape: torch.Tensor,
    site_mask: torch.Tensor | None = None,
) -> float:
    """Mean of nonzero EVEscape tensor entries, optionally restricted to sites.

    Comparable random baseline for mean-over-mutations metrics: draws uniformly
    among official scored substitutions (and among antigenic columns when
    ``site_mask`` is set). Not a sequence-level product.
    """
    if site_mask is None:
        nz = evescape[evescape != 0]
    else:
        # Keep AA axis; zero out non-antigenic columns.
        masked = evescape.clone()
        masked[~site_mask.bool()] = 0
        nz = masked[masked != 0]
    if nz.numel() == 0:
        return float("nan")
    return nz.mean().item()


def mutating_cols(root_seq: str, leaf_seq: str, L: int) -> list[int]:
    """0-based columns where leaf AA differs from root (valid AA only)."""
    cols = []
    for p in range(min(len(root_seq), len(leaf_seq), L)):
        r, g = root_seq[p], leaf_seq[p]
        if g != r and g in AA_TO_IDX and r in AA_TO_IDX:
            cols.append(p)
    return cols


def hotspot_mut_frac(root_seq: str, leaf_seq: str, hotspot: torch.Tensor, L: int) -> float:
    """
    Fraction of root→leaf mutation columns that fall in ``hotspot`` (bool [L]).

    Denominator = all positions with leaf≠root (valid AAs). Numerator = those
    columns where hotspot[col] is True. NaN if the leaf has no mutations.
    """
    cols = mutating_cols(root_seq, leaf_seq, L)
    if not cols:
        return float("nan")
    hot = hotspot.bool()
    n_hot = sum(1 for c in cols if bool(hot[c].item()))
    return n_hot / len(cols)


def load_hotspot_mask(path: str | None, L: int) -> tuple[torch.Tensor | None, dict]:
    """Load bool [L] mask from .pt (dict with mut_hotspot_mask or bare tensor)."""
    if not path:
        return None, {}
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = {}
    if isinstance(blob, dict) and "mut_hotspot_mask" in blob:
        mask = blob["mut_hotspot_mask"].bool().cpu()
        meta = {
            "score": blob.get("score"),
            "spike_positions_1based": blob.get("spike_positions_1based"),
            "h3_positions_1based": blob.get("h3_positions_1based"),
            "orf_positions_1based": blob.get("orf_positions_1based"),
            "signal_len": blob.get("signal_len"),
            "source": blob.get("source"),
            "path": path,
        }
    else:
        mask = torch.as_tensor(blob).bool().cpu()
        meta = {"path": path}
    if mask.ndim != 1 or mask.numel() != L:
        raise SystemExit(
            f"--lit-hotspot-mask length {tuple(mask.shape)} != max_seq_len={L}"
        )
    return mask, meta


def _lit_family(meta: dict) -> str:
    """Classify lit mask as 'flu' | 'covid' | 'unknown' from score/path only.

    Infer pathogen family from mask metadata / path, not max_seq_len alone.
    """
    score = str(meta.get("score") or "").lower()
    path = str(meta.get("path") or "").lower()
    blob = f"{score} {path}"
    is_flu = any(
        tok in blob
        for tok in (
            "nmicrobiol",
            "flu_mutfreq",
            "flu_lit",
            "h3_globular",
            "h3n2",
            "ha_hotspot",
            "h1_nmicrobiol",
            "h1_globular",
            "antigenic_guidance",
            "flu_h1",
        )
    )
    is_covid = any(
        tok in blob
        for tok in ("pmc10142771", "pmc_lit", "covid_mutfreq", "rbd_band", "spike")
    )
    if is_flu and not is_covid:
        return "flu"
    if is_covid and not is_flu:
        return "covid"
    return "unknown"


def _lit_metric_names(meta: dict) -> tuple[str, str, str]:
    """Return (gen_key, gt_key, label) for lit hotspot frac metrics.

    COVID PMC → pmc_*; flu/nmicrobiol → flu_*; always also lit_*.
    """
    fam = _lit_family(meta)
    if fam == "flu":
        return "flu_hotspot_mut_frac", "gt_flu_hotspot_mut_frac", "flu/nmicrobiol lit"
    if fam == "covid":
        return "pmc_hotspot_mut_frac", "gt_pmc_hotspot_mut_frac", "PMC lit"
    return "lit_hotspot_mut_frac", "gt_lit_hotspot_mut_frac", "lit hotspot"


def _assert_no_cross_pathogen_mask(meta: dict, L: int, path: str) -> None:
    """Refuse flu↔COVID mask mixups (path/score vs length)."""
    fam = _lit_family(meta)
    pl = path.lower()
    if "covid_mutfreq" in pl or "pmc_lit" in pl or "pmc10142771" in str(meta.get("score", "")).lower():
        if L <= 600:
            raise SystemExit(
                f"REFUSED: COVID/PMC hotspot mask ({path}) with max_seq_len={L} "
                f"(flu HA). Flu eval must use results/flu_mutfreq_vs_lit/* only."
            )
    if "flu_mutfreq" in pl or "nmicrobiol" in pl or fam == "flu":
        if L >= 1000:
            raise SystemExit(
                f"REFUSED: flu/nmicrobiol hotspot mask ({path}) with max_seq_len={L} "
                f"(COVID spike). COVID eval must use results/covid_mutfreq_vs_lit/* only."
            )


def gt_leaves_of(batch: dict) -> list[str]:
    parents = {p for p, _ in batch["edges"]}
    return [nid for nid in batch["node_ids"] if nid not in parents]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-seq-len", type=int, default=1280)
    ap.add_argument(
        "--evescape",
        default=None,
        help="[L,20] EVEscape .pt from prepare_evescape.py. Use Spike RBD for "
             "COVID and flu_h1 for H1N1 — never cross pathogens.",
    )
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--max-leaves", type=int, default=300)
    ap.add_argument("--branch-rate-scale", type=float, default=6.0)
    ap.add_argument("--mutation-rate-scale", type=float, default=0.3)
    ap.add_argument("--site-softmax-sample", action="store_true",
                    help="Use site-propensity sampling (categorical over sites, then AA|site)")
    ap.add_argument("--site-temperature", type=float, default=1.0,
                    help="Temperature on site-propensity logits (--site-softmax-sample)")
    ap.add_argument(
        "--cache-esm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Incremental seq-keyed ESM emb + R0/MLM cache during generation "
             "(default: on). Use --no-cache-esm to disable.",
    )
    ap.add_argument("--max-trees", type=int, default=20)
    ap.add_argument("--gt-leaves-sampled", type=int, default=30,
                    help="GT leaves per tree to match against gen for recovery")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--pmc-hotspot-mask",
        default=None,
        help="Deprecated alias of --lit-hotspot-mask. COVID scripts may still pass "
             "results/covid_mutfreq_vs_lit/mut_hotspot_mask_pmc_lit.pt here. "
             "NO length-based default — omit both flags (or pass '') to disable.",
    )
    ap.add_argument(
        "--lit-hotspot-mask",
        default=None,
        help="Bool [L] curated lit-hotspot mask .pt. Use pathogen-matched masks: "
             "flu → results/flu_mutfreq_vs_lit/mut_hotspot_mask_nmicrobiol_lit.pt; "
             "COVID → results/covid_mutfreq_vs_lit/mut_hotspot_mask_pmc_lit.pt. "
             "No cross-default. Empty string disables.",
    )
    ap.add_argument(
        "--spike-domain-metrics",
        action="store_true",
        default=None,
        help="Emit NTD/RBD/RBM/S1/S2/furin mut-frac + region site_recall/any_mut "
             "(Wuhan 1-based bands). Default: on when max_seq_len>=1000 (COVID), "
             "off for flu.",
    )
    ap.add_argument(
        "--no-spike-domain-metrics",
        action="store_true",
        help="Disable spike domain mut-fraction metrics even for COVID-length L.",
    )
    ap.add_argument(
        "--ha-region-metrics",
        action="store_true",
        default=None,
        help="Emit HA globular-head mut-frac + region site_recall/any_mut "
             "(H3 #63–252). Default: on when max_seq_len<=600 (flu HA), off for COVID.",
    )
    ap.add_argument(
        "--no-ha-region-metrics",
        action="store_true",
        help="Disable HA head region metrics even for flu-length L.",
    )
    ap.add_argument(
        "--region-annotations-dir",
        default=None,
        help="Write region_annotations.json under this dir (default: "
             "results/covid_mutfreq_vs_lit or results/flu_mutfreq_vs_lit).",
    )
    # Generation ablations (reuse full checkpoint; no retrain)
    ap.add_argument("--ablate-bridge", action="store_true",
                    help="Force log R_θ = log R0 (no learned c_θ).")
    ap.add_argument("--ablate-tree-context", action="store_true",
                    help="Zero tree-encoder embeddings into RateHeads.")
    ap.add_argument("--ablate-branch-length-head", action="store_true",
                    help="Use constant BL=dt instead of the branch-length head.")
    ap.add_argument("--ablate-internal-node-seqs", action="store_true",
                    help="Zero PLM embeddings for non-leaf (internal) nodes.")
    ap.add_argument("--branching-mode", choices=["learned", "poisson_ref"],
                    default="learned",
                    help="poisson_ref: disable sequence-dependent branching.")
    ap.add_argument("--ref-lambda", type=float, default=1.0,
                    help="Constant Poisson λ when --branching-mode poisson_ref.")
    ap.add_argument("--fitness-beta", type=float, default=None,
                    help="Override fitness tilt β (default: checkpoint / 0).")
    ap.add_argument(
        "--r0-backend",
        default=None,
        help="Swap frozen R0 prior (esm2 / esm2_650m / esmc / jtt / wag / lg / progen2).",
    )
    ap.add_argument("--r0-model", default=None, help="Optional R0 model id override.")
    ap.add_argument(
        "--no-lit-hotspot-mask",
        action="store_true",
        help="Turn off curated lit/PMC hotspot mask at eval "
             "(no antigenic/pmc_hotspot_mut_frac; generation unchanged).",
    )
    args = ap.parse_args()

    # Explicit path only — never infer flu vs COVID from max_seq_len.
    if args.no_lit_hotspot_mask:
        resolved_lit_mask = ""
    elif args.lit_hotspot_mask is not None:
        resolved_lit_mask = args.lit_hotspot_mask
    elif args.pmc_hotspot_mask is not None:
        resolved_lit_mask = args.pmc_hotspot_mask
    else:
        resolved_lit_mask = ""
    args.pmc_hotspot_mask = resolved_lit_mask
    args.lit_hotspot_mask = resolved_lit_mask

    # Domain metrics: COVID spike only (never auto-enable for flu HA lengths).
    if args.no_spike_domain_metrics:
        args.spike_domain_metrics = False
    elif args.spike_domain_metrics is None:
        args.spike_domain_metrics = args.max_seq_len >= 1000
    if args.spike_domain_metrics and args.max_seq_len < 1000:
        raise SystemExit(
            f"REFUSED: --spike-domain-metrics with max_seq_len={args.max_seq_len} "
            "(flu HA). Spike domain bands are Wuhan/P0DTC2 only; use COVID L≥1000."
        )
    # HA head metrics: flu only (never auto-enable for COVID spike lengths).
    if args.no_ha_region_metrics:
        args.ha_region_metrics = False
    elif args.ha_region_metrics is None:
        args.ha_region_metrics = args.max_seq_len <= 600
    if args.ha_region_metrics and args.max_seq_len >= 1000:
        raise SystemExit(
            f"REFUSED: --ha-region-metrics with max_seq_len={args.max_seq_len} "
            "(COVID spike). HA head bands are H3-numbered only; use flu L≤600."
        )
    if args.spike_domain_metrics and args.ha_region_metrics:
        raise SystemExit(
            "REFUSED: cannot enable both --spike-domain-metrics and "
            "--ha-region-metrics (pathogen separation)."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if args.no_lit_hotspot_mask:
        print("Lit hotspot mask DISABLED (--no-lit-hotspot-mask)")
    r0_live = None
    if args.r0_backend:
        from src.r0_backends import STUB_BACKENDS, build_r0_backend, normalize_backend_name
        r0_name = normalize_backend_name(args.r0_backend)
        print(f"R0 backend override: {r0_name}")
        if r0_name in STUB_BACKENDS:
            raise SystemExit(
                f"R0 backend {r0_name!r} is stubbed (skipped). "
                "Install/wire it or pick esm2 / esm2_650m / esmc / jtt / wag / lg."
            )
        r0_live = build_r0_backend(r0_name, model_id=args.r0_model, device=device)
    node_enc, tree_enc, rate_heads, col_entropy = load_models(
        args.checkpoint, device, args.max_seq_len)
    embedder = ESM2Embedder(device=device)
    mid = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(mid)
    esm_model = EsmForMaskedLM.from_pretrained(mid).to(device).eval()
    for p in esm_model.parameters():
        p.requires_grad = False
    aa_token_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(a) for a in AA_VOCAB], dtype=torch.long)

    evescape = base_ev = None
    ev_meta: dict = {}
    if args.evescape:
        # prepare_evescape.py saves a dict ({"scores": [L,20], ...}), not a bare tensor.
        ev_blob = torch.load(args.evescape, map_location="cpu", weights_only=False)
        if isinstance(ev_blob, dict):
            evescape = ev_blob["scores"]
            ev_meta = {
                k: ev_blob.get(k)
                for k in (
                    "pathogen",
                    "score_col",
                    "score_scale",
                    "score_scale_note",
                    "standardized",
                    "match_rate",
                    "source_csv",
                    "mean",
                    "std",
                )
                if k in ev_blob
            }
        else:
            evescape = ev_blob
        # Pathogen / length sanity (soft warn; hard refuse known cross-use).
        path_l = args.evescape.lower()
        if ("spike" in path_l or "rbd" in path_l or "covid" in path_l) and args.max_seq_len <= 600:
            raise SystemExit(
                f"REFUSED: COVID/Spike EVEscape tensor ({args.evescape}) with "
                f"max_seq_len={args.max_seq_len} (flu HA). Use data/evescape_h1n1_ha.pt."
            )
        if ("h1n1" in path_l or "flu_h1" in path_l or "h1_ha" in path_l) and args.max_seq_len >= 1000:
            raise SystemExit(
                f"REFUSED: flu H1 EVEscape tensor ({args.evescape}) with "
                f"max_seq_len={args.max_seq_len} (COVID spike). Use "
                f"data/covid/evescape_spike_rbd.pt."
            )
        if ev_meta.get("standardized"):
            print(
                "WARNING: EVEscape tensor is z-scored (standardized=True). "
                "Official paper scores are log-scale (mean≈−2.3). Rebuild with "
                "scripts/prepare_evescape.py --no-standardize."
            )
        nz = evescape[evescape != 0]
        base_ev = random_baseline_evescape(evescape)  # all scored substitutions
        scale = ev_meta.get("score_scale") or (
            "zscored_evescape" if ev_meta.get("standardized") else "official_evescape_log_product"
        )
        print(
            f"EVEscape {tuple(evescape.shape)}  nonzero={nz.numel()}  "
            f"random-baseline(mean nonzero)={base_ev:.4f}  scale={scale}"
        )
        if abs(base_ev) < 0.05 and not ev_meta.get("standardized"):
            print(
                "NOTE: baseline≈0 with standardized=False is unexpected for "
                "official EVEscape (usually ≈−2.3)."
            )

    pmc_mask_path = args.lit_hotspot_mask.strip() if args.lit_hotspot_mask else ""
    if pmc_mask_path and not Path(pmc_mask_path).exists():
        print(
            f"NOTE: lit hotspot mask not found at {pmc_mask_path}; "
            f"skipping lit/pmc/flu_hotspot_mut_frac"
        )
        pmc_mask_path = ""
    pmc_mask, pmc_meta = load_hotspot_mask(pmc_mask_path or None, args.max_seq_len)
    lit_gen_key, lit_gt_key, lit_label = (
        "lit_hotspot_mut_frac", "gt_lit_hotspot_mut_frac", "lit hotspot"
    )
    base_ev_ag = float("nan")
    if pmc_mask is not None:
        _assert_no_cross_pathogen_mask(pmc_meta, args.max_seq_len, pmc_mask_path)
        lit_gen_key, lit_gt_key, lit_label = _lit_metric_names(pmc_meta)
        print(
            f"Lit hotspot mask ({lit_label}): n={int(pmc_mask.sum())}/{args.max_seq_len}  "
            f"score={pmc_meta.get('score')}  path={pmc_mask_path}"
        )
        if evescape is not None:
            base_ev_ag = random_baseline_evescape(evescape, pmc_mask)
            print(
                f"  antigenic EVEscape random baseline "
                f"(nonzero @ lit sites)={base_ev_ag:.4f}"
            )

    ds = TreeDataset(args.data, max_seq_len=args.max_seq_len)
    n = min(args.max_trees, len(ds)) if args.max_trees else len(ds)
    print(f"Evaluating {n} test trees  (mut_rate={args.mutation_rate_scale} n_steps={args.n_steps})\n")

    recs, rets, idents = [], [], []
    site_recs, aa_accs, site_precs = [], [], []
    any_desc_recs, any_desc_site, path_union_site_rec = [], [], []
    dist_to_roots, gt_dist_to_roots = [], []  # mean Hamming(root, best-match gen / GT)
    # Per-mutation EVEscape scores pooled across leaves (simple mean later).
    # *_all = any scored site; *_ag = lit/antigenic mask sites only.
    model_ev, gt_ev = [], []
    model_ev_ag, gt_ev_ag = [], []
    model_frac_rbd, gt_frac_rbd = [], []  # fraction of muts with nonzero EVEscape
    model_n_ag_muts, gt_n_ag_muts = [], []  # antigenic mut counts per leaf (incl. unscored)
    # Per-leaf fraction of mutations at PMC lit sites (gen≠root / GT≠root)
    model_pmc_frac, gt_pmc_frac = [], []
    # Region metrics (COVID spike XOR flu HA head — pathogen-separated)
    spike_masks = (
        all_domain_masks(args.max_seq_len) if args.spike_domain_metrics else {}
    )
    ha_masks = (
        all_ha_domain_masks(args.max_seq_len, signal_len=H3_SIGNAL_LEN)
        if args.ha_region_metrics else {}
    )
    mut_frac_keys = (
        list(PRIMARY_DOMAIN_KEYS) if spike_masks
        else list(PRIMARY_HA_DOMAIN_KEYS) if ha_masks
        else []
    )
    recall_keys = (
        list(REGION_RECALL_KEYS) if spike_masks
        else list(REGION_RECALL_HA_KEYS) if ha_masks
        else []
    )
    region_masks = spike_masks or ha_masks
    domain_gen: dict[str, list[float]] = {k: [] for k in mut_frac_keys}
    domain_gt: dict[str, list[float]] = {k: [] for k in mut_frac_keys}
    region_site_rec: dict[str, list[float]] = {k: [] for k in recall_keys}
    region_any_gen: dict[str, list[float]] = {k: [] for k in recall_keys}
    region_any_gt: dict[str, list[float]] = {k: [] for k in recall_keys}
    ann_paths: list[str] = []
    if spike_masks:
        print(
            "Spike domain metrics (Wuhan 1-based mut_frac + region site_recall/any_mut): "
            + ", ".join(
                f"{k}={SPIKE_DOMAINS_1BASED[k][0]}–{SPIKE_DOMAINS_1BASED[k][1]}"
                for k in PRIMARY_DOMAIN_KEYS
            )
        )
        ann_dir = Path(
            args.region_annotations_dir or "results/covid_mutfreq_vs_lit"
        )
        ann_paths.append(
            str(write_spike_region_annotations(
                ann_dir / "region_annotations.json", L=args.max_seq_len
            ))
        )
    if ha_masks:
        lo, hi = HA_DOMAINS_H3_1BASED["HA_head"]
        print(
            f"HA head region metrics (H3 #{lo}–{hi}; "
            f"col = h3_pos + {H3_SIGNAL_LEN} - 1): "
            "ha_head_mut_frac + ha_head_site_recall + ha_head_any_mut"
        )
        ann_dir = Path(
            args.region_annotations_dir or "results/flu_mutfreq_vs_lit"
        )
        ann_paths.append(
            str(write_ha_region_annotations(
                ann_dir / "region_annotations.json", L=args.max_seq_len
            ))
        )
    if ann_paths:
        print(f"Region annotations -> {', '.join(ann_paths)}")
    per_tree = []
    rng = random.Random(args.seed)

    def _rk(name: str) -> str:
        """Metric key prefix: NTD→ntd, HA_head→ha_head, furin→furin."""
        return name.lower()

    for i in range(n):
        batch = ds[i]
        root_id = batch["node_ids"][batch["root_index"]]
        root_seq = batch["seqs"][root_id]
        try:
            random.seed(args.seed + i)
            torch.manual_seed(args.seed + i)
            gen = generate_tree(
                root_seq, args.n_steps, args.max_seq_len, args.branch_rate_scale,
                args.max_leaves, args.mutation_rate_scale, node_enc, tree_enc, rate_heads,
                embedder, tokenizer, esm_model, aa_token_ids, device, col_entropy=col_entropy,
                site_softmax_sample=args.site_softmax_sample,
                site_temperature=args.site_temperature,
                cache_esm=args.cache_esm,
                ablate_bridge=args.ablate_bridge,
                ablate_tree_context=args.ablate_tree_context,
                branching_mode=args.branching_mode,
                ref_lambda=args.ref_lambda,
                ablate_branch_length_head=args.ablate_branch_length_head,
                ablate_internal_node_seqs=args.ablate_internal_node_seqs,
                fitness_beta=args.fitness_beta,
                r0_backend=r0_live,
            )
        except Exception as e:
            print(f"[{i+1}/{n}] ERROR: {e}")
            continue

        gen_leaves = get_leaves(gen)
        gen_seqs = [gen.node_seqs[g] for g in gen_leaves]
        gt_leaves = gt_leaves_of(batch)
        gt_sample = rng.sample(gt_leaves, min(args.gt_leaves_sampled, len(gt_leaves)))

        # recovery/retention + identity: best-match gen leaf per sampled GT leaf
        # PRIMARY metrics are leaf-only (gen leaf vs GT leaf vs root).
        t_rec, t_ret, t_id = [], [], []
        t_site_rec, t_aa_acc, t_site_prec = [], [], []
        t_any_desc, t_any_site = [], []
        t_dist_root, t_gt_dist_root = [], []
        tree_region_site: dict[str, list[float]] = {k: [] for k in recall_keys}
        gt_sample_seqs = []
        for gl in gt_sample:
            gt_seq = batch["seqs"][gl]
            gt_sample_seqs.append(gt_seq)
            best, best_id = None, -1.0
            for gs in gen_seqs:
                idv = seq_identity(gt_seq, gs)
                if idv > best_id:
                    best_id, best = idv, gs
            if best is None:
                continue
            r = positional_recovery(root_seq, gt_seq, best)
            t_rec.append(r["mut_recovery"]); t_ret.append(r["cons_retention"]); t_id.append(best_id)
            t_site_rec.append(r["site_recall"]); t_aa_acc.append(r["aa_acc_given_hit"])
            t_site_prec.append(r["site_precision"])
            # Hamming distance to root (AA edits) for best-match gen and GT leaf.
            Lr = min(len(root_seq), len(best), len(gt_seq))
            t_dist_root.append(sum(root_seq[i] != best[i] for i in range(Lr)))
            t_gt_dist_root.append(sum(root_seq[i] != gt_seq[i] for i in range(Lr)))
            # Region site_recall: among GT muts in R, did best-match gen also mutate?
            for name in recall_keys:
                f = region_site_recall(
                    root_seq, gt_seq, best, region_masks[name]
                )
                if f == f:
                    tree_region_site[name].append(f)
                    region_site_rec[name].append(f)
            # Tree-wide companion: credit if ANY gen leaf recovers the GT leaf mut.
            ad = any_descendant_mut_recovery(root_seq, gt_seq, gen_seqs)
            if ad["mut_recovery_any_descendant"] == ad["mut_recovery_any_descendant"]:
                t_any_desc.append(ad["mut_recovery_any_descendant"])
            if ad["site_recall_any_descendant"] == ad["site_recall_any_descendant"]:
                t_any_site.append(ad["site_recall_any_descendant"])
        recs.append(_mean(t_rec)); rets.append(_mean(t_ret)); idents.append(_mean(t_id))
        site_recs.append(_mean(t_site_rec))
        aa_accs.append(_mean(t_aa_acc))
        site_precs.append(_mean(t_site_prec))
        any_desc_recs.append(_mean(t_any_desc))
        any_desc_site.append(_mean(t_any_site))
        dist_to_roots.append(_mean(t_dist_root))
        gt_dist_to_roots.append(_mean(t_gt_dist_root))
        # Path-union set recovery over sampled GT leaves vs all gen leaves.
        pu = path_union_mutation_recovery(root_seq, gt_sample_seqs, gen_seqs)
        path_union_site_rec.append(pu["mut_site_recall_path_union"])
        # EVEscape: pool per-mutation scores across gen leaves and (real) GT leaves.
        # Mean of official per-mut scores — not a strain-level product/sum.
        tree_model_ev, tree_gt_ev = [], []
        tree_model_ev_ag, tree_gt_ev_ag = [], []
        if evescape is not None:
            for gs in gen_seqs:
                sc, nm, _ = mutation_evescape(
                    root_seq, gs, evescape, args.max_seq_len
                )
                tree_model_ev += sc
                if nm:
                    model_frac_rbd.append(len(sc) / nm)
                if pmc_mask is not None:
                    sc_ag, _, n_ag = mutation_evescape(
                        root_seq, gs, evescape, args.max_seq_len, site_mask=pmc_mask
                    )
                    tree_model_ev_ag += sc_ag
                    model_n_ag_muts.append(n_ag)
            for gl in gt_sample:
                gseq = batch["seqs"][gl]
                sc, nm, _ = mutation_evescape(
                    root_seq, gseq, evescape, args.max_seq_len
                )
                tree_gt_ev += sc
                if nm:
                    gt_frac_rbd.append(len(sc) / nm)
                if pmc_mask is not None:
                    sc_ag, _, n_ag = mutation_evescape(
                        root_seq, gseq, evescape, args.max_seq_len, site_mask=pmc_mask
                    )
                    tree_gt_ev_ag += sc_ag
                    gt_n_ag_muts.append(n_ag)
            model_ev += tree_model_ev
            gt_ev += tree_gt_ev
            model_ev_ag += tree_model_ev_ag
            gt_ev_ag += tree_gt_ev_ag

        # PMC lit hotspot consistency: frac of mut columns in curated sites
        tree_pmc_gen, tree_pmc_gt = [], []
        if pmc_mask is not None:
            for gs in gen_seqs:
                f = hotspot_mut_frac(root_seq, gs, pmc_mask, args.max_seq_len)
                if f == f:  # not nan
                    tree_pmc_gen.append(f)
                    model_pmc_frac.append(f)
            for gl in gt_sample:
                f = hotspot_mut_frac(
                    root_seq, batch["seqs"][gl], pmc_mask, args.max_seq_len
                )
                if f == f:
                    tree_pmc_gt.append(f)
                    gt_pmc_frac.append(f)

        # Region mut_frac + binary any_mut (gen≠root / GT≠root columns)
        tree_dom_gen: dict[str, list[float]] = {k: [] for k in mut_frac_keys}
        tree_dom_gt: dict[str, list[float]] = {k: [] for k in mut_frac_keys}
        tree_any_gen: dict[str, list[float]] = {k: [] for k in recall_keys}
        tree_any_gt: dict[str, list[float]] = {k: [] for k in recall_keys}
        if region_masks:
            for gs in gen_seqs:
                cols = mutating_cols(root_seq, gs, args.max_seq_len)
                for name in mut_frac_keys:
                    f = domain_mut_frac(cols, region_masks[name])
                    if f == f:
                        tree_dom_gen[name].append(f)
                        domain_gen[name].append(f)
                for name in recall_keys:
                    a = region_any_mut(root_seq, gs, region_masks[name])
                    tree_any_gen[name].append(a)
                    region_any_gen[name].append(a)
            for gl in gt_sample:
                gseq = batch["seqs"][gl]
                cols = mutating_cols(root_seq, gseq, args.max_seq_len)
                for name in mut_frac_keys:
                    f = domain_mut_frac(cols, region_masks[name])
                    if f == f:
                        tree_dom_gt[name].append(f)
                        domain_gt[name].append(f)
                for name in recall_keys:
                    a = region_any_mut(root_seq, gseq, region_masks[name])
                    tree_any_gt[name].append(a)
                    region_any_gt[name].append(a)

        tree_rec = {
            "tree": i, "gen_leaves": len(gen_leaves),
            "mut_recovery": _mean(t_rec), "cons_retention": _mean(t_ret),
            "site_recall": _mean(t_site_rec),
            "aa_acc_given_hit": _mean(t_aa_acc),
            "site_precision": _mean(t_site_prec),
            "identity": _mean(t_id),
            "dist_to_root": _mean(t_dist_root),
            "gt_dist_to_root": _mean(t_gt_dist_root),
            # Tree-wide companions (do not replace leaf mut_recovery)
            "mut_recovery_any_descendant": _mean(t_any_desc),
            "site_recall_any_descendant": _mean(t_any_site),
            "mut_site_recall_path_union": pu["mut_site_recall_path_union"],
            "mut_sub_recall_path_union": pu["mut_sub_recall_path_union"],
            "model_evescape": _mean(tree_model_ev) if evescape is not None else None,
            "gt_evescape": _mean(tree_gt_ev) if evescape is not None else None,
            # Explicit aliases: mean over ALL scored muts (legacy whole-set).
            "evescape_mean_all_scored_muts": (
                _mean(tree_model_ev) if evescape is not None else None
            ),
            "gt_evescape_mean_all_scored_muts": (
                _mean(tree_gt_ev) if evescape is not None else None
            ),
            "metrics_scope": "leaf_only_primary; any_descendant_and_path_union_are_extra",
        }
        if evescape is not None and pmc_mask is not None:
            tree_rec["evescape_mean_antigenic_muts"] = _mean(tree_model_ev_ag)
            tree_rec["gt_evescape_mean_antigenic_muts"] = _mean(tree_gt_ev_ag)
            tree_rec["n_evescape_antigenic_muts_scored"] = len(tree_model_ev_ag)
            tree_rec["n_gt_evescape_antigenic_muts_scored"] = len(tree_gt_ev_ag)
        if pmc_mask is not None:
            gen_frac = _mean(tree_pmc_gen)
            gt_frac = _mean(tree_pmc_gt)
            tree_rec["lit_hotspot_mut_frac"] = gen_frac
            tree_rec["gt_lit_hotspot_mut_frac"] = gt_frac
            tree_rec[lit_gen_key] = gen_frac
            tree_rec[lit_gt_key] = gt_frac
        if region_masks:
            for name in mut_frac_keys:
                key = f"{_rk(name)}_mut_frac"
                tree_rec[key] = _mean(tree_dom_gen[name])
                tree_rec[f"gt_{key}"] = _mean(tree_dom_gt[name])
            for name in recall_keys:
                sk = f"{_rk(name)}_site_recall"
                ak = f"{_rk(name)}_any_mut"
                tree_rec[sk] = _mean(tree_region_site[name])
                tree_rec[ak] = _mean(tree_any_gen[name])
                tree_rec[f"gt_{ak}"] = _mean(tree_any_gt[name])
        per_tree.append(tree_rec)
        print(f"[{i+1}/{n}] gen_leaves={len(gen_leaves):3d}  "
              f"recovery={_mean(t_rec):.4f}  site_rec={_mean(t_site_rec):.4f}  "
              f"aa_acc={_mean(t_aa_acc):.4f}  retention={_mean(t_ret):.4f}  "
              f"identity={_mean(t_id):.4f}"
              + (f"  model_EVEscape={_mean(tree_model_ev):.4f}" if evescape is not None else "")
              + (
                  f"  ag_EVEscape={_mean(tree_model_ev_ag):.4f}"
                  if (evescape is not None and pmc_mask is not None) else ""
              )
              + (f"  lit_frac={_mean(tree_pmc_gen):.4f}" if pmc_mask is not None else "")
              + (f"  any_desc={_mean(t_any_desc):.4f}" if t_any_desc else ""))

    # ── summary
    print("\n" + "=" * 64)
    print(f"AGGREGATE ({len(recs)} trees)  checkpoint={args.checkpoint}")
    print("=" * 64)
    print(f"  mutation recovery : {_mean(recs):.4f}  [PRIMARY: leaf best-match vs GT leaf]")
    print(f"  site recall       : {_mean(site_recs):.4f}  (= P(gen!=root | root!=GT))")
    print(f"  aa_acc | hit      : {_mean(aa_accs):.4f}  (= P(gen==GT | root!=GT & gen!=root))")
    print(f"  site precision    : {_mean(site_precs):.4f}  (= P(root!=GT | gen!=root))")
    print(f"  conserved retention: {_mean(rets):.4f}")
    print(f"  best-match identity: {_mean(idents):.4f}")
    print(f"  dist_to_root      : {_mean(dist_to_roots):.4f}  [mean Hamming(root, best-match gen)]")
    print(f"  gt_dist_to_root   : {_mean(gt_dist_to_roots):.4f}  [mean Hamming(root, GT leaf)]")
    print(f"  mut_recovery_any_descendant : {_mean(any_desc_recs):.4f}  [extra: any gen leaf]")
    print(f"  site_recall_any_descendant  : {_mean(any_desc_site):.4f}")
    print(f"  mut_site_recall_path_union  : {_mean(path_union_site_rec):.4f}  [extra: leaf unions]")
    summary = {"checkpoint": args.checkpoint, "n_trees": len(recs),
               "r0_backend": args.r0_backend,
               "fitness_beta": args.fitness_beta,
               "no_lit_hotspot_mask": bool(args.no_lit_hotspot_mask),
               "ablate_bridge": bool(args.ablate_bridge),
               "mut_recovery": _mean(recs), "cons_retention": _mean(rets),
               "site_recall": _mean(site_recs),
               "aa_acc_given_hit": _mean(aa_accs),
               "site_precision": _mean(site_precs),
               "identity": _mean(idents),
               "dist_to_root": _mean(dist_to_roots),
               "gt_dist_to_root": _mean(gt_dist_to_roots),
               "mut_recovery_any_descendant": _mean(any_desc_recs),
               "site_recall_any_descendant": _mean(any_desc_site),
               "mut_site_recall_path_union": _mean(path_union_site_rec),
               "metrics_scope": (
                   "PRIMARY mut_recovery/site_recall/aa_acc/cons_retention are "
                   "leaf-only (best-match gen leaf vs GT leaf vs root). "
                   "Internal nodes are not scored. "
                   "dist_to_root = mean Hamming(root, best-match gen leaf); "
                   "gt_dist_to_root = mean Hamming(root, GT leaf). "
                   "mut_recovery_any_descendant / mut_site_recall_path_union are "
                   "additional tree-wide companions; they do not replace primary."
               )}
    if evescape is not None:
        m_ev, g_ev = _mean(model_ev), _mean(gt_ev)
        m_frac = _mean(model_frac_rbd)
        g_frac = _mean(gt_frac_rbd)
        print(f"\n  EVEscape enrichment (simple mean of per-mut official scores; "
              f"NOT a strain-level product):")
        print(f"    scale: {ev_meta.get('score_scale', 'official_evescape_log_product')} "
              f"(official final EVEscape is log-scale, typically negative / not [0,1]; "
              f"NOT raw EVE)")
        print(f"    --- all scored muts (legacy companion) ---")
        print(f"    model mutations   : mean EVEscape = {m_ev:.4f}   ({len(model_ev)} scored muts, "
              f"{m_frac*100:.1f}% of muts in scored region)")
        print(f"    real GT mutations : mean EVEscape = {g_ev:.4f}   ({len(gt_ev)} scored muts, "
              f"{g_frac*100:.1f}% of muts in scored region)")
        print(f"    random scored baseline: mean EVEscape = {base_ev:.4f}")
        print(f"    model - random    : {m_ev - base_ev:+.4f}   (model vs GT: {m_ev - g_ev:+.4f})")
        summary.update({
            # Legacy keys (= mean over all scored muts).
            "model_evescape": m_ev,
            "gt_evescape": g_ev,
            "random_baseline_evescape": base_ev,
            # Explicit names so paper captions do not mix sets.
            "evescape_mean_all_scored_muts": m_ev,
            "gt_evescape_mean_all_scored_muts": g_ev,
            # Legacy key name (COVID RBD); keep for JSON compatibility.
            "model_frac_rbd": m_frac,
            "gt_frac_rbd": g_frac,
            # Clearer aliases (scored region = RBD for Spike, nearly full HA for flu_h1).
            "model_frac_scored": m_frac,
            "gt_frac_scored": g_frac,
            "n_model_evescape_scored_muts": len(model_ev),
            "n_gt_evescape_scored_muts": len(gt_ev),
            "evescape_meta": ev_meta,
            "evescape_definition": (
                "Mean (simple average, not product) of official EVEscape[col, leaf_aa] "
                "over root→leaf substitutions with nonzero tensor entry. "
                "model_evescape / evescape_mean_all_scored_muts = all scored muts; "
                "evescape_mean_antigenic_muts = same mean restricted to "
                "--lit-hotspot-mask sites (H1 Sa/Sb/Ca1/Ca2/Cb∪guidance; COVID PMC; "
                "H3 lit when used). Official EVEscape = "
                "log σ(z(fitness)) + log σ(z(accessibility)) + log σ(z(dissimilarity)); "
                "typically ≈ −2 to −4, not [0,1] and not raw EVE."
            ),
        })
        if pmc_mask is not None:
            m_ag, g_ag = _mean(model_ev_ag), _mean(gt_ev_ag)
            print(f"    --- antigenic / lit-hotspot muts (preferred flu KPI) ---")
            print(f"    model antigenic    : mean EVEscape = {m_ag:.4f}   "
                  f"({len(model_ev_ag)} scored antigenic muts; "
                  f"mean n_ag_muts/leaf={_mean(model_n_ag_muts):.2f})")
            print(f"    real GT antigenic  : mean EVEscape = {g_ag:.4f}   "
                  f"({len(gt_ev_ag)} scored antigenic muts; "
                  f"mean n_ag_muts/leaf={_mean(gt_n_ag_muts):.2f})")
            print(f"    random antigenic baseline: mean EVEscape = {base_ev_ag:.4f}")
            if m_ag == m_ag and base_ev_ag == base_ev_ag:
                print(f"    model - random_ag : {m_ag - base_ev_ag:+.4f}   "
                      f"(model vs GT: {m_ag - g_ag:+.4f})")
            summary.update({
                "evescape_mean_antigenic_muts": m_ag,
                "gt_evescape_mean_antigenic_muts": g_ag,
                "random_baseline_evescape_antigenic": base_ev_ag,
                "n_evescape_antigenic_muts_scored": len(model_ev_ag),
                "n_gt_evescape_antigenic_muts_scored": len(gt_ev_ag),
                "mean_n_antigenic_muts_per_leaf": _mean(model_n_ag_muts),
                "gt_mean_n_antigenic_muts_per_leaf": _mean(gt_n_ag_muts),
                "evescape_antigenic_definition": (
                    "Simple mean of official per-mutation EVEscape scores for "
                    "root→leaf AA changes at lit/antigenic mask sites only "
                    f"({lit_label}; n_sites={int(pmc_mask.sum())}). "
                    "Not a whole-sequence product/sum. Companion "
                    "evescape_mean_all_scored_muts averages over all scored muts."
                ),
            })
        else:
            print(
                "    NOTE: no --lit-hotspot-mask → "
                "evescape_mean_antigenic_muts not computed "
                "(pass H1/H3 lit or COVID PMC mask; never cross pathogens)."
            )

    if pmc_mask is not None:
        # Mean over leaves (pooled), equivalent to leaf-avg then tree-avg when
        # trees contribute similar leaf counts (same pattern as model_frac_rbd).
        pmc_gen = _mean(model_pmc_frac)
        pmc_gt = _mean(gt_pmc_frac)
        print(f"\n  {lit_label} hotspot mut fraction "
              f"(higher = more muts at curated lit sites):")
        print(f"    definition: avg over leaves of |mut_cols ∩ lit| / |mut_cols|")
        print(f"      mut_cols_gen = {{col : gen[col] != root[col]}}")
        print(f"      mut_cols_gt  = {{col : GT[col]  != root[col]}}")
        print(f"    lit_hotspot_mut_frac       (gen): {pmc_gen:.4f}  "
              f"({len(model_pmc_frac)} leaves)")
        print(f"    gt_lit_hotspot_mut_frac    (GT) : {pmc_gt:.4f}  "
              f"({len(gt_pmc_frac)} leaves)")
        print(f"    {lit_gen_key:24s} (gen): {pmc_gen:.4f}")
        print(f"    {lit_gt_key:24s} (GT) : {pmc_gt:.4f}")
        indexing_note = (
            "Flu HA: H3 numbering on mature HA1; col = h3_pos + signal_len - 1 "
            f"(signal_len={pmc_meta.get('signal_len', 16)} on full-ORF L=566)."
            if "flu" in lit_label
            else
            "COVID spike: 0-based Wuhan/P0DTC2 frame (spike_pos = col + 1 on "
            "full-length ungapped spike)."
        )
        summary.update({
            "lit_hotspot_mut_frac": pmc_gen,
            "gt_lit_hotspot_mut_frac": pmc_gt,
            lit_gen_key: pmc_gen,
            lit_gt_key: pmc_gt,
            "lit_hotspot_mask": pmc_meta,
            "lit_hotspot_n_sites": int(pmc_mask.sum()),
            "lit_hotspot_definition": (
                "Mean over leaves of (fraction of root→leaf mutation columns "
                "that lie in the curated lit hotspot mask). Gen uses gen≠root; "
                "GT uses GT≠root. " + indexing_note
            ),
            # Backward-compatible aliases when COVID PMC path is used
            **(
                {
                    "pmc_hotspot_mut_frac": pmc_gen,
                    "gt_pmc_hotspot_mut_frac": pmc_gt,
                    "pmc_hotspot_mask": pmc_meta,
                    "pmc_hotspot_n_sites": int(pmc_mask.sum()),
                }
                if lit_gen_key == "pmc_hotspot_mut_frac"
                else {}
            ),
        })

    if region_masks:
        pathogen = "covid_spike" if spike_masks else "flu_h3n2_ha"
        print(f"\n  Region metrics ({pathogen}):")
        print(f"    mut_frac     = avg_leaf |mut_cols ∩ R| / |mut_cols|")
        print(f"    site_recall  = avg_bestmatch P(gen≠root | root≠GT, col∈R)")
        print(f"    any_mut      = avg_leaf 1[≥1 mut in R]")
        region_summary: dict = {
            "pathogen": pathogen,
            "region_annotations": ann_paths,
            "mut_frac_definition": (
                "Mean over leaves of fraction of root→leaf mutation columns in "
                "each region. Gen: gen≠root; GT: GT≠root. Same leaf scope as "
                "pmc/flu_hotspot_mut_frac."
            ),
            "site_recall_definition": (
                "Mean over best-match (GT leaf → nearest gen leaf) pairs of "
                "P(gen≠root | root≠GT and col in region). Broader than lit-site "
                "hotspot frac; measures mutating-site identification inside R."
            ),
            "any_mut_definition": (
                "Mean over leaves of binary indicator that the leaf has ≥1 "
                "mutation inside the region (vs root)."
            ),
        }
        if spike_masks:
            region_summary["indexing"] = "1-based Wuhan/P0DTC2; col = spike_pos - 1"
            region_summary["bands_1based"] = {
                k: list(SPIKE_DOMAINS_1BASED[k]) for k in PRIMARY_DOMAIN_KEYS
            }
        else:
            lo, hi = HA_DOMAINS_H3_1BASED["HA_head"]
            region_summary["indexing"] = (
                f"1-based H3 mature; col = h3_pos + {H3_SIGNAL_LEN} - 1"
            )
            region_summary["bands_h3_1based"] = {
                k: list(HA_DOMAINS_H3_1BASED[k])
                for k in PRIMARY_HA_DOMAIN_KEYS
                if k in HA_DOMAINS_H3_1BASED
            }
            region_summary["rbs_sites_h3_1based"] = list(HA_RBS_SITES_H3_1BASED)
            region_summary["signal_len"] = H3_SIGNAL_LEN

        for name in mut_frac_keys:
            key = f"{_rk(name)}_mut_frac"
            g = _mean(domain_gen[name])
            t = _mean(domain_gt[name])
            if spike_masks:
                band = (
                    f"{SPIKE_DOMAINS_1BASED[name][0]}–"
                    f"{SPIKE_DOMAINS_1BASED[name][1]}"
                )
            elif name in HA_DOMAINS_H3_1BASED:
                band = (
                    f"H3#{HA_DOMAINS_H3_1BASED[name][0]}–"
                    f"{HA_DOMAINS_H3_1BASED[name][1]}"
                )
            else:
                band = f"H3 sites {list(HA_RBS_SITES_H3_1BASED)}"
            print(f"    {key:22s} (gen): {g:.4f}   gt_{key}: {t:.4f}  band={band}")
            summary[key] = g
            summary[f"gt_{key}"] = t
            region_summary[key] = g
            region_summary[f"gt_{key}"] = t

        for name in recall_keys:
            sk = f"{_rk(name)}_site_recall"
            ak = f"{_rk(name)}_any_mut"
            sr = _mean(region_site_rec[name])
            ag = _mean(region_any_gen[name])
            at = _mean(region_any_gt[name])
            print(f"    {sk:22s}       : {sr:.4f}")
            print(f"    {ak:22s} (gen): {ag:.4f}   gt_{ak}: {at:.4f}")
            summary[sk] = sr
            summary[ak] = ag
            summary[f"gt_{ak}"] = at
            region_summary[sk] = sr
            region_summary[ak] = ag
            region_summary[f"gt_{ak}"] = at

        if spike_masks:
            summary["spike_domain_metrics"] = region_summary
        else:
            summary["ha_region_metrics"] = region_summary
        summary["region_metrics"] = region_summary

    out = args.out or f"checkpoints/eval_enrichment_{Path(args.checkpoint).parent.name}.json"
    Path(out).write_text(json.dumps({"summary": summary, "per_tree": per_tree}, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
