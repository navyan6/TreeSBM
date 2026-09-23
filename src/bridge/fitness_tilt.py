"""Exponential fitness tilting of reference mutation rates (log R0)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from src.r0_backends import AA_TO_IDX, AA_VOCAB

SCORE_LOG_R0 = "log_R0"
SCORE_LOG_SOFTMAX = "log_softmax"
_VALID_SCORES = (SCORE_LOG_R0, SCORE_LOG_SOFTMAX)

TILT_SITE_LOCAL = "site_local"
TILT_FULL_ESM = "full_esm"
_VALID_MODES = (TILT_SITE_LOCAL, TILT_FULL_ESM)

# fitness_scorer: list[str] -> float tensor [len(seqs)] (higher = fitter)
FitnessScorer = Callable[[Sequence[str]], torch.Tensor]


def tilt_log_R0_by_fitness(
    log_R0: torch.Tensor,
    beta: float = 0.0,
    score: str = SCORE_LOG_R0,
    mode: str = TILT_SITE_LOCAL,
    sequences: Optional[Sequence[str]] = None,
    fitness_scorer: Optional[FitnessScorer] = None,
    cache: Optional[dict] = None,
    batch_size: int = 8,
    top_k_aas: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply exponential fitness tilt to ESM reference log-rates.

    Args:
        log_R0: [..., 20] or [N, L, 20] ESM mutation log-rates.
        beta: fitness temperature. 0 → return log_R0 unchanged.
        score: Option A fitness proxy (ignored for full_esm).
        mode: ``site_local`` (Option A) or ``full_esm`` (Option B).
        sequences: required for full_esm; length must match batch dim of log_R0.
        fitness_scorer: required for full_esm when beta ≠ 0.
        cache: optional mutable dict for mutant PLL memoization.
        batch_size: ESM mutant scoring batch size (full_esm).
        top_k_aas: if set, only score/tilt top-k AAs per site by untilted R0.

    Returns:
        log_R0_tilted with the same shape; last dim log-normalized when beta ≠ 0.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"fitness tilt mode must be one of {_VALID_MODES}, got {mode!r}")
    beta = float(beta)
    if beta == 0.0:
        return log_R0

    if mode == TILT_SITE_LOCAL:
        return _tilt_site_local(log_R0, beta=beta, score=score)

    return _tilt_full_esm(
        log_R0,
        sequences=sequences,
        fitness_scorer=fitness_scorer,
        beta=beta,
        cache=cache,
        batch_size=batch_size,
        top_k_aas=top_k_aas,
    )


def _tilt_site_local(
    log_R0: torch.Tensor,
    beta: float,
    score: str,
) -> torch.Tensor:
    if score not in _VALID_SCORES:
        raise ValueError(
            f"fitness score must be one of {_VALID_SCORES}, got {score!r}"
        )
    log_q = F.log_softmax(log_R0, dim=-1)
    if score == SCORE_LOG_SOFTMAX:
        score_a = log_q
    else:
        score_a = log_R0
    return F.log_softmax(log_q + beta * score_a, dim=-1)


def _tilt_full_esm(
    log_R0: torch.Tensor,
    sequences: Optional[Sequence[str]],
    fitness_scorer: Optional[FitnessScorer],
    beta: float,
    cache: Optional[dict],
    batch_size: int,
    top_k_aas: Optional[int],
) -> torch.Tensor:
    if sequences is None:
        raise ValueError("full_esm tilt requires sequences=...")
    if fitness_scorer is None:
        raise ValueError(
            "full_esm tilt requires fitness_scorer=... "
            "(use make_sequence_pll_scorer(r0_backend))"
        )
    if log_R0.dim() != 3:
        raise ValueError(
            f"full_esm expects log_R0 shaped [N, L, 20], got {tuple(log_R0.shape)}"
        )
    N, L, K = log_R0.shape
    if K != 20:
        raise ValueError(f"expected last dim 20, got {K}")
    if len(sequences) != N:
        raise ValueError(
            f"sequences length {len(sequences)} != log_R0 batch {N}"
        )

    if cache is None:
        cache = {}

    log_q = F.log_softmax(log_R0, dim=-1)
    fitness = torch.zeros_like(log_R0)

    # Collect unique mutants to score (batched).
    pending: list[tuple[str, int, int, str]] = []  # (key, n, pos, mutant_seq)
    keys_needed: list[str] = []

    for n, seq in enumerate(sequences):
        seq_L = min(len(seq), L)
        clean = seq[:seq_L]
        # Wild-type fitness (for ΔF option — we use absolute PLL as F).
        wt_key = f"WT::{clean}"
        if wt_key not in cache:
            keys_needed.append(wt_key)
            pending.append((wt_key, -1, -1, clean))

        # Which AAs to evaluate per site.
        if top_k_aas is not None and top_k_aas > 0 and top_k_aas < 20:
            # Top-k by untilted log_R0 (exclude current AA so we still tilt alternatives).
            scores_site = log_R0[n, :seq_L]  # [seq_L, 20]
            topk = min(int(top_k_aas), 20)
            _, top_idx = scores_site.topk(topk, dim=-1)  # [seq_L, topk]
            aa_lists = [set(int(x) for x in top_idx[i].tolist()) for i in range(seq_L)]
        else:
            aa_lists = [set(range(20)) for _ in range(seq_L)]

        for pos in range(seq_L):
            cur = clean[pos]
            cur_i = AA_TO_IDX.get(cur)
            for aa_i in aa_lists[pos]:
                aa = AA_VOCAB[aa_i]
                if cur_i is not None and aa_i == cur_i:
                    # Identity mutant: F = PLL(WT); fill after WT scored.
                    continue
                mut = clean[:pos] + aa + clean[pos + 1 :]
                key = f"{pos}:{aa}:{mut}"
                if key not in cache:
                    keys_needed.append(key)
                    pending.append((key, n, pos, mut))

    # Score uncached sequences in batches.
    _score_pending(pending, fitness_scorer, cache, batch_size)

    # Fill fitness tensor.
    for n, seq in enumerate(sequences):
        seq_L = min(len(seq), L)
        clean = seq[:seq_L]
        wt_key = f"WT::{clean}"
        f_wt = float(cache[wt_key])
        for pos in range(seq_L):
            cur = clean[pos]
            cur_i = AA_TO_IDX.get(cur)
            for aa_i in range(20):
                aa = AA_VOCAB[aa_i]
                if cur_i is not None and aa_i == cur_i:
                    fitness[n, pos, aa_i] = f_wt
                    continue
                mut = clean[:pos] + aa + clean[pos + 1 :]
                key = f"{pos}:{aa}:{mut}"
                if key in cache:
                    fitness[n, pos, aa_i] = float(cache[key])
                else:
                    # Not scored (outside top_k): use WT fitness → no relative tilt.
                    fitness[n, pos, aa_i] = f_wt

    return F.log_softmax(log_q + beta * fitness, dim=-1)


def _score_pending(
    pending: list[tuple[str, int, int, str]],
    fitness_scorer: FitnessScorer,
    cache: dict,
    batch_size: int,
) -> None:
    # Dedupe by key while preserving order.
    seen = set()
    uniq: list[tuple[str, str]] = []
    for key, _n, _pos, mut in pending:
        if key in seen or key in cache:
            continue
        seen.add(key)
        uniq.append((key, mut))

    bs = max(1, int(batch_size))
    for i in range(0, len(uniq), bs):
        chunk = uniq[i : i + bs]
        seqs = [m for _, m in chunk]
        scores = fitness_scorer(seqs)
        if not isinstance(scores, torch.Tensor):
            scores = torch.as_tensor(scores, dtype=torch.float32)
        scores = scores.detach().float().cpu().view(-1)
        if scores.numel() != len(chunk):
            raise RuntimeError(
                f"fitness_scorer returned {scores.numel()} scores for {len(chunk)} seqs"
            )
        for (key, _), s in zip(chunk, scores.tolist()):
            cache[key] = float(s)


def make_sequence_pll_scorer(
    r0_backend,
    max_seq_len: Optional[int] = None,
) -> FitnessScorer:
    """
    Build a FitnessScorer from an R0Backend.

    Scores each sequence as mean per-position log-prob of the observed AA under
    one unmasked ESM (or substitution) forward — the same one-pass PLL proxy used
    by generation fitness gating. True masked PLL (L forwards/seq) is not used.
    """

    def _score(sequences: Sequence[str]) -> torch.Tensor:
        if not sequences:
            return torch.zeros(0, dtype=torch.float32)
        L = max_seq_len or max(len(s) for s in sequences)
        log_R0 = r0_backend.log_mutation_rates(list(sequences), max_seq_len=L)
        # log_R0: [N, L, 20]
        out = []
        for i, seq in enumerate(sequences):
            vals = []
            for pos, aa in enumerate(seq[:L]):
                j = AA_TO_IDX.get(aa)
                if j is None:
                    continue
                vals.append(float(log_R0[i, pos, j].item()))
            out.append(sum(vals) / len(vals) if vals else float("-inf"))
        return torch.tensor(out, dtype=torch.float32)

    return _score


def make_fake_pll_scorer(base_log_R0: torch.Tensor) -> FitnessScorer:
    """
    Unit-test helper: score a mutant by mean gather under a fixed [L,20] log_R0
    (site-local scores — NOT true full-context PLL). Useful to test wiring without ESM.
    """
    if base_log_R0.dim() != 2 or base_log_R0.size(-1) != 20:
        raise ValueError("base_log_R0 must be [L, 20]")
    L = base_log_R0.size(0)

    def _score(sequences: Sequence[str]) -> torch.Tensor:
        out = []
        for seq in sequences:
            vals = []
            for pos, aa in enumerate(seq[:L]):
                j = AA_TO_IDX.get(aa)
                if j is None:
                    continue
                vals.append(float(base_log_R0[pos, j].item()))
            out.append(sum(vals) / len(vals) if vals else float("-inf"))
        return torch.tensor(out, dtype=torch.float32)

    return _score
