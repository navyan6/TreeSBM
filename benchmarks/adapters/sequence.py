"""
Shared sequence adapters: evolve internal+terminal sequences along a *given*
topology from a fixed root sequence. Two variants, both used natively by the
CTMC/empirical/pLM baselines and as the shared adapter for topology-only methods
(ARTreeFormer/PhyloVAE) — clearly labeled at the call site.

  evolve_pyvolve  — Neutral / JTT / WAG / LG substitution (pyvolve, established impl)
  evolve_plm      — per-branch ESM mutation prior (this repo's ESM)

Both return {node_id: sequence} for every node in the topology (internal+leaf).
Requires pyvolve / torch respectively (cluster). Not needed by the pure metric
tests.
"""

from __future__ import annotations

import math
import random

from src.tree_state import TreeState
from benchmarks.metrics import trees as T

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_VOCAB)}


# ── pyvolve: neutral / empirical AA substitution ─────────────────────────────

def sanitize_aa_for_pyvolve(root_seq: str) -> str:
    """Map gaps / X / non-canonical residues to a standard AA so pyvolve accepts the root.

    ASR trees often carry ``X`` (ambiguous) or ``-``; pyvolve's 20-letter alphabet
    rejects those and aborts NeutralBD / ARTreeFormer sequence adapters.
    """
    counts: dict[str, int] = {}
    for ch in root_seq.upper():
        if ch in AA_TO_IDX:
            counts[ch] = counts.get(ch, 0) + 1
    fill = max(counts, key=counts.get) if counts else "A"
    return "".join(ch if ch in AA_TO_IDX else fill for ch in root_seq.upper())


def evolve_pyvolve(topology: TreeState, root_seq: str, model: str = "JTT",
                   seed: int = 0) -> dict[str, str]:
    """
    Evolve sequences down `topology` under an amino-acid model.
    model in {"neutral","JTT","WAG","LG"}. `neutral` = uniform freqs + equal
    exchangeabilities (a flat CTMC).
    """
    import pyvolve

    root_seq = sanitize_aa_for_pyvolve(root_seq)

    # pyvolve rejects the internal NODE_* labels our TreeState serializer emits
    # for adapted-topology rows; keep branch lengths but omit internal labels.
    newick = T.to_newick(topology, with_names=False)
    phylo = pyvolve.read_tree(tree=newick)
    if model.lower() == "neutral":
        import numpy as np
        # flat amino-acid generator: equal off-diagonal rates, uniform stationary.
        # pyvolve normalizes so branch length = expected substitutions/site.
        Q = np.full((20, 20), 1.0)
        np.fill_diagonal(Q, -19.0)
        # explicit state_freqs: matrix is symmetric so the stationary dist is
        # exactly uniform; without this pyvolve recomputes + prints a notice
        # and rewrites custom_matrix_frequencies.txt on every single call.
        freqs = np.full(20, 0.05)
        m = pyvolve.Model("custom", {"matrix": Q, "state_freqs": freqs})
    else:
        m = pyvolve.Model(model.upper())           # JTT / WAG / LG
    part = pyvolve.Partition(models=m, root_sequence=root_seq)
    ev = pyvolve.Evolver(tree=phylo, partitions=part)
    ev(seqfile=None, ratefile=None, infofile=None)
    # pyvolve keys sequences by the newick node labels (== TreeState node ids)
    return {k: v for k, v in ev.get_sequences(anc=True).items()}


# ── ESM per-branch mutation prior ────────────────────────────────────────────

def evolve_plm(topology: TreeState, root_seq: str, lm_logits_fn,
               subs_per_site_scale: float = 1.0, seed: int = 0) -> dict[str, str]:
    """
    Walk the topology root→leaves; along each edge (p,c) draw the number of
    substitutions ~ Poisson(branch_length * L * scale) and place them using the
    ESM per-position distribution of the current sequence. Not tree-aware.

    lm_logits_fn(seq) -> [L,20] tensor of log-probs (e.g. get_lm_logits wrapper).
    """
    import torch
    rng = random.Random(seed)
    cm = T.children_map(topology)
    seqs = {topology.root_id: root_seq}
    order, stack = [], [topology.root_id]
    while stack:
        n = stack.pop(); order.append(n); stack.extend(cm.get(n, []))
    for p in order:
        for c in cm.get(p, []):
            bl = float(topology.branch_lengths.get((p, c), 0.0))
            seq = list(seqs[p])
            L = len(seq)
            n_sub = rng.poisson(bl * L * subs_per_site_scale) \
                if hasattr(rng, "poisson") else _poisson(bl * L * subs_per_site_scale, rng)
            if n_sub > 0:
                logits = lm_logits_fn(seqs[p])                 # [L,20]
                probs = torch.softmax(logits, dim=-1)
                positions = rng.sample(range(L), min(n_sub, L))
                for pos in positions:
                    pv = probs[pos].clone()
                    ci = AA_TO_IDX.get(seq[pos], -1)
                    if ci >= 0:
                        pv[ci] = 0.0
                    tot = float(pv.sum())
                    if tot > 0:
                        seq[pos] = AA_VOCAB[int(torch.multinomial(pv / tot, 1).item())]
            seqs[c] = "".join(seq)
    return seqs


def _poisson(lmbda: float, rng: random.Random) -> int:
    # Knuth sampler (rng has no .poisson)
    L, k, p = math.exp(-lmbda), 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1
