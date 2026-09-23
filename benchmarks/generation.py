"""
K-sample generation harness (model-facing) for the benchmark tracks.

Thin wrapper around the proven `generate_tree` in scripts/eval_single_tree.py:
loads the checkpoint + ESM once, then samples K trees per root with distinct
seeds. Runs where a checkpoint + ESM are available (cluster / GPU), not in the
pure-metric unit tests.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from scripts.eval_single_tree import load_models, generate_tree, AA_VOCAB
from src.treeencoder.plm_embeddings import ESM2Embedder
from src.tree_state import TreeState

ESM_ID = "facebook/esm2_t6_8M_UR50D"


class TreeSBMGenerator:
    """Load a TreeSBM checkpoint once; sample K future trees from any root."""

    def __init__(
        self,
        checkpoint: str,
        max_seq_len: int = 566,
        device: str | None = None,
        r0_backend=None,
        fitness_beta: float | None = None,
        ablate_bridge: bool = False,
        gene_id: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_len = max_seq_len
        self.r0_backend = r0_backend
        self.fitness_beta = fitness_beta
        self.ablate_bridge = ablate_bridge
        self.gene_id = gene_id
        self.node_enc, self.tree_enc, self.rate_heads, self.col_entropy = load_models(
            checkpoint, self.device, max_seq_len
        )
        self.embedder = ESM2Embedder(device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(ESM_ID)
        self.esm_model = EsmForMaskedLM.from_pretrained(ESM_ID).to(self.device).eval()
        for p in self.esm_model.parameters():
            p.requires_grad = False
        self.aa_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(aa) for aa in AA_VOCAB], dtype=torch.long
        )

    def _resolved_col_entropy(self):
        """Panviral ckpts store gene_id→[L]; single-gene trains store a tensor."""
        ce = self.col_entropy
        if ce is None or not isinstance(ce, dict):
            return ce
        if self.gene_id and self.gene_id in ce:
            return ce[self.gene_id]
        if "default" in ce:
            return ce["default"]
        # Deterministic fallback so generation still runs; prefer matching gene_id.
        key = next(iter(ce))
        print(f"WARN: col_entropy dict has no key {self.gene_id!r}; using {key!r}")
        return ce[key]

    def generate_k(
        self,
        root_seq: str,
        K: int,
        n_steps: int = 50,
        max_leaves: int = 400,
        branch_rate_scale: float = 6.0,
        mutation_rate_scale: float = 0.04,
        base_seed: int = 0,
        cache_esm: bool = True,
    ) -> list[TreeState]:
        trees: list[TreeState] = []
        col_entropy = self._resolved_col_entropy()
        for k in range(K):
            random.seed(base_seed + k)
            torch.manual_seed(base_seed + k)
            trees.append(generate_tree(
                root_seq, n_steps, self.max_seq_len, branch_rate_scale, max_leaves,
                mutation_rate_scale, self.node_enc, self.tree_enc, self.rate_heads,
                self.embedder, self.tokenizer, self.esm_model, self.aa_token_ids, self.device,
                col_entropy=col_entropy,
                cache_esm=cache_esm,
                fitness_beta=self.fitness_beta,
                ablate_bridge=self.ablate_bridge,
                r0_backend=self.r0_backend,
            ))
        return trees


def gt_treestate(batch: dict) -> tuple[TreeState, str]:
    """Build the ground-truth TreeState (+ root sequence) from a TreeDataset item."""
    node_ids = batch["node_ids"]
    root_id = node_ids[batch["root_index"]]
    seqs = {n: batch["seqs"].get(n, "") for n in node_ids}
    has_children = {p for p, _ in batch["edges"]}
    gt = TreeState(
        node_ids=node_ids, root_id=root_id, edges=batch["edges"],
        branch_lengths=batch["branch_lengths"], node_seqs=seqs,
        active_leaves=[n for n in node_ids if n not in has_children],
    )
    return gt, seqs[root_id]
