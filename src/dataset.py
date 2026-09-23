"""
PyTorch Dataset over processed phylogenetic tree groups.

Each item is one tree: loads rooted NWK + ancestral AA FASTA + branch_lengths JSON
and returns everything needed for one forward pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler
from Bio import Phylo, SeqIO

from src.tree_state import TreeState
from src.treeencoder.structural_features import compute_structural_features
from src.treeencoder.laplacian import compute_laplacian_pe
from src.treeencoder.edges import build_edges
from src.r0_backends import ref_rates_filename

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
PAD_IDX = len(AA_VOCAB)  # 20


def aa_seq_to_tensor(seq: str, length: int) -> torch.Tensor:
    """Convert AA string to [length] int tensor. Unknown AAs to PAD_IDX."""
    t = torch.full((length,), PAD_IDX, dtype=torch.long)
    for i, aa in enumerate(seq[:length]):
        t[i] = AA_TO_IDX.get(aa, PAD_IDX)
    return t


def _seq_has_aa(seq: str | None) -> bool:
    """True if seq contains at least one standard amino acid (not empty/all-gap)."""
    if not seq:
        return False
    return any(c in AA_TO_IDX for c in seq)


def fill_missing_node_seqs(
    root_id: str,
    edges: list[tuple[str, str]],
    seqs: dict[str, str],
) -> dict[str, str]:
    """
    Fill nodes missing from ``seqs`` (or all-gap) by copying the parent sequence
    in BFS order from the root. Used by PLM precompute and dataset loading.
    """
    out = dict(seqs)
    for alias in ("NODE_ROOT", "ROOT"):
        if alias in out and not _seq_has_aa(out.get(root_id)):
            out[root_id] = out[alias]
    if not _seq_has_aa(out.get(root_id)) and "NODE_0000000" in out and _seq_has_aa(
        out["NODE_0000000"]
    ):
        out[root_id] = out["NODE_0000000"]
    if not _seq_has_aa(out.get(root_id)):
        for parent, child in edges:
            if parent == root_id and _seq_has_aa(out.get(child)):
                out[root_id] = out[child]
                break
    if not _seq_has_aa(out.get(root_id)):
        raise ValueError(f"root {root_id!r} has no usable AA sequence")

    children: dict[str, list[str]] = {}
    for p, c in edges:
        children.setdefault(p, []).append(c)

    queue = [root_id]
    while queue:
        parent = queue.pop(0)
        p_seq = out[parent]
        for child in children.get(parent, []):
            if not _seq_has_aa(out.get(child)):
                out[child] = p_seq
            queue.append(child)
    return out


def _pad_or_truncate_ref_rates(
    log_ref: torch.Tensor, max_seq_len: int
) -> torch.Tensor:
    """Align cached R0 [N, L_cache, 20] to training max_seq_len."""
    L = log_ref.shape[1]
    if L == max_seq_len:
        return log_ref
    if L > max_seq_len:
        return log_ref[:, :max_seq_len, :].contiguous()
    pad = log_ref.new_zeros(log_ref.shape[0], max_seq_len - L, log_ref.shape[2])
    return torch.cat([log_ref, pad], dim=1)


def parse_newick(nwk_path: str):
    tree = Phylo.read(nwk_path, "newick")
    edges, branch_lengths = [], {}
    _c = [0]

    def name(clade):
        if clade.name:
            return clade.name
        n = f"NODE_{_c[0]:07d}"
        clade.name = n
        _c[0] += 1
        return n

    def walk(parent):
        pn = name(parent)
        for child in parent.clades:
            cn = name(child)
            bl = child.branch_length or 0.0
            edges.append((pn, cn))
            branch_lengths[(pn, cn)] = bl
            walk(child)

    walk(tree.root)
    root_id = name(tree.root)
    node_ids = [root_id]
    visited = {root_id}
    children = {}
    for p, c in edges:
        children.setdefault(p, []).append(c)
    queue = [root_id]
    while queue:
        curr = queue.pop(0)
        for ch in children.get(curr, []):
            if ch not in visited:
                visited.add(ch)
                node_ids.append(ch)
                queue.append(ch)

    return root_id, node_ids, edges, branch_lengths


def _infer_gene_id(data_dir: Path, group: int, default: str = "default") -> str:
    """Best-effort gene/subtype tag from group_*_meta.csv or *group_*.csv.

    Prefers source CSVs (e.g. pflutrain_group_XXX.csv) when meta was rewritten
    without subtype columns by run_all_groups.py.
    """
    import csv

    keys = ("gene_id", "subtype", "gene", "protein")
    candidates: list[Path] = []
    # Prefer prep CSVs that still carry subtype/season tags.
    candidates.extend(sorted(data_dir.glob(f"*group_{group:03d}.csv")))
    meta = data_dir / f"group_{group:03d}_meta.csv"
    if meta.exists():
        candidates.append(meta)
    seen: set[Path] = set()
    for csv_path in candidates:
        if csv_path in seen or not csv_path.is_file():
            continue
        seen.add(csv_path)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        for key in keys:
            val = (rows[0].get(key) or "").strip()
            if val:
                return val.lower()
    return default


class TreeDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        laplacian_dim: int = 8,
        max_seq_len: int = 566,
        ref_rates_tag: str = "",
        gene_id: str | None = None,
        infer_gene_id: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.laplacian_dim = laplacian_dim
        self.max_seq_len = max_seq_len
        # Filename tag for multi-pLM R0 caches: group_XXX_ref_rates{tag}.pt
        # Empty tag → legacy group_XXX_ref_rates.pt (ESM-2-8M default).
        self.ref_rates_tag = ref_rates_tag or ""
        self.gene_id = gene_id
        self.infer_gene_id = bool(infer_gene_id)

        # Find all complete groups (need all 3 files)
        self.groups = sorted([
            int(p.stem.split("_")[1])
            for p in self.data_dir.glob("group_*_rooted.nwk")
            if (self.data_dir / p.name.replace("_rooted.nwk", "_anc_aa.fasta")).exists()
            and (self.data_dir / p.name.replace("_rooted.nwk", "_bl.json")).exists()
        ])
        tag = f" gene_id={gene_id}" if gene_id else ""
        print(f"TreeDataset: {len(self.groups)} trees found{tag}")

    def __len__(self):
        return len(self.groups)

    def gene_id_at(self, idx: int) -> str:
        g = self.groups[idx]
        if self.gene_id is not None:
            return self.gene_id
        if self.infer_gene_id:
            return _infer_gene_id(self.data_dir, g, default="default")
        return "default"

    def __getitem__(self, idx: int) -> dict:
        g = self.groups[idx]
        d = self.data_dir

        root_id, node_ids, edges, branch_lengths = parse_newick(
            str(d / f"group_{g:03d}_rooted.nwk")
        )

        seqs = {
            rec.id: str(rec.seq)
            for rec in SeqIO.parse(d / f"group_{g:03d}_anc_aa.fasta", "fasta")
        }
        seqs = fill_missing_node_seqs(root_id, edges, seqs)

        with open(d / f"group_{g:03d}_bl.json") as f:
            node_data = json.load(f)["nodes"]
        node_times = {
            nid: node_data.get(nid, {}).get("numdate", 0.0) for nid in node_ids
        }

        has_children = {p for p, _ in edges}
        active_leaves = [nid for nid in node_ids if nid not in has_children]

        tree_state = TreeState(
            node_ids=node_ids, root_id=root_id, edges=edges,
            branch_lengths=branch_lengths, node_seqs=seqs,
            active_leaves=active_leaves,
        )

        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        structural = compute_structural_features(tree_state, node_to_idx)
        lap_pe = compute_laplacian_pe(tree_state, node_to_idx, self.laplacian_dim)
        edge_index, edge_type, edge_attr = build_edges(tree_state, node_to_idx)

        targets = torch.stack([
            aa_seq_to_tensor(seqs[nid], self.max_seq_len) for nid in node_ids
        ])

        leaf_indices = [node_to_idx[nid] for nid in active_leaves]

        plm_path = d / f"group_{g:03d}_plm.pt"
        if plm_path.exists():
            cached = torch.load(plm_path, weights_only=True)
            plm_embeddings = cached["plm"]  # [N, 320], BFS-ordered to match node_ids
        else:
            plm_embeddings = None

        ref_path = d / ref_rates_filename(g, self.ref_rates_tag)
        if ref_path.exists():
            log_ref_mut_rates = _pad_or_truncate_ref_rates(
                torch.load(ref_path, weights_only=True)["log_mut_rates"],
                self.max_seq_len,
            )
        else:
            log_ref_mut_rates = None

        if self.gene_id is not None:
            gid = self.gene_id
        elif self.infer_gene_id:
            gid = _infer_gene_id(d, g, default="default")
        else:
            gid = "default"

        return {
            "group": g,
            "gene_id": gid,
            "node_ids": node_ids,
            "node_times": torch.tensor([node_times[nid] for nid in node_ids], dtype=torch.float32),
            "structural_features": structural,        # [N, 3]
            "lap_pe": lap_pe,                         # [N, lap_dim]
            "edge_index": edge_index,                 # [2, 2E]
            "edge_attr": edge_attr,                   # [2E, 1]
            "targets": targets,                       # [N, max_seq_len]
            "leaf_indices": leaf_indices,
            "root_index": node_to_idx[root_id],
            "seqs": seqs,                             # for ESM2 embedding
            # raw graph topology (needed by SampleBridgeState)
            "edges": edges,
            "branch_lengths": branch_lengths,
            # precomputed ESM2 [N, 320] or None if not yet cached
            "plm_embeddings": plm_embeddings,
            # precomputed log R0 mutation rates [N, 566, 20] or None
            "log_ref_mut_rates": log_ref_mut_rates,
        }


class ConcatGeneTreeDataset(Dataset):
    """Concatenate TreeDatasets and expose per-index gene_id for balanced sampling."""

    def __init__(self, datasets: list[TreeDataset]):
        if not datasets:
            raise ValueError("ConcatGeneTreeDataset needs ≥1 TreeDataset")
        self.datasets = list(datasets)
        self._cum: list[int] = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self._cum.append(total)
        self.gene_ids = []
        for ds in self.datasets:
            if hasattr(ds, "gene_id_at"):
                self.gene_ids.extend(ds.gene_id_at(i) for i in range(len(ds)))
            else:
                gid = getattr(ds, "gene_id", None) or "default"
                self.gene_ids.extend([gid] * len(ds))
        print(
            f"ConcatGeneTreeDataset: {len(self)} trees across "
            f"{len(self.datasets)} sources "
            f"({sorted(set(self.gene_ids))})"
        )

    def __len__(self) -> int:
        return self._cum[-1] if self._cum else 0

    def _locate(self, idx: int) -> tuple[TreeDataset, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        for ds, end in zip(self.datasets, self._cum):
            start = end - len(ds)
            if idx < end:
                return ds, idx - start
        raise IndexError(idx)

    def gene_id_at(self, idx: int) -> str:
        return self.gene_ids[idx]

    def __getitem__(self, idx: int) -> dict:
        ds, local = self._locate(idx)
        return ds[local]


class BalancedGeneSampler(Sampler[int]):
    """Each step: sample a gene uniformly, then a tree of that gene (with replacement)."""

    def __init__(self, gene_ids: list[str], num_samples: int | None = None, seed: int = 0):
        from collections import defaultdict

        self.by_gene: dict[str, list[int]] = defaultdict(list)
        for i, g in enumerate(gene_ids):
            self.by_gene[g].append(i)
        self.genes = sorted(self.by_gene)
        if not self.genes:
            raise ValueError("BalancedGeneSampler: empty gene_ids")
        self.num_samples = int(num_samples) if num_samples is not None else len(gene_ids)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        import random as _random

        rng = _random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_samples):
            gene = rng.choice(self.genes)
            yield rng.choice(self.by_gene[gene])
