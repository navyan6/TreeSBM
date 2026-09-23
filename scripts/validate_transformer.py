#!/usr/bin/env python3
"""Linear-probe suite for the node encoder / graph transformer.

Trains simple probes on frozen embeddings and writes CSV/JSON metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.dataset import TreeDataset
from src.tree_state import TreeState
from src.treeencoder.edges import build_edges
from src.treeencoder.laplacian import compute_laplacian_pe
from src.treeencoder.node_encoder import NodeEncoder
from src.treeencoder.plm_embeddings import ESM2Embedder
from src.treeencoder.structural_features import compute_structural_features
from src.networks import TreeEncoder


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}


@dataclass
class TreeRecord:
    tree_id: int
    group: int
    tree: TreeState
    node_ids: list[str]
    root_id: str
    seqs: dict[str, str]
    node_times: dict[str, float]
    depths: dict[str, int]
    root_dist: dict[str, float]
    subtree_size: dict[str, int]
    num_children: dict[str, int]
    parent_map: dict[str, str]
    edge_set: set[tuple[str, str]]
    plm: torch.Tensor
    structural: torch.Tensor
    lap_pe: torch.Tensor
    branch_len_scalar: torch.Tensor
    topological_only: torch.Tensor
    esm_branch: torch.Tensor
    node_encoder: torch.Tensor | None = None
    graph_transformer: torch.Tensor | None = None
    graph_random: torch.Tensor | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bfs_depths(tree: TreeState) -> dict[str, int]:
    depths = {tree.root_id: 0}
    stack = [tree.root_id]
    while stack:
        node = stack.pop()
        for child in tree.get_children(node):
            depths[child] = depths[node] + 1
            stack.append(child)
    return depths


def children_map(tree: TreeState) -> dict[str, list[str]]:
    cm: dict[str, list[str]] = {}
    for parent, child in tree.edges:
        cm.setdefault(parent, []).append(child)
    return cm


def parent_map(tree: TreeState) -> dict[str, str]:
    return {child: parent for parent, child in tree.edges}


def subtree_sizes(tree: TreeState) -> dict[str, int]:
    cm = children_map(tree)
    leaves = set(n for n in tree.node_ids if n not in cm)
    out: dict[str, int] = {}
    order = []
    stack = [tree.root_id]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(cm.get(node, []))
    for node in reversed(order):
        if node in leaves:
            out[node] = 1
        else:
            out[node] = sum(out[ch] for ch in cm.get(node, []))
    return out


def root_distances(tree: TreeState) -> dict[str, float]:
    cm = children_map(tree)
    out = {tree.root_id: 0.0}
    stack = [tree.root_id]
    while stack:
        node = stack.pop()
        for child in cm.get(node, []):
            out[child] = out[node] + float(tree.branch_lengths.get((node, child), 0.0))
            stack.append(child)
    return out


def parent_branch_lengths(tree: TreeState) -> dict[str, float]:
    pm = parent_map(tree)
    out = {tree.root_id: 0.0}
    for node in tree.node_ids:
        if node == tree.root_id:
            continue
        out[node] = float(tree.branch_lengths.get((pm[node], node), 0.0))
    return out


def pairwise_cosine_stats(x: torch.Tensor, max_pairs: int = 5000) -> dict[str, float]:
    if x.shape[0] < 2:
        return {"cos_mean": float("nan"), "cos_std": float("nan"), "cos_min": float("nan"), "cos_max": float("nan")}
    n = x.shape[0]
    rng = torch.Generator().manual_seed(0)
    num = min(max_pairs, n * (n - 1) // 2)
    idx1 = torch.randint(0, n, (num,), generator=rng)
    idx2 = torch.randint(0, n, (num,), generator=rng)
    mask = idx1 != idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]
    if idx1.numel() == 0:
        return {"cos_mean": float("nan"), "cos_std": float("nan"), "cos_min": float("nan"), "cos_max": float("nan")}
    x1 = x[idx1]
    x2 = x[idx2]
    cos = torch.nn.functional.cosine_similarity(x1, x2, dim=-1)
    return {
        "cos_mean": float(cos.mean().item()),
        "cos_std": float(cos.std(unbiased=False).item()),
        "cos_min": float(cos.min().item()),
        "cos_max": float(cos.max().item()),
    }


def embedding_health(x: torch.Tensor, max_svd_rows: int = 5000) -> dict[str, float]:
    x = x.detach().float()
    if x.shape[0] == 0:
        return {}
    if x.shape[0] > max_svd_rows:
        idx = torch.randperm(x.shape[0])[:max_svd_rows]
        x = x[idx]
    centered = x - x.mean(dim=0, keepdim=True)
    norms = x.norm(dim=1)
    out = {
        "n": float(x.shape[0]),
        "d": float(x.shape[1]),
        "var_mean": float(x.var(dim=0, unbiased=False).mean().item()),
        "var_min": float(x.var(dim=0, unbiased=False).min().item()),
        "var_max": float(x.var(dim=0, unbiased=False).max().item()),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std(unbiased=False).item()),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
    }
    out.update(pairwise_cosine_stats(x))
    svals = torch.linalg.svdvals(centered.cpu())
    top = svals[: min(10, svals.numel())]
    out["sv1"] = float(top[0].item()) if top.numel() else float("nan")
    out["sv10"] = float(top[-1].item()) if top.numel() >= 10 else float("nan")
    out["sv_ratio_1_10"] = float((top[0] / top[-1]).item()) if top.numel() >= 10 and top[-1] > 0 else float("nan")
    out["rank_gt_1e-3"] = float((svals > 1e-3 * svals.max()).sum().item()) if svals.numel() else 0.0
    return out


def auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    ranks = rankdata(y_score)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    return float((2 * tp / denom) if denom else 0.0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    vals = []
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        denom = 2 * tp + fp + fn
        vals.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(vals))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, with_spearman: bool = False) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    out = {"mae": mae, "r2": r2}
    if with_spearman:
        if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            rho, _ = spearmanr(y_true, y_pred)
            out["spearman"] = float(rho)
        else:
            out["spearman"] = float("nan")
    return out


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> dict[str, float]:
    if y_score.ndim == 1:
        return {
            "auroc": auc_binary(y_true, y_score),
            "f1": f1_binary(y_true, y_pred),
            "accuracy": accuracy(y_true, y_pred),
        }
    if num_classes is None:
        num_classes = y_score.shape[1]
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
    }


def standardize(train_x: torch.Tensor, *xs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return tuple((x - mean) / std for x in xs)


def remap_multiclass_labels(*ys: torch.Tensor) -> tuple[list[torch.Tensor], dict[int, int]]:
    flat = torch.cat([y.reshape(-1).long().cpu() for y in ys], dim=0)
    uniq = sorted(int(v) for v in flat.unique().tolist())
    mapping = {old: new for new, old in enumerate(uniq)}
    remapped: list[torch.Tensor] = []
    for y in ys:
        vals = [mapping[int(v)] for v in y.reshape(-1).long().cpu().tolist()]
        remapped.append(torch.tensor(vals, dtype=torch.long))
    return remapped, mapping


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def fit_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    task: str,
    num_classes: int | None = None,
    max_epochs: int = 300,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    patience: int = 20,
) -> LinearProbe:
    if task == "regression":
        out_dim = 1
        criterion = nn.MSELoss()
        best_key = "loss"
    elif task == "binary":
        out_dim = 1
        criterion = nn.BCEWithLogitsLoss()
        best_key = "loss"
    elif task == "multiclass":
        if num_classes is None:
            raise ValueError("num_classes required for multiclass")
        out_dim = num_classes
        criterion = nn.CrossEntropyLoss()
        best_key = "loss"
    else:
        raise ValueError(f"unknown task: {task}")

    model = LinearProbe(x_train.shape[1], out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if task == "regression":
        y_train = y_train.float().view(-1, 1)
        y_val = y_val.float().view(-1, 1)
    elif task == "binary":
        y_train = y_train.float().view(-1, 1)
        y_val = y_val.float().view(-1, 1)
    else:
        # Labels arrive already remapped to a contiguous 0..K-1 space by the
        # caller (probe_and_score), jointly over train/val/test, with
        # num_classes=K matching. Do NOT remap again here: train+val alone may
        # not contain every class, so a second remap would compact them and
        # desync from the test labels and the head width.
        y_train = y_train.long().view(-1)
        y_val = y_val.long().view(-1)

    best_state = None
    best_val = float("inf")
    wait = 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_train)
        if task == "regression":
            loss = criterion(logits, y_train)
        elif task == "binary":
            loss = criterion(logits, y_train)
        else:
            loss = criterion(logits, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(x_val)
            if task == "regression":
                v_loss = criterion(v_logits, y_val).item()
            elif task == "binary":
                v_loss = criterion(v_logits, y_val).item()
            else:
                v_loss = criterion(v_logits, y_val).item()
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def tree_depth_features(tree: TreeState) -> tuple[dict[str, int], dict[str, float], dict[str, int], dict[str, int]]:
    depths = bfs_depths(tree)
    cm = children_map(tree)
    leaves = set(n for n in tree.node_ids if n not in cm)
    subtree = subtree_sizes(tree)
    num_children = {n: len(cm.get(n, [])) for n in tree.node_ids}
    root_dist = root_distances(tree)
    return depths, root_dist, subtree, num_children


def mutation_record(node: str, parent: str, seqs: dict[str, str]) -> tuple[int | None, int | None, int | None]:
    child = seqs[node]
    par = seqs[parent]
    diffs = [i for i, (a, b) in enumerate(zip(par, child)) if a != b]
    if len(diffs) != 1:
        return None, None, None
    pos = diffs[0]
    return pos, AA_TO_IDX.get(par[pos], None), AA_TO_IDX.get(child[pos], None)


def collect_tree_records(
    dataset: TreeDataset,
    indices: list[int],
    device: str,
    max_seq_len: int,
    lap_dim: int,
    checkpoint: str | None,
    use_cache_plm: bool = True,
) -> list[TreeRecord]:
    embedder = ESM2Embedder(device=device) if not use_cache_plm else None

    actual_node_enc = None
    actual_tree_enc = None
    random_node_enc = None
    random_tree_enc = None
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        actual_node_enc = NodeEncoder(d_plm=320, d_struct=3, d_laplacian=lap_dim, d_node=128).to(device)
        actual_tree_enc = TreeEncoder(d_model=128, n_layers=4, n_heads=8, dropout=0.1).to(device)
        actual_node_enc.load_state_dict(ckpt["node_enc"])
        actual_tree_enc.load_state_dict(ckpt["tree_enc"])
        actual_node_enc.eval()
        actual_tree_enc.eval()
        for m in [actual_node_enc, actual_tree_enc]:
            for p in m.parameters():
                p.requires_grad = False

    random_node_enc = NodeEncoder(d_plm=320, d_struct=3, d_laplacian=lap_dim, d_node=128).to(device)
    random_tree_enc = TreeEncoder(d_model=128, n_layers=4, n_heads=8, dropout=0.1).to(device)
    random_node_enc.eval()
    random_tree_enc.eval()
    for m in [random_node_enc, random_tree_enc]:
        for p in m.parameters():
            p.requires_grad = False

    records: list[TreeRecord] = []
    for tree_pos, idx in enumerate(indices):
        batch = dataset[idx]
        node_ids = batch["node_ids"]
        root_id = node_ids[batch["root_index"]]
        seqs = batch["seqs"]
        node_times = {nid: float(batch["node_times"][i].item()) for i, nid in enumerate(node_ids)}
        tree = TreeState(
            node_ids=node_ids,
            root_id=root_id,
            edges=batch["edges"],
            branch_lengths=batch["branch_lengths"],
            node_seqs=seqs,
            active_leaves=[nid for nid in node_ids if nid not in {p for p, _ in batch["edges"]}],
        )
        depths, root_dist, subtree, num_children = tree_depth_features(tree)
        parent = parent_map(tree)
        edge_set = set(tree.edges)

        if batch.get("plm_embeddings") is not None:
            plm = batch["plm_embeddings"].to(device)
        elif embedder is not None:
            plm = embedder.embed_sequences([seqs[nid] for nid in node_ids]).to(device)
        else:
            raise RuntimeError("No PLM embeddings cached and no embedder available.")

        n2i = {nid: i for i, nid in enumerate(node_ids)}
        structural = compute_structural_features(tree, n2i).to(device)
        lap_pe = compute_laplacian_pe(tree, n2i, lap_dim, device=device)
        edge_index, _, edge_attr = build_edges(tree, n2i)
        edge_index = edge_index.to(device)
        branch_attr = edge_attr.squeeze(-1).to(device)

        branch_len = torch.zeros(len(node_ids), 1, device=device)
        root_d = torch.zeros(len(node_ids), 1, device=device)
        for i, nid in enumerate(node_ids):
            branch_len[i, 0] = float(parent_branch_lengths(tree)[nid])
            root_d[i, 0] = float(root_dist[nid])
        topo_only = torch.cat([structural, lap_pe], dim=-1)
        esm_branch = torch.cat([plm, structural, lap_pe, branch_len, root_d], dim=-1)

        node_encoder = None
        graph_transformer = None
        graph_random = None

        with torch.no_grad():
            if actual_node_enc is not None:
                node_encoder = actual_node_enc(plm, structural, lap_pe).detach().cpu()
                graph_transformer = actual_tree_enc(
                    node_encoder.to(device), node_ids, node_times, edge_index, branch_attr, t_scalar=1.0
                )[0].detach().cpu()
            random_node = random_node_enc(plm, structural, lap_pe).detach()
            graph_random = random_tree_enc(
                random_node.to(device), node_ids, node_times, edge_index, branch_attr, t_scalar=1.0
            )[0].detach().cpu()

        records.append(
            TreeRecord(
                tree_id=tree_pos,
                group=int(batch["group"]),
                tree=tree,
                node_ids=node_ids,
                root_id=root_id,
                seqs=seqs,
                node_times=node_times,
                depths=depths,
                root_dist=root_dist,
                subtree_size=subtree,
                num_children=num_children,
                parent_map=parent,
                edge_set=edge_set,
                plm=plm.detach().cpu(),
                structural=structural.detach().cpu(),
                lap_pe=lap_pe.detach().cpu(),
                branch_len_scalar=branch_len.detach().cpu(),
                topological_only=topo_only.detach().cpu(),
                esm_branch=esm_branch.detach().cpu(),
                node_encoder=node_encoder,
                graph_transformer=graph_transformer,
                graph_random=graph_random,
            )
        )
    return records


def split_records(records: list[TreeRecord], seed: int, train_frac: float, val_frac: float) -> tuple[list[TreeRecord], list[TreeRecord], list[TreeRecord]]:
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    n = len(idxs)
    if n == 0:
        return [], [], []
    if n == 1:
        return [records[idxs[0]]], [records[idxs[0]]], [records[idxs[0]]]
    if n == 2:
        return [records[idxs[0]]], [records[idxs[1]]], [records[idxs[1]]]
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n - n_val - n_test)
    train = [records[i] for i in idxs[:n_train]]
    val = [records[i] for i in idxs[n_train:n_train + n_val]]
    test = [records[i] for i in idxs[n_train + n_val:n_train + n_val + n_test]]
    return train, val, test


def flatten_node_samples(records: list[TreeRecord], source: str, max_seq_len: int) -> dict[str, torch.Tensor]:
    xs = []
    ys_leaf = []
    ys_depth = []
    ys_rootdist = []
    ys_subtree = []
    ys_numchildren = []
    ys_parent_hamming = []
    ys_root_hamming = []
    ys_mutpos = []
    ys_mutaa = []
    ys_edge_parent = []
    ys_edge_child = []
    ys_edge_label = []
    ys_single_node_idx = []

    for rec in records:
        emb = getattr(rec, source)
        if emb is None:
            continue
        for i, nid in enumerate(rec.node_ids):
            xs.append(emb[i].float())
            ys_leaf.append(float(nid not in {p for p, _ in rec.tree.edges}))
            ys_depth.append(float(rec.depths[nid]))
            ys_rootdist.append(float(rec.root_dist[nid]))
            ys_subtree.append(float(rec.subtree_size[nid]))
            ys_numchildren.append(float(rec.num_children[nid]))
            if nid == rec.root_id:
                ys_parent_hamming.append(float("nan"))
                ys_mutpos.append(-1)
                ys_mutaa.append(-1)
                ys_root_hamming.append(0.0)
            else:
                parent = rec.parent_map[nid]
                child_seq = rec.seqs[nid]
                parent_seq = rec.seqs[parent]
                ys_parent_hamming.append(float(sum(a != b for a, b in zip(child_seq, parent_seq))))
                ys_root_hamming.append(float(sum(a != b for a, b in zip(child_seq, rec.seqs[rec.root_id]))))
                pos, anc_idx, mut_idx = mutation_record(nid, parent, rec.seqs)
                ys_mutpos.append(pos if pos is not None else -1)
                ys_mutaa.append(anc_idx if anc_idx is not None else -1)

    return {
        "x": torch.stack(xs) if xs else torch.empty(0),
        "leaf": torch.tensor(ys_leaf, dtype=torch.float32),
        "depth": torch.tensor(ys_depth, dtype=torch.float32),
        "rootdist": torch.tensor(ys_rootdist, dtype=torch.float32),
        "subtree": torch.tensor(ys_subtree, dtype=torch.float32),
        "numchildren": torch.tensor(ys_numchildren, dtype=torch.long),
        "parent_hamming": torch.tensor(ys_parent_hamming, dtype=torch.float32),
        "root_hamming": torch.tensor(ys_root_hamming, dtype=torch.float32),
        "mutpos": torch.tensor(ys_mutpos, dtype=torch.long),
        "mutaa": torch.tensor(ys_mutaa, dtype=torch.long),
    }


def build_pair_samples(records: list[TreeRecord], source: str, max_negative_per_tree: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    feats = []
    labels = []
    rng = random.Random(0)
    for rec in records:
        emb = getattr(rec, source)
        if emb is None:
            continue
        node_to_idx = {nid: i for i, nid in enumerate(rec.node_ids)}
        positives = list(rec.edge_set)
        pos_set = set(positives)
        node_ids = rec.node_ids
        all_pairs = [(p, c) for p in node_ids for c in node_ids if p != c and (p, c) not in pos_set]
        if max_negative_per_tree and len(all_pairs) > max_negative_per_tree:
            all_pairs = rng.sample(all_pairs, max_negative_per_tree)
        for p, c in positives:
            u = emb[node_to_idx[p]]
            v = emb[node_to_idx[c]]
            feats.append(torch.cat([u, v, (u - v).abs(), u * v], dim=-1))
            labels.append(1.0)
        for p, c in all_pairs[: len(positives)]:
            u = emb[node_to_idx[p]]
            v = emb[node_to_idx[c]]
            feats.append(torch.cat([u, v, (u - v).abs(), u * v], dim=-1))
            labels.append(0.0)
    if not feats:
        return torch.empty(0), torch.empty(0)
    return torch.stack(feats), torch.tensor(labels, dtype=torch.float32)


def probe_and_score(
    train_pack: dict[str, torch.Tensor],
    val_pack: dict[str, torch.Tensor],
    test_pack: dict[str, torch.Tensor],
    task: str,
    num_classes: int | None = None,
    with_spearman: bool = False,
) -> dict[str, float]:
    if train_pack["x"].numel() == 0 or val_pack["x"].numel() == 0 or test_pack["x"].numel() == 0:
        return {"error": float("nan")}
    x_train, x_val, x_test = standardize(train_pack["x"], train_pack["x"], val_pack["x"], test_pack["x"])

    if task == "regression":
        model = fit_linear_probe(x_train, train_pack["y"], x_val, val_pack["y"], task="regression")
        with torch.no_grad():
            pred = model(x_test).squeeze(-1).cpu().numpy()
        return regression_metrics(test_pack["y"].cpu().numpy(), pred, with_spearman=with_spearman)

    if task == "binary":
        model = fit_linear_probe(x_train, train_pack["y"], x_val, val_pack["y"], task="binary")
        with torch.no_grad():
            score = model(x_test).squeeze(-1)
            pred = (torch.sigmoid(score) > 0.5).long().cpu().numpy()
            score = torch.sigmoid(score).cpu().numpy()
        return classification_metrics(test_pack["y"].cpu().numpy(), score, pred)

    if task == "multiclass":
        # Derive the class space from the DATA, not the caller's num_classes.
        # Remap train/val/test labels JOINTLY to a contiguous 0..K-1 range and
        # size the head to K. Fixes two bugs: (a) IndexError when the real
        # #classes exceeds the passed num_classes (num_children was hardcoded
        # to 3, but nodes with >2 children / unary nodes yield a 4th class),
        # and (b) test being scored in a different label space than the head
        # was trained on (the old remap only touched train+val). num_classes
        # arg is now advisory only.
        [y_tr, y_va, y_te], _ = remap_multiclass_labels(
            train_pack["y"], val_pack["y"], test_pack["y"])
        k = int(torch.cat([y_tr, y_va, y_te]).max().item()) + 1
        model = fit_linear_probe(x_train, y_tr, x_val, y_va, task="multiclass", num_classes=k)
        with torch.no_grad():
            logits = model(x_test)
            pred = logits.argmax(dim=-1).cpu().numpy()
            score = logits.cpu().numpy()
        return classification_metrics(y_te.cpu().numpy(), score, pred, num_classes=k)

    raise ValueError(task)


def filter_pack(pack: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[mask] for k, v in pack.items()}


def summarize_source(records: list[TreeRecord], source: str) -> dict[str, float]:
    xs = torch.cat([getattr(r, source) for r in records if getattr(r, source) is not None], dim=0)
    return embedding_health(xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out-dir", default="results/node_validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-trees", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=566)
    ap.add_argument("--lap-dim", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-negative-pairs-per-tree", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = TreeDataset(args.data, max_seq_len=args.max_seq_len)
    tree_indices = list(range(len(dataset)))
    if args.max_trees is not None:
        tree_indices = tree_indices[: args.max_trees]

    records = collect_tree_records(
        dataset=dataset,
        indices=tree_indices,
        device=args.device,
        max_seq_len=args.max_seq_len,
        lap_dim=args.lap_dim,
        checkpoint=args.checkpoint,
        use_cache_plm=True,
    )
    train_recs, val_recs, test_recs = split_records(records, args.seed, args.train_frac, args.val_frac)

    sources = [
        "plm",
        "topological_only",
        "esm_branch",
        "node_encoder",
        "graph_transformer",
        "graph_random",
    ]
    health_rows = []
    probe_rows = []

    for source in sources:
        if source in {"node_encoder", "graph_transformer"} and any(getattr(r, source) is None for r in records):
            continue
        if source == "graph_random" and any(r.graph_random is None for r in records):
            continue

        health = summarize_source(records, source)
        health["source"] = source
        health_rows.append(health)

        train_pack = flatten_node_samples(train_recs, source, args.max_seq_len)
        val_pack = flatten_node_samples(val_recs, source, args.max_seq_len)
        test_pack = flatten_node_samples(test_recs, source, args.max_seq_len)

        # (task_name, kind, y_tr, y_va, y_te, ncls, with_spearman)
        node_tasks = [
            ("leaf_vs_internal", "binary", train_pack["leaf"], val_pack["leaf"], test_pack["leaf"], None, False),
            ("node_depth", "regression", train_pack["depth"], val_pack["depth"], test_pack["depth"], None, False),
            ("root_to_node_distance", "regression", train_pack["rootdist"], val_pack["rootdist"], test_pack["rootdist"], None, False),
            ("subtree_size", "regression", train_pack["subtree"], val_pack["subtree"], test_pack["subtree"], None, True),
            ("num_children", "multiclass", train_pack["numchildren"], val_pack["numchildren"], test_pack["numchildren"], 3, False),
            ("hamming_to_parent", "regression", train_pack["parent_hamming"], val_pack["parent_hamming"], test_pack["parent_hamming"], None, False),
            ("substitutions_from_root", "regression", train_pack["root_hamming"], val_pack["root_hamming"], test_pack["root_hamming"], None, False),
        ]
        for task_name, kind, y_tr, y_va, y_te, ncls, with_rho in node_tasks:
            if kind == "regression" and task_name == "hamming_to_parent":
                tr_mask = torch.isfinite(y_tr)
                va_mask = torch.isfinite(y_va)
                te_mask = torch.isfinite(y_te)
                train_pack_use = filter_pack({"x": train_pack["x"], "y": y_tr}, tr_mask)
                val_pack_use = filter_pack({"x": val_pack["x"], "y": y_va}, va_mask)
                test_pack_use = filter_pack({"x": test_pack["x"], "y": y_te}, te_mask)
            else:
                train_pack_use = {"x": train_pack["x"], "y": y_tr}
                val_pack_use = {"x": val_pack["x"], "y": y_va}
                test_pack_use = {"x": test_pack["x"], "y": y_te}
            if train_pack_use["x"].numel() == 0 or test_pack_use["x"].numel() == 0:
                continue
            if kind == "binary":
                metrics = probe_and_score(
                    train_pack_use, val_pack_use, test_pack_use, task="binary",
                )
            elif kind == "regression":
                metrics = probe_and_score(
                    train_pack_use, val_pack_use, test_pack_use,
                    task="regression", with_spearman=with_rho,
                )
            else:
                metrics = probe_and_score(
                    train_pack_use, val_pack_use, test_pack_use,
                    task="multiclass", num_classes=ncls,
                )
            metrics.update({"source": source, "task": task_name})
            probe_rows.append(metrics)

        # Edge-level parent-child relationship
        pair_x_tr, pair_y_tr = build_pair_samples(train_recs, source, args.max_negative_pairs_per_tree)
        pair_x_va, pair_y_va = build_pair_samples(val_recs, source, args.max_negative_pairs_per_tree)
        pair_x_te, pair_y_te = build_pair_samples(test_recs, source, args.max_negative_pairs_per_tree)
        if pair_x_tr.numel() and pair_x_va.numel() and pair_x_te.numel():
            pair_metrics = probe_and_score(
                {"x": pair_x_tr, "y": pair_y_tr},
                {"x": pair_x_va, "y": pair_y_va},
                {"x": pair_x_te, "y": pair_y_te},
                task="binary",
            )
            pair_metrics.update({"source": source, "task": "parent_child_pair"})
            probe_rows.append(pair_metrics)

        # Mutation-specific probes on single-substitution edges only.
        mutpos_rows = []
        mutaa_rows = []
        for rec in train_recs:
            emb = getattr(rec, source)
            if emb is None:
                continue
            n2i = {nid: i for i, nid in enumerate(rec.node_ids)}
            for nid in rec.node_ids:
                if nid == rec.root_id:
                    continue
                pos, anc_idx, mut_idx = mutation_record(nid, rec.parent_map[nid], rec.seqs)
                if pos is None or anc_idx is None:
                    continue
                mutpos_rows.append((emb[n2i[nid]], pos))
                mutaa_rows.append((emb[n2i[nid]], anc_idx))

        def split_mut_samples(rows):
            if not rows:
                return None, None, None
            rng = random.Random(args.seed)
            rng.shuffle(rows)
            n = len(rows)
            n_tr = max(1, int(n * args.train_frac))
            n_va = max(1, int(n * args.val_frac))
            tr = rows[:n_tr]
            va = rows[n_tr:n_tr + n_va]
            te = rows[n_tr + n_va:]
            if not te:
                te = rows[-max(1, n - n_tr - n_va):]
            return tr, va, te

        def make_pack(rows):
            if not rows:
                return None
            x = torch.stack([r[0] for r in rows])
            y = torch.tensor([r[1] for r in rows])
            return {"x": x, "y": y}

        mp_tr, mp_va, mp_te = split_mut_samples(mutpos_rows)
        if mp_tr and mp_va and mp_te:
            mutpos_metrics = probe_and_score(
                make_pack(mp_tr), make_pack(mp_va), make_pack(mp_te), task="multiclass", num_classes=args.max_seq_len
            )
            mutpos_metrics.update({"source": source, "task": "mutated_position"})
            probe_rows.append(mutpos_metrics)

        ma_tr, ma_va, ma_te = split_mut_samples(mutaa_rows)
        if ma_tr and ma_va and ma_te:
            mutaa_metrics = probe_and_score(
                make_pack(ma_tr), make_pack(ma_va), make_pack(ma_te), task="multiclass", num_classes=len(AA_VOCAB)
            )
            mutaa_metrics.update({"source": source, "task": "ancestral_aa_identity"})
            probe_rows.append(mutaa_metrics)

    health_df = pd.DataFrame(health_rows)
    probe_df = pd.DataFrame(probe_rows)

    health_df.to_csv(out_dir / "embedding_health.csv", index=False)
    probe_df.to_csv(out_dir / "probe_results.csv", index=False)
    summary = {
        "n_trees": len(records),
        "n_train_trees": len(train_recs),
        "n_val_trees": len(val_recs),
        "n_test_trees": len(test_recs),
        "data": args.data,
        "checkpoint": args.checkpoint,
        "sources": sources,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if not probe_df.empty:
        wide = probe_df.copy()
        metric_cols = [c for c in probe_df.columns if c not in {"source", "task"}]
        wide = wide.melt(id_vars=["source", "task"], value_vars=metric_cols, var_name="metric", value_name="value")
        pivot = wide.pivot_table(index=["task", "metric"], columns="source", values="value")
        pivot.to_csv(out_dir / "probe_table_wide.csv")
        # Compact J.1-style view: primary metric per task.
        primary = {
            "leaf_vs_internal": "auroc",
            "node_depth": "r2",
            "root_to_node_distance": "r2",
            "subtree_size": "spearman",
            "num_children": "accuracy",
            "parent_child_pair": "auroc",
            "hamming_to_parent": "r2",
            "substitutions_from_root": "r2",
            "mutated_position": "accuracy",
            "ancestral_aa_identity": "accuracy",
        }
        j1_rows = []
        for task_name, metric in primary.items():
            sub = probe_df[probe_df["task"] == task_name]
            if sub.empty or metric not in sub.columns:
                continue
            row = {"task": task_name, "metric": metric}
            for _, r in sub.iterrows():
                row[r["source"]] = r[metric]
            j1_rows.append(row)
        if j1_rows:
            j1_df = pd.DataFrame(j1_rows)
            j1_df.to_csv(out_dir / "probe_table_j1_primary.csv", index=False)
            print("\nAppendix J.1 primary metrics (measured, not invented):")
            print(j1_df.to_string(index=False))

    print(f"\nWrote validation outputs to {out_dir}")
    if args.checkpoint is None:
        print("Note: --checkpoint omitted; trained node_encoder / graph_transformer skipped.")


if __name__ == "__main__":
    main()
