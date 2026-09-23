#!/usr/bin/env python3
"""Build a size-N H3N2 topology bank for PhylaFlow sampling.

Uses TreeSBM H3N2 train subtree shapes as targets (not public DS1–8 banks).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path


def _ensure_semicolon(tree: str) -> str:
    tree = str(tree).strip()
    return tree if tree.endswith(";") else tree + ";"


def _load_topologies(path: Path, n_cases: int, seed: int) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"No topologies in {path}")
    rng = random.Random(seed)
    if n_cases > 0 and n_cases < len(lines):
        return rng.sample(lines, n_cases)
    rng.shuffle(lines)
    return lines if n_cases <= 0 else lines[:n_cases]


def _random_start_newick(num_leaves: int, seed: int) -> str:
    """Random N-leaf start in PhylaFlow's Tree convention (adds ROOT_DUMMY)."""
    # Prefer PhylaFlow's Tree generator when importable (run from PhylaFlow cwd).
    try:
        from utils.random_tree import Tree  # type: ignore

        rng_state = random.getstate()
        try:
            random.seed(seed)
            t = Tree(num_leaves=num_leaves, random=True)
            # Drop dummy for JSON bank — PhylaFlow re-wraps on load.
            from ete3 import Tree as EteTree

            nw = str(t) if not hasattr(t, "write") else None
            # Tree.__str__ returns newick via networkx helper; parse and prune dummy.
            raw = nw if nw else EteTree(str(t), format=1)
            if isinstance(raw, str):
                ete = EteTree(raw, format=1)
            else:
                ete = raw
            # Prefer writing biological leaves only (0..N-1).
            leaves = [lf for lf in ete.get_leaves() if lf.name != "ROOT_DUMMY"]
            if len(leaves) == num_leaves:
                keep = [lf.name for lf in leaves]
                ete.prune(keep, preserve_branch_length=True)
                return _ensure_semicolon(ete.write(format=1))
        finally:
            random.setstate(rng_state)
    except Exception:
        pass

    # Fallback: ete3 random topology with integer leaf names.
    from ete3 import Tree as EteTree

    ete = EteTree()
    ete.populate(num_leaves, random_branches=True)
    for i, lf in enumerate(sorted(ete.get_leaves(), key=lambda x: x.name or "")):
        lf.name = str(i)
        if lf.dist is None or lf.dist <= 0:
            lf.dist = 0.1
    return _ensure_semicolon(ete.write(format=1))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_config(
    repo_dir: Path,
    data_root: Path,
    N: int,
    n_cases: int,
    case_prefix: str,
    artifact_dir: Path,
    short_run_root: Path,
    golden_run_root: Path,
    dataset_id: str,
) -> Path:
    start_paths = sorted(artifact_dir.glob(f"{case_prefix}_case*_start.json"))
    target_paths = sorted(artifact_dir.glob(f"{case_prefix}_case*_target.json"))
    if len(start_paths) != n_cases or len(target_paths) != n_cases:
        raise RuntimeError(
            f"Expected {n_cases} start/target pairs, found "
            f"{len(start_paths)}/{len(target_paths)} in {artifact_dir}"
        )

    # Base on DS2 recipe shape (proven fixed-pair + full-path control), but point
    # all data roots at H3N2 bank — never DS1–8.
    cfg = {
        "model": {
            "num_node_types": 3,
            "num_edge_types": 2,
            "hidden_dim": 64,
            "embed_dim": 64,
            "output_dim": 1,
            "n_layers": 2,
            "n_heads": 4,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "drop_path_rate": 0.0,
            "use_performer": False,
            "performer_nb_features": 64,
            "performer_generalized_attention": True,
            "layernorm_style": "prenorm",
            "tokenizer_lap_dim": 8,
            "tokenizer_lap_dropout": 0.0,
            "tokenizer_n_layers": 2,
            "phyla_dim": 16,
            "autoregressive_head_mode": "structured_subset",
            "first_hit_head_use_phase_input": True,
            "first_hit_head_phase_hidden_dim": 64,
            "first_hit_head_mode": "case_adapted_mlp",
            "first_hit_head_num_cases": int(n_cases),
        },
        "trainer": {
            "record": True,
            "epochs": 200,
            "checkpoint_dir": (
                f"${{PHYLAFLOW_OUTPUT_ROOT}}/checkpoints/h3n2_N{N}_"
                f"multipair{n_cases}_fullpath"
            ),
            "phyla_checkpoint_path": None,
            "phyla_precomputed_embeddings_path": None,
            "lr": 0.005,
            "steps_callback": 1000,
            "val_callback_freq": 0,
            "limit_val_batches": 0,
            "velocity_loss_mode": "plain",
            "velocity_loss_plain_weight": 1.0,
            "training_step_velocity_weight": 1,
            "training_step_autoregressive_weight": 0.2,
            "training_step_gradient_clip_val": 1.0,
            "training_step_separate_optimizer_steps": True,
            "autoregressive_use_time": True,
            "autoregressive_target_mode": "scheduled",
            "velocity_event_weight": 10.0,
            "velocity_first_hit_head_weight": 10.0,
            "velocity_first_hit_head_use_at_sampling": True,
            "velocity_terminal_head_weight": 1.0,
            "velocity_terminal_head_input_mode": "edge_topology",
            "velocity_probe_direct_set_loss": True,
            "velocity_probe_direct_set_anchor_only": True,
            "training_step_probe_parity_joint_update": True,
            "sampling_discrete_phase_rollout_use_at_sampling": True,
            "sampling_discrete_phase_exact_boundary_step_use_at_sampling": True,
            "sampling_discrete_phase_max_phases": 8,
            "sampling_max_steps": 128,
            "sampling_max_events": 500,
            "sampling_use_inference_mode": True,
            "sampling_random_fixed_pair_bank_use_at_sampling": True,
            "training_sampling_start": 1000000000,  # defer sample-KL until dumps
            "training_sampling_frequency": 1000000000,
            "seed": 42,
            "wandb_dir": "${PHYLAFLOW_OUTPUT_ROOT}/wandb",
            "wandb_project": "TreeSBM_PhylaFlow_H3N2",
            "wandb_name": f"h3n2_N{N}_multipair{n_cases}",
            "wandb_group": f"H3N2_N{N}",
            "wandb_tags": [f"H3N2_N{N}", "treesbm_table2", "not_ds1-8"],
            "sample_metrics_num_pairs": int(n_cases),
            "sample_metrics_trace_path": (
                f"${{PHYLAFLOW_OUTPUT_ROOT}}/metrics/h3n2_N{N}_multipair{n_cases}.jsonl"
            ),
        },
        "data": {
            "nexus_root": "unused",
            "mrbayes_root": "unused",
            "short_run_dataset_id": dataset_id,
            "short_run_root": str(short_run_root),
            # Point golden at the same H3N2 bank (metrics plumbing only).
            "golden_run_root": str(golden_run_root),
            "trprobs_sample_count_per_file": min(1000, max(50, n_cases * 10)),
            "use_random_sequence_distribution": True,
            "random_distribution_sequence_length": 256,
            "random_distribution_alphabet": "ACDEFGHIKLMNPQRSTVWY",
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "overfit_fixed_pair": True,
            "overfit_fixed_pair_group_by_json_metadata": True,
            "overfit_fixed_pair_cache_virtual_index_selection": True,
            "overfit_virtual_epoch_size": int(n_cases),
            "overfit_full_path_control_mode": True,
            "overfit_full_path_control_use_discrete_phase_time": True,
            "overfit_full_path_control_seed": 271828,
            "overfit_velocity_explicit_boundary_end_states": True,
            "overfit_split_multi_subset_events": True,
            "overfit_boundary_prefix_k": -1,
            "overfit_fixed_pair_start_tree_json_path": str(start_paths[0]),
            "overfit_fixed_pair_target_tree_json_path": str(target_paths[0]),
            "overfit_fixed_pair_start_tree_json_paths": [str(p) for p in start_paths],
            "overfit_fixed_pair_target_tree_json_paths": [str(p) for p in target_paths],
        },
    }

    out = repo_dir / "configs" / f"h3n2_N{N}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)

    def emit(obj, indent=0):
        sp = "  " * indent
        if isinstance(obj, dict):
            lines = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{sp}{k}:")
                    lines.append(emit(v, indent + 1))
                elif v is None:
                    lines.append(f"{sp}{k}: null")
                elif isinstance(v, bool):
                    lines.append(f"{sp}{k}: {'true' if v else 'false'}")
                elif isinstance(v, (int, float)):
                    lines.append(f"{sp}{k}: {v}")
                else:
                    s = str(v).replace("\\", "\\\\").replace("\"", "\\\"")
                    lines.append(f"{sp}{k}: \"{s}\"")
            return "\n".join(lines)
        if isinstance(obj, list):
            lines = []
            for item in obj:
                if isinstance(item, (dict, list)):
                    lines.append(f"{sp}-")
                    lines.append(emit(item, indent + 1))
                elif isinstance(item, bool):
                    lines.append(f"{sp}- {'true' if item else 'false'}")
                elif isinstance(item, (int, float)):
                    lines.append(f"{sp}- {item}")
                else:
                    s = str(item).replace("\\", "\\\\").replace("\"", "\\\"")
                    lines.append(f"{sp}- \"{s}\"")
            return "\n".join(lines)
        return f"{sp}{obj}"

    try:
        import yaml  # type: ignore

        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    except Exception:
        out.write_text(emit(cfg) + "\n")
    return out


def build(args: argparse.Namespace) -> dict:
    N = int(args.N)
    n_cases = int(args.n_cases)
    seed = int(args.seed)
    data_root = Path(args.data_root).expanduser().resolve()
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    topologies = Path(args.topologies).expanduser().resolve()
    trprobs = Path(args.trprobs).expanduser().resolve() if args.trprobs else None

    bank = data_root / f"h3n2_N{N}"
    artifact_dir = bank / "fixed_path_artifacts"
    short_run_root = bank / "short_run"
    golden_run_root = bank / "golden_run"
    dataset_id = f"H3N2_N{N}"
    case_prefix = f"h3n2_N{N}"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    (short_run_root / dataset_id).mkdir(parents=True, exist_ok=True)
    (golden_run_root / dataset_id).mkdir(parents=True, exist_ok=True)

    targets = _load_topologies(topologies, n_cases=n_cases, seed=seed)
    n_cases = len(targets)

    # Posterior reference = H3N2 train topologies (NOT DS1–8).
    if trprobs and trprobs.is_file():
        dest = short_run_root / dataset_id / trprobs.name
        shutil.copy2(trprobs, dest)
        gdest = golden_run_root / dataset_id / trprobs.name
        if gdest.exists() or gdest.is_symlink():
            gdest.unlink()
        try:
            gdest.symlink_to(dest)
        except OSError:
            shutil.copy2(dest, gdest)
    else:
        # Write a minimal NEXUS trees block from the selected targets.
        dest = short_run_root / dataset_id / f"train_topologies_N{N}.trprobs"
        with dest.open("w") as fh:
            fh.write("#NEXUS\nbegin trees;\n")
            w = 1.0 / max(len(targets), 1)
            for i, nw in enumerate(targets, start=1):
                fh.write(f"  tree t{i} = [&W {w}] [&U] {_ensure_semicolon(nw)}\n")
            fh.write("end;\n")
        gdest = golden_run_root / dataset_id / dest.name
        if gdest.exists() or gdest.is_symlink():
            gdest.unlink()
        try:
            gdest.symlink_to(dest)
        except OSError:
            shutil.copy2(dest, gdest)

    index_rows = []
    width = max(2, len(str(n_cases - 1)))
    for i, target_nw in enumerate(targets):
        group_key = f"{case_prefix}_case{i:0{width}d}"
        start_nw = _random_start_newick(N, seed=seed + 17 * i + 1)
        start_path = artifact_dir / f"{group_key}_start.json"
        target_path = artifact_dir / f"{group_key}_target.json"
        _write_json(
            start_path,
            {
                "group_key": group_key,
                "bank_group_key": group_key,
                "dataset_id": dataset_id,
                "case_index": i,
                "num_leaves": N,
                "start_tree": _ensure_semicolon(start_nw),
                "tree": _ensure_semicolon(start_nw),
            },
        )
        _write_json(
            target_path,
            {
                "group_key": group_key,
                "bank_group_key": group_key,
                "dataset_id": dataset_id,
                "case_index": i,
                "num_leaves": N,
                "target_tree": _ensure_semicolon(target_nw),
                "tree": _ensure_semicolon(target_nw),
            },
        )
        index_rows.append(
            {
                "case_index": i,
                "dataset_id": dataset_id,
                "num_leaves": N,
                "start_path": str(start_path),
                "target_path": str(target_path),
                "bank_group_key": group_key,
            }
        )

    index_path = bank / "topology_stream_index.jsonl"
    with index_path.open("w") as fh:
        for row in index_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    cfg_path = _write_config(
        repo_dir=repo_dir,
        data_root=data_root,
        N=N,
        n_cases=n_cases,
        case_prefix=case_prefix,
        artifact_dir=artifact_dir,
        short_run_root=short_run_root,
        golden_run_root=golden_run_root,
        dataset_id=dataset_id,
    )

    manifest = {
        "bank": str(bank),
        "N": N,
        "n_cases": n_cases,
        "dataset_id": dataset_id,
        "short_run_root": str(short_run_root),
        "golden_run_root": str(golden_run_root),
        "fixed_path_artifacts": str(artifact_dir),
        "topology_stream_index": str(index_path),
        "config": str(cfg_path),
        "note": "H3N2 Table-2 bank — not PhylaFlow DS1–8",
    }
    _write_json(bank / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, required=True, help="Leaf count (typically 16 or 32)")
    ap.add_argument("--topologies", type=Path, required=True)
    ap.add_argument("--trprobs", type=Path, default=None)
    ap.add_argument("--n-cases", type=int, default=42)
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get(
            "PHYLAFLOW_DATA_ROOT",
            "/vast/projects/pranam/lab/nnori/baselines/phylaflow_h3n2_data",
        )),
    )
    ap.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(os.environ.get(
            "PHYLAFLOW_REPO",
            "/vast/projects/pranam/lab/nnori/baselines/PhylaFlow",
        )),
    )
    args = ap.parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
