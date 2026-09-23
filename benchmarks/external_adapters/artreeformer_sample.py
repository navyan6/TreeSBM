#!/usr/bin/env python3
"""Sample topologies from a trained ARTreeFormer TDE checkpoint.

Place this script in the upstream ARTreeFormer ``TDE/`` directory and run it
in that environment (needs local ``models`` / ``datasets`` imports).

Example::

    python artreeformer_sample.py --checkpoint final.pt --ntips 16 \
        --n-samples 300 --out artreeformer_N16.nwk
"""

import argparse
import sys
sys.path.append("..")

import torch
from omegaconf import OmegaConf

from models import TDE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ntips", type=int, required=True)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="ema", choices=["ema", "model"])
    args, unknown = ap.parse_known_args()

    cfg_file = OmegaConf.load("config.yaml")
    cfg_cmd = OmegaConf.from_cli(unknown)
    cfg = OmegaConf.merge(cfg_file, cfg_cmd)
    cfg.base.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TDE(dataloader=None, ntips=args.ntips, emp_tree_freq=None,
                model_cfg=cfg.model).to(cfg.base.device)
    ckpt = torch.load(args.checkpoint, map_location=cfg.base.device)
    model.tree_model.load_state_dict(ckpt[args.key])
    model.eval()

    with torch.no_grad():
        trees, _, _ = model.sample_trees(args.n_samples)

    with open(args.out, "w") as f:
        for t in trees:
            f.write(t.write(format=9) + "\n")   # format=9: leaf names only, no branch lengths
    print(f"wrote {len(trees)} sampled topologies -> {args.out}")


if __name__ == "__main__":
    main()
