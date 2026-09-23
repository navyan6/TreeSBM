#!/usr/bin/env python3
"""Sample topologies from a trained PhyloVAE checkpoint.

Run from the upstream PhyloVAE repo root so local imports resolve.

Example::

    python phylovae_sample.py --checkpoint model.pt --ntips 16 \
        --n-samples 300 --out phylovae_N16.nwk
"""

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from src.latent_tree_model import VAETree
from src.vector_representation import vec2tree


class _FakeDataset:
    # VAETree.__init__ reads self.dataloader.dataset.wts for an (unused-at-
    # sampling-time) entropy bookkeeping constant; sampling itself never
    # touches the dataloader again, so a dummy avoids needing real data here.
    wts = np.array([1.0])


class _FakeLoader:
    dataset = _FakeDataset()


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

    taxa = [str(i) for i in range(args.ntips)]
    model = VAETree(taxa, _FakeLoader(), None, cfg=cfg).to(cfg.base.device)
    ckpt = torch.load(args.checkpoint, map_location=cfg.base.device)
    model.load_state_dict(ckpt[args.key])
    model.eval()

    ltm = model.latent_tree_model
    with torch.no_grad():
        samp_z, _ = ltm.sample_z(args.n_samples)              # [K, latent_dim]
        latent_logits = ltm.decoder(samp_z)                   # [K, (ntips-3)*(ntips-1)]
        cond_probs_mat = ltm.cond_prob_mat(latent_logits)      # [K, ntips-3, 2*ntips-4]

    trees = []
    for k in range(args.n_samples):
        vec = [torch.multinomial(cond_probs_mat[k, i], 1).item() for i in range(args.ntips - 3)]
        trees.append(vec2tree(vec))

    with open(args.out, "w") as f:
        for t in trees:
            f.write(t.write(format=9) + "\n")   # format=9: leaf names only, no branch lengths
    print(f"wrote {len(trees)} sampled topologies -> {args.out}")


if __name__ == "__main__":
    main()
