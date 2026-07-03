#!/usr/bin/env python3
"""
07 - Render error images for a REGION-DIVERSE set of held-out tiles.

Unlike 06 (which classifies a whole raw cloud), this runs a trained checkpoint
directly on the preprocessed tiles - which already carry the input features, the
previous-year DTM patch, and the ground-truth labels - so it is fast and needs no
raw LAZ or DTM lookup. It picks held-out (test) spheres spread across a RANGE OF
REGIONAL AREAS (not any terrain taxonomy) and writes one error PNG (and optionally
a classified .laz) per sphere, so you get a spread of scenes to choose from.

    python scripts/07_classify_tiles.py \
        --checkpoint runs/meepo_nz_ground/model_best.pt \
        --tiles data/nz/tiles \
        --out-dir runs/meepo_nz_ground/gallery_best \
        --n 8 --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meepo_nz.utils.config import Config
from meepo_nz.models import build_meepo
from meepo_nz.features.shallow_features import expected_feature_dim
from meepo_nz.data.dataset import SphereDataset
from meepo_nz.data.ptv3_collate import PTv3Collate
from meepo_nz.data.batch import move_batch
from meepo_nz.training.visualize import render_error_image
from meepo_nz.utils.laz_io import write_classified


def _per_cloud(points, lengths):
    out, s = [], 0
    for L in [int(x) for x in lengths]:
        out.append(points[s:s + L]); s += L
    return out


def main():
    ap = argparse.ArgumentParser(description="Error images for a region-diverse set of held-out tiles.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tiles", required=True, help="Preprocessed tile dir (holds norm_stats.json).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=8, help="number of region-diverse spheres to render.")
    ap.add_argument("--split", default="test", help="prefer this split; falls back to any.")
    ap.add_argument("--neighbor-limit", type=int, default=50)
    ap.add_argument("--save-laz", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model (restore config from checkpoint; strip torch.compile prefix) ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = Config()
    for k, v in (ckpt.get("config", {}) if isinstance(ckpt, dict) else {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.in_features_dim = expected_feature_dim(cfg)
    cfg.use_augmentation = False
    model = build_meepo(cfg).to(device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # ---- pick a region-diverse set: prefer the requested split, else any ----
    ds = SphereDataset(args.tiles, cfg=cfg, split=args.split, augment=False)
    if len(ds) == 0:
        ds = SphereDataset(args.tiles, cfg=cfg, split=None, augment=False)
    if len(ds) == 0:
        sys.exit(f"no spheres in {args.tiles}")
    selected = ds.gallery_center_indices(n_want=int(args.n))

    os.makedirs(args.out_dir, exist_ok=True)
    collate = PTv3Collate(cfg, neighbor_limit=args.neighbor_limit, device=None)
    tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    print(f"[07] rendering {len(selected)} region-diverse spheres "
          f"(split-pref={args.split}) -> {args.out_dir}", flush=True)

    for i in selected:
        sample = ds[i]
        batch = move_batch(collate([sample]), device, cfg)
        with torch.no_grad():
            logits = model(batch)
        pred = logits.argmax(dim=1).cpu().numpy()

        lengths0 = batch["lengths"][0].detach().cpu()
        local_pc = _per_cloud(batch["points"][0].detach().cpu(), lengths0)[0].numpy()
        labels_pc = batch["labels"].detach().cpu().numpy()
        world = local_pc.astype(np.float64) + sample["origin"][None, :]

        base = os.path.splitext(os.path.basename(sample["path"]))[0]
        png = os.path.join(args.out_dir, f"error_{base}.png")
        render_error_image(world, labels_pc, pred, png,
                           title=f"MEEPO ({tag})  {base}")
        if args.save_laz:
            try:
                write_classified(os.path.join(args.out_dir, f"classified_{base}.laz"), world, pred)
            except Exception as e:
                print(f"  [warn] LAZ skipped for {base}: {e}")
        ng = int((pred == 1).sum())
        print(f"[07] {base}  ground={ng:,}/{pred.shape[0]:,}  -> {os.path.basename(png)}",
              flush=True)

    print(f"[07] done. images in {args.out_dir}")


if __name__ == "__main__":
    main()
