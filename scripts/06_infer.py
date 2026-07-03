#!/usr/bin/env python3
"""
06 - Run a trained MEEPO checkpoint on a new LAS/LAZ cloud.

Tiles the cloud, computes the same nine input channels (eight shallow features +
the previous-year DTM deviation channel, if a DTM is supplied), runs the model
tile-by-tile, stitches the per-point ground / non-ground prediction back to the
full cloud (overlap resolved by majority vote), and writes:
  * a classified ``.laz`` (prediction 1 -> ASPRS class 2 ground, 0 -> 1);
  * optionally a paper-style error image (needs ground-truth classes present).

    python scripts/06_infer.py --checkpoint runs/.../epoch_100/model.pt \
        --input tile.laz --out classified.laz --tiles data/nz/tiles \
        --prev-dtm data/nz/dtm/otago_2018_000.npz --error-image err.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meepo_nz.utils.config import Config
from meepo_nz.models import build_meepo
from meepo_nz.features.shallow_features import assemble_features, expected_feature_dim
from meepo_nz.data.batch import move_batch
from meepo_nz.data.dtm import Raster, height_above_prev_dtm, crop_dtm_patch, load_prior_raster
from meepo_nz.inference.voting import predict_cloud_spheres
from meepo_nz.utils.laz_io import (
    read_points, write_classified, label_from_classification,
)
from meepo_nz.training.visualize import render_error_image


def _load_raster(path):
    if not path or not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return Raster(data=d["data"].astype(np.float32),
                  x_min=float(d["x_min"]), y_min=float(d["y_min"]), res=float(d["res"]))


def _cfg_from_checkpoint(ckpt, override_path):
    if override_path:
        return Config.load(override_path)
    cfg = Config()
    saved = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if "in_features_dim" in ckpt:
        cfg.in_features_dim = ckpt["in_features_dim"]
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Classify a cloud with a trained model.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="LAS/LAZ to classify.")
    ap.add_argument("--out", required=True, help="Output classified .laz path.")
    ap.add_argument("--config", default=None, help="Override config (else read from checkpoint).")
    ap.add_argument("--tiles", default=None, help="Training tile dir (for norm_stats.json).")
    ap.add_argument("--norm-stats", default=None, help="Explicit norm_stats.json path.")
    ap.add_argument("--prev-dtm", default=None, help="Previous-year prior raster (.npz, 5-ch from stage 02; legacy 1-ch DTM also accepted).")
    ap.add_argument("--error-image", default=None, help="Render an error PNG to this path.")
    ap.add_argument("--neighbor-limit", type=int, default=50)
    ap.add_argument("--infer-batch", type=int, default=None, help="Spheres per forward pass (default 16; raise on a big GPU).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tta", action="store_true", help="Test-time augmentation: average softmax over z-rotations 0/90/180/270 (rotates cloud + prior raster).")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = _cfg_from_checkpoint(ckpt, args.config)
    cfg.in_features_dim = expected_feature_dim(cfg)

    model = build_meepo(cfg).to(device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    # checkpoints saved from a torch.compile'd model carry an "_orig_mod." prefix; strip it
    state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # normalisation stats
    mean = std = None
    norm_path = args.norm_stats or (os.path.join(args.tiles, "norm_stats.json")
                                    if args.tiles else None)
    if norm_path and os.path.exists(norm_path):
        with open(norm_path) as fh:
            st = json.load(fh)
        mean = np.asarray(st["mean"], dtype=np.float32)
        std = np.asarray(st["std"], dtype=np.float32)

    prev_dtm = load_prior_raster(args.prev_dtm)   # 5-ch prior (or legacy 1-ch, auto-promoted)

    xyz, classification, num_returns, return_number, intensity, rgb, meta = read_points(args.input, want_rgb=cfg.use_rgb)
    n = xyz.shape[0]

    if args.infer_batch is not None:

        cfg.infer_batch_spheres = int(args.infer_batch)


    pred_full = predict_cloud_spheres(
        xyz, num_returns, return_number, cfg, model, device,
        mean=mean, std=std, prev_dtm=prev_dtm,
        neighbor_limit=args.neighbor_limit, progress=50, intensity=intensity,
        tta=bool(args.tta))

    write_classified(args.out, xyz, pred_full, meta)
    n_ground = int(pred_full.sum())
    print(f"[06] sphere-voted {n:,} points -> ground={n_ground:,} "
          f"({100.0*n_ground/max(n,1):.1f}%)  wrote {args.out}")

    if args.error_image:
        true_label = label_from_classification(classification)
        if classification.max() >= 2:
            render_error_image(xyz, true_label, pred_full, args.error_image,
                               title=f"MEEPO inference  {os.path.basename(args.input)}")
            print(f"[06] error image -> {args.error_image}")
        else:
            print("[06] no ground-truth classes present; skipped error image")


if __name__ == "__main__":
    main()
