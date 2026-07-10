"""
08 - Classify a spatial WINDOW of a cloud with sphere voting.

With KPConv input spheres there are no tiles to stitch: step 06 already sphere-
votes a whole cloud contiguously. This script is the spot-check tool - it crops a
``--window`` metre square (centred on ``--center x,y`` or the cloud centroid) out
of a raw cloud, sphere-votes just that window, and writes a classified ``.laz``
plus an optional paper-style error image. Handy for eyeballing one neighbourhood
of a very large cloud without running the whole thing.

Example:
    python scripts/08_classify_grid.py \
        --checkpoint runs/meepo_nz_nz_ground/epoch_500/model.pt \
        --input data/nz/raw/CL2_BX24_2023_1000_2743.laz \
        --tiles data/nz/tiles --center 1000.0,2743.0 --window 200 \
        --out runs/.../inference/window_2743.laz --error-image runs/.../window_2743.png
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
from meepo_nz.features.shallow_features import expected_feature_dim
from meepo_nz.inference.voting import predict_cloud_spheres
from meepo_nz.data.dtm import Raster, load_prior_raster
from meepo_nz.utils.laz_io import read_points, write_classified, label_from_classification
from meepo_nz.training.visualize import render_error_image


def _load_raster(path):
    if not path or not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return Raster(data=d["data"].astype(np.float32), x_min=float(d["x_min"]),
                  y_min=float(d["y_min"]), res=float(d["res"]))


def main():
    ap = argparse.ArgumentParser(description="Sphere-vote a window of a cloud.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="raw LAS/LAZ cloud.")
    ap.add_argument("--tiles", default=None, help="training tile dir (for norm_stats.json).")
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--center", default=None, help='"x,y" window centre (default cloud centroid).')
    ap.add_argument("--window", type=float, default=200.0, help="window side length (m).")
    ap.add_argument("--prev-dtm", default=None)
    ap.add_argument("--out", default=None, help="output classified .laz (default: auto).")
    ap.add_argument("--error-image", default=None)
    ap.add_argument("--neighbor-limit", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tta", action="store_true", help="Test-time augmentation: average softmax over z-rotations 0/90/180/270 (rotates cloud + prior raster).")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    mean = std = None
    norm_path = args.norm_stats or (os.path.join(args.tiles, "norm_stats.json") if args.tiles else None)
    if norm_path and os.path.exists(norm_path):
        with open(norm_path) as fh:
            st = json.load(fh)
        mean = np.asarray(st["mean"], dtype=np.float32); std = np.asarray(st["std"], dtype=np.float32)
    prev_dtm = load_prior_raster(args.prev_dtm)

    xyz, classification, num_returns, return_number, intensity, rgb, meta = read_points(args.input, want_rgb=cfg.use_rgb)
    if args.center:
        cx, cy = (float(v) for v in args.center.split(","))
    else:
        cx, cy = float(xyz[:, 0].mean()), float(xyz[:, 1].mean())
    h = float(args.window) / 2.0
    m = ((xyz[:, 0] >= cx - h) & (xyz[:, 0] < cx + h) &
         (xyz[:, 1] >= cy - h) & (xyz[:, 1] < cy + h))
    sub = np.where(m)[0]
    if sub.size < int(getattr(cfg, 'tile_stats_min_points', 100)):
        sys.exit(f"window has too few points ({sub.size}); widen --window or move --center")
    wx = xyz[sub]; wnr = num_returns[sub]; wrn = return_number[sub]; wint = intensity[sub]

    pred = predict_cloud_spheres(wx, wnr, wrn, cfg, model, device,
                                 mean=mean, std=std, prev_dtm=prev_dtm,
                                 neighbor_limit=args.neighbor_limit, progress=25, intensity=wint,
                                 tta=bool(args.tta))

    base = os.path.splitext(os.path.basename(args.input))[0]
    out = args.out or os.path.join(os.path.dirname(args.checkpoint), "inference",
                                   f"window_{int(cx)}_{int(cy)}_{base}.laz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_classified(out, wx, pred, meta)
    ng = int(pred.sum())
    print(f"[08] window {args.window:.0f} m @ ({cx:.0f},{cy:.0f}): {wx.shape[0]:,} pts "
          f"-> ground={ng:,} ({100.0*ng/max(wx.shape[0],1):.1f}%)  wrote {out}")

    if args.error_image and classification[sub].max() >= 2:
        true_label = label_from_classification(classification[sub])
        render_error_image(wx, true_label, pred, args.error_image,
                            title=f"MEEPO window  {base}")
        print(f"[08] error image -> {args.error_image}")


if __name__ == "__main__":
    main()
