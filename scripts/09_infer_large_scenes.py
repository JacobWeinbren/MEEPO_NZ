#!/usr/bin/env python3
"""
09 - Vote LARGE scenes with a trained checkpoint, reusing the EXACT per-epoch
gallery path (so results match what you saw during training).

This runs on the STAGE-04 PREPROCESSED TILES (data/<...>/tiles/*.npz), not raw
clouds. Each tile already holds the model's training-time inputs -- 10 cm-subsampled
points in a tile-local frame, the previous-year DTM, labels, returns/intensity.
The gallery voted 100 m windows of a single tile this way; here we vote a WHOLE
tile, or a contiguous BLOCK of merged adjacent tiles, for arbitrarily large scenes.

The tile-scan / grid-grouping / DTM-mosaic / merge / resume logic lives in
``meepo_nz.inference.large_scene`` (imported below) so it is shared and
testable rather than duplicated here; voting uses ``predict_cloud_spheres`` -- the
same routine the gallery and 06 call.

Per scene it writes a classified .laz (pred -> ASPRS 2/1, truth in ``true_class``),
a hillshaded error map, and a truth|pred|error review panel.

    # three largest single tiles (closest to the gallery; safest)
    python scripts/09_infer_large_scenes.py \
        --checkpoint runs/meepo_nz_nz_ground/model_best.pt \
        --tiles data/nz/tiles \
        --out-dir runs/meepo_nz_nz_ground/large_scenes --num-scenes 3

    # three largest contiguous grid blocks (bigger scenes)
    python scripts/09_infer_large_scenes.py \
        --checkpoint runs/meepo_nz_nz_ground/model_best.pt \
        --tiles data/nz/tiles \
        --out-dir runs/meepo_nz_nz_ground/large_scenes \
        --merge --num-scenes 3 --max-tiles 9
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
from meepo_nz.inference.large_scene import (
    scan_tiles, build_grid_blocks, load_block, resume_from_laz,
)
from meepo_nz.utils.laz_io import write_classified
from meepo_nz.training.visualize import render_error_image, render_review_panel


def _cfg_from_checkpoint(ckpt):
    cfg = Config()
    for k, v in (ckpt.get("config", {}) if isinstance(ckpt, dict) else {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if isinstance(ckpt, dict) and "in_features_dim" in ckpt:
        cfg.in_features_dim = ckpt["in_features_dim"]
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Vote large scenes from preprocessed tiles (gallery path).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tiles", required=True, help="Stage-04 preprocessed tile dir (holds *.npz + norm_stats.json).")
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-scenes", type=int, default=3)
    ap.add_argument("--merge", action="store_true", help="Merge contiguous adjacent tiles into larger grid blocks.")
    ap.add_argument("--max-tiles", type=int, default=9, help="--merge: max tiles per block (voting cost grows with area).")
    ap.add_argument("--split", default=None, help="Only use tiles from this split (e.g. test) for honest held-out scenes.")
    ap.add_argument("--neighbor-limit", type=int, default=50)
    ap.add_argument("--review-max-points", type=int, default=2_000_000)
    ap.add_argument("--no-dtm", action="store_true", help="Ignore the prev-year DTM patch (debug).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tta", action="store_true", help="Test-time augmentation: average softmax over z-rotations 0/90/180/270 (rotates cloud + prior raster).")
    ap.add_argument("--epsg", type=int, default=2193)
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-vote even if a scene's LAZ exists (default: resume visuals).")
    ap.add_argument("--inference", choices=["auto", "scene", "spheres"], default="auto",
                    help="Large-scene inference path. 'scene' = PTv3-native whole-scene "
                         "(grid-subsample at dl, tile into blocks with a context ring, voxelise, "
                         "expand voxel->point) and now supports the prior-raster branch; "
                         "'spheres' = overlapping-sphere voting. 'auto' (default) = scene for "
                         "every model. dl=0.1 either way.")
    ap.add_argument("--refine", choices=["spag_dc", "off"], default=None,
                    help="Ground-spike refinement at inference: 'spag_dc' (default, the SPAG-DC "
                         "misclassification corrector) or 'off' (raw MEEPO argmax).")
    ap.add_argument("--spag-alpha", type=float, default=None,
                    help="SPAG-DC region-growing curvature coeff alpha. Paper 0.5 (recommend 0.5-0.7).")
    ap.add_argument("--spag-beta", type=float, default=None,
                    help="SPAG-DC angle-relaxation coeff beta in [0,1). Paper 0.7 (recommend 0.6-0.8).")
    ap.add_argument("--spag-n-sigma", type=float, default=None,
                    help="SPAG-DC correction cut residual > mu2+n*sigma2. Paper-consistent n=3.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = _cfg_from_checkpoint(ckpt)
    cfg.in_features_dim = expected_feature_dim(cfg)
    model = build_meepo(cfg).to(device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # ---- choose inference path (PTv3-native whole-scene vs sphere voting) ----
    # PTv3-native whole-scene inference now supports the prior-raster branch: per-voxel
    # raster features are built the training way (local 2R windows) inside predict_scene,
    # so a raster-trained model runs whole-scene too. 'auto' prefers the whole-scene path
    # for every model; pass --inference spheres for the old sphere voting. Per-scene priors
    # are loaded below as `dtm`; if a scene has no prior, its raster features are zero.
    model_has_raster = bool(getattr(model, "use_raster", False))
    mode = args.inference
    if mode == "auto":
        mode = "scene"
    use_scene = (mode == "scene")
    cfg.scene_mode = bool(use_scene)             # routes predict_cloud_spheres -> predict_scene
    if args.refine is not None:            cfg.refine_method = args.refine
    if args.spag_alpha is not None:        cfg.spag_alpha = float(args.spag_alpha)
    if args.spag_beta is not None:         cfg.spag_beta = float(args.spag_beta)
    if args.spag_n_sigma is not None:      cfg.spag_n_sigma = float(args.spag_n_sigma)
    _refine_off = str(getattr(cfg, "refine_method", "spag_dc")).lower() in ("off", "none", "")
    _dl = float(getattr(cfg, "first_subsampling_dl", 0.1))
    if use_scene:
        _desc = (f"PTv3-native whole-scene (block-tiled, dl={_dl:.2f}"
                 f"{', + prior-raster branch' if model_has_raster else ''})")
    else:
        _desc = f"sphere voting (dl={_dl:.2f}, in_radius={float(getattr(cfg,'in_radius',6.0)):.0f} m)"
    print(f"[09] inference path: {_desc}"
          f"{'  | SPAG-DC OFF (raw argmax)' if _refine_off else f'  | SPAG-DC correct (alpha={cfg.spag_alpha:g}, beta={cfg.spag_beta:g}, n_sigma={cfg.spag_n_sigma:g})'}")
    mean = std = None
    norm_path = args.norm_stats or os.path.join(args.tiles, "norm_stats.json")
    if os.path.exists(norm_path):
        with open(norm_path) as fh:
            st = json.load(fh)
        mean = np.asarray(st["mean"], dtype=np.float32)
        std = np.asarray(st["std"], dtype=np.float32)
    else:
        print(f"[09] WARNING: no norm_stats.json ({norm_path}); features will be unnormalised.")

    # ---- pick scenes from the preprocessed tiles ----
    tiles = scan_tiles(args.tiles, args.split)
    if not tiles:
        sys.exit(f"[09] ERROR: no tiles in {args.tiles}"
                 + (f" for split={args.split}" if args.split else "") + ". Run stage 04 first.")
    if args.merge:
        blocks = build_grid_blocks(tiles, args.max_tiles, args.num_scenes)
        scenes = [{"name": b["name"], "tiles": b["tiles"], "n_tiles": b["n_tiles"], "n": b["n"]} for b in blocks]
        print(f"[09] --merge: {len(scenes)} grid blocks from {len(tiles)} tiles (<= {args.max_tiles} tiles each):")
        for s in scenes:
            print(f"[09]   {s['name']}: {s['n_tiles']} tiles, {s['n']:,} pts")
    else:
        tiles.sort(key=lambda t: t["n"], reverse=True)
        picked = tiles[: max(args.num_scenes, 1)]
        scenes = [{"name": os.path.splitext(os.path.basename(t["path"]))[0],
                   "tiles": [t], "n_tiles": 1, "n": t["n"]} for t in picked]
        npts = ", ".join(f"{s['n']:,}" for s in scenes)
        print(f"[09] {len(tiles)} tiles; running the {len(scenes)} largest (pts: {npts})")

    rng = np.random.default_rng(0)
    summary = []
    for i, s in enumerate(scenes, 1):
        name = s["name"]
        laz = os.path.join(args.out_dir, f"classified_{name}.laz")
        print(f"\n[09] === scene {i}/{len(scenes)}: {name}  ({s['n_tiles']} tile(s), {s['n']:,} pts) ===")

        have = os.path.exists(laz) and not args.force
        xyz = pred = true_label = None
        if have:
            try:
                xyz, pred, true_label = resume_from_laz(laz)
                print(f"[09]   resume: re-rendering visuals from existing LAZ ({xyz.shape[0]:,} pts; no re-vote).")
            except Exception as e:
                print(f"[09]   resume failed ({e}); re-voting"); have = False

        if not have:
            rxyz, nr, rn, inten, rr, lab, dtm, origin = load_block(s["tiles"])
            if args.no_dtm:
                dtm = None
            if dtm is None:
                print("[09]   note: no prev-year DTM patch for this scene (DTM channel = 0).")
            _path = "PTv3-native scene (block-tiled)" if use_scene else "sphere-voting"
            print(f"[09]   {rxyz.shape[0]:,} points; {_path} inference (dl="
                  f"{float(getattr(cfg,'first_subsampling_dl',0.1)):.2f}) ...")
            pred = predict_cloud_spheres(
                rxyz, nr, rn, cfg, model, device, mean=mean, std=std,
                prev_dtm=dtm, neighbor_limit=args.neighbor_limit,
                intensity=inten, ret_ratio=rr, progress=200,
                tta=bool(args.tta))
            xyz = (rxyz.astype(np.float64) + origin[None, :])     # back to world coords
            true_label = lab
            # clean, QGIS-safe LAZ (standard LAS fields only; reference labels live in the visuals)
            write_classified(laz, xyz, pred,
                             num_returns=nr, return_number=rn, intensity=inten,
                             epsg=args.epsg)
            ng = int(np.asarray(pred).sum())
            print(f"[09]   ground={ng:,} ({100.0*ng/max(rxyz.shape[0],1):.1f}%)  -> {laz}")

        n = xyz.shape[0]
        pred = np.asarray(pred)
        if true_label is None:
            true_label = np.zeros(n, dtype=np.int64)

        span = float(max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]))) if n else 0.0
        grid_res = int(np.clip(span / 2.0, 260, 1400))
        err = os.path.join(args.out_dir, f"error_{name}.png")
        try:
            render_error_image(xyz, true_label, pred, err, title=f"MEEPO  {name}", grid_res=grid_res)
            print(f"[09]   error image -> {err}  (grid_res={grid_res})")
        except Exception as e:
            err = None
            print(f"[09]   error image skipped ({e})")

        rev = None
        if not args.no_review:
            rev = os.path.join(args.out_dir, f"review_{name}.png")
            if n > args.review_max_points:
                sel = rng.choice(n, args.review_max_points, replace=False)
                vx, vt, vp = xyz[sel], true_label[sel], pred[sel]
                note = f"{name}  ({n:,} pts, plot shows {args.review_max_points:,})"
            else:
                vx, vt, vp, note = xyz, true_label, pred, f"{name}  ({n:,} pts)"
            try:
                m = render_review_panel(vx, vt, vp, rev, subtitle=note)
                print(f"[09]   review panel -> {rev}  "
                      f"(IoU_ground={m.get('IoU_ground', float('nan')):.1f} "
                      f"IoU_nonground={m.get('IoU_nonground', float('nan')):.1f} "
                      f"OA={m.get('OA', float('nan')):.1f})")
            except Exception as e:
                rev = None
                print(f"[09]   review panel skipped ({e})")
        summary.append((name, laz, err, rev))

    print("\n[09] done. outputs:")
    for name, laz, err, rev in summary:
        print(f"     {name}: {laz}" + (f" | {err}" if err else "") + (f" | {rev}" if rev else ""))


if __name__ == "__main__":
    main()
