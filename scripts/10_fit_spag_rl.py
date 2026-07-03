#!/usr/bin/env python3
"""Calibrate the learned SPAG-DC globals to MINIMISE DTM-RMSE via self-critical REINFORCE.

The regime head -- fed pooled backbone features AND the model's own per-scene prediction
statistics -- outputs the six SPAG-DC globals. SPAG-DC is non-differentiable, so we treat
the head's outputs as a Gaussian policy mean, sample globals, run the real corrector, and
reward the NEGATIVE OpenGF DTM-RMSE-vs-GT-ground (the metric the validator reports). That
metric is computed over a DEM grid spanning the whole tile, so demoting genuine cliff-ground
corrupts the predicted DEM there and RAISES RMSE -- optimising it discourages cliff
destruction with no hand-coded slope rule. Only the regime head is trained; backbone frozen.

Start from a trained checkpoint (backbone/seg-head loaded as-is; the regime head is re-fit
here). Produces <out> with the RMSE-optimised head; pass it as the inference checkpoint.

  python3 scripts/10_fit_spag_rl.py --tiles data/nz/tiles --ckpt runs/meepo/model_best.pt \
      --device cuda --dl 0.1 --scene-max-points 200000 \
      --rl-scenes 64 --rl-max-points 40000 --rl-iters 300 --rl-batch 8 --out runs/meepo/model_rl.pt
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meepo_nz.utils.config import Config
from meepo_nz.models import build_meepo
from meepo_nz.data.ptv3_collate import PTv3Collate
from meepo_nz.data.scene_dataset import SceneDataset
from meepo_nz.inference.spag_rl import reinforce_update


def _subsample(n, cap, rng):
    return np.arange(n) if n <= cap else rng.choice(n, size=cap, replace=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--dl", type=float, default=None)
    ap.add_argument("--scene-max-points", type=int, default=None)
    ap.add_argument("--refine", default="spag_dc")
    ap.add_argument("--no-dtm-raster", action="store_true")
    ap.add_argument("--rl-scenes", type=int, default=64, help="tiles to cache as the calibration set")
    ap.add_argument("--rl-max-points", type=int, default=40000, help="per-cloud subsample for the corrector reward")
    ap.add_argument("--rl-iters", type=int, default=300)
    ap.add_argument("--rl-batch", type=int, default=8, help="scenes per REINFORCE step")
    ap.add_argument("--rl-sigma", type=float, default=0.5, help="policy std in logit space")
    ap.add_argument("--rl-lr", type=float, default=3e-3)
    ap.add_argument("--reward", choices=["rmse", "p95", "p99", "max"], default="p95",
                    help="reward aggregation over per-pixel DEM error (default p95; tail metrics target cliffs)")
    ap.add_argument("--rmse-res", type=float, default=None, help="DTM-RMSE grid res (m); default cfg.dtm_rmse_res or 1.0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    cfg_path = os.path.join(args.tiles, "config.used.yaml")
    cfg = Config.load(cfg_path) if os.path.exists(cfg_path) else Config()
    cfg.scene_mode = True
    cfg.spag_learned = True
    cfg.refine_method = args.refine
    if args.dl is not None:
        cfg.first_subsampling_dl = float(args.dl)
    if args.scene_max_points is not None:
        cfg.scene_max_points = int(args.scene_max_points)
    cfg.use_dtm_raster = (not args.no_dtm_raster) and bool(getattr(cfg, "use_dtm_raster", True))
    cfg.mix_prob = 0.0
    res = float(args.rmse_res if args.rmse_res is not None else getattr(cfg, "dtm_rmse_res", 1.0))

    # model + checkpoint. The regime head changed shape (now ingests prediction stats), so
    # drop any old regime_head.* weights and re-fit them here; backbone/seg-head load as-is.
    model = build_meepo(cfg).to(device)
    if not getattr(model, "spag_learned", False):
        sys.exit("[rl] model built with spag_learned=False -- set cfg.spag_learned/--spag-learned")
    sd = torch.load(args.ckpt, map_location=device)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    sd = {k: v for k, v in sd.items() if not k.startswith("regime_head")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[rl] loaded {args.ckpt}: {len(missing)} missing / {len(unexpected)} unexpected "
          f"(regime_head intentionally re-initialised for RMSE calibration)")

    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("regime_head")
    head_params = [p for n, p in model.named_parameters() if n.startswith("regime_head")]
    opt = torch.optim.Adam(head_params, lr=args.rl_lr)
    model.eval()   # head is norm/dropout-free, so eval is correct and keeps backbone stats frozen

    ds = SceneDataset(args.tiles, cfg, split=args.split)
    collate = PTv3Collate(cfg, device=None, mix_prob=0.0)
    n_avail = len(ds)
    pick = [int(i) for i in rng.choice(n_avail, size=min(args.rl_scenes, n_avail), replace=False)]
    print(f"[rl] caching {len(pick)} scenes from split={args.split} (of {n_avail}); "
          f"per-cloud subsample {args.rl_max_points} pts for the corrector reward ...")

    scenes = []
    for j, idx in enumerate(pick):
        try:
            batch = collate([ds[idx]])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(batch)
            if getattr(model, "_regime_pooled", None) is None:
                continue
            pooled = model._regime_pooled.detach().cpu()
            pstats = model._regime_pred_stats.detach().cpu()
            pred_all = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
            coord = batch["coord"].detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy().reshape(-1)
            offs = batch["offset"].detach().cpu().numpy().reshape(-1)
            starts = np.concatenate([[0], offs[:-1]])
            for b in range(pooled.shape[0]):
                s, e = int(starts[b]), int(offs[b])
                if e - s < 64:
                    continue
                sel = _subsample(e - s, args.rl_max_points, rng) + s
                scenes.append({"pooled": pooled[b:b + 1], "pred_stats": pstats[b:b + 1],
                               "xyz": coord[sel].astype(np.float64),
                               "pred": pred_all[sel].astype(np.int64),
                               "gt": labels[sel].astype(np.int64)})
        except Exception as ex:
            print(f"[rl]  scene {idx} skipped: {ex}")
        if (j + 1) % 10 == 0:
            print(f"[rl]   cached {len(scenes)} clouds from {j + 1}/{len(pick)} tiles")

    if not scenes:
        sys.exit("[rl] no scenes cached -- check --tiles / --split / --ckpt")
    print(f"[rl] cached {len(scenes)} clouds; REINFORCE: {args.rl_iters} iters x batch {args.rl_batch}, "
          f"sigma={args.rl_sigma}, lr={args.rl_lr}, rmse_res={res} m")

    out = args.out or os.path.join(os.path.dirname(args.ckpt) or ".", "model_rl.pt")
    best, ema = float("inf"), None
    for it in range(1, args.rl_iters + 1):
        sel = rng.choice(len(scenes), size=min(args.rl_batch, len(scenes)), replace=False)
        m = reinforce_update(model, [scenes[k] for k in sel], opt, cfg, sigma=args.rl_sigma, res=res, metric=args.reward)
        if np.isfinite(m["rmse_base"]):
            ema = m["rmse_base"] if ema is None else 0.9 * ema + 0.1 * m["rmse_base"]
            if ema < best:
                best = ema
                torch.save(model.state_dict(), out)
        if it == 1 or it % 10 == 0:
            print(f"[rl] it {it:4d}/{args.rl_iters}  greedy_RMSE={m['rmse_base']:.3f}m  "
                  f"sample_RMSE={m['rmse_sample']:.3f}m  {args.reward}: g={m['score_base']:.3f} "
                  f"s={m['score_sample']:.3f} adv={m['advantage']:+.4f}  "
                  f"reclass={m['reclass_frac'] * 100:4.1f}%  ema={(ema if ema else float('nan')):.3f}")
    print(f"[rl] done. best greedy-RMSE (ema)={best:.3f} m -> saved to {out}")
    print(f"[rl] inference: use --ckpt {out}; the regime head now predicts RMSE-optimal SPAG-DC globals.")


if __name__ == "__main__":
    main()
