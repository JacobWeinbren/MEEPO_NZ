#!/usr/bin/env python3
"""Measure peak VRAM for ONE forward+backward(+optimizer step) of the MEEPO model at a given
point-count and checkpoint granularity -- so you can see whether a target block fits a small
card (e.g. 16 GB) BEFORE committing to an 80-epoch run.

Two modes:
  * --tiles data/nz/tiles  : crop one REAL block (raster branch included) -> exact number.
  * --synthetic            : fabricate a block (raster off) -> portable, backbone-only, a
                             slight under-estimate but a clean per-granularity comparison.

Sweeps granularities by default (none/stage/block/layer) and prints peak allocated & reserved
(reserved ~ what nvidia-smi shows and what triggers OOM). Example:

  python3 scripts/probe_vram.py --tiles data/nz/tiles --points 250000 --amp bf16
  python3 scripts/probe_vram.py --synthetic --points 250000 --granularity all
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meepo_nz.utils.config import Config
from meepo_nz.models import build_meepo


def _synthetic_batch(cfg, n, device):
    dl = float(getattr(cfg, "first_subsampling_dl", 0.1))
    fdim = int(getattr(cfg, "in_features_dim", 7))
    rng = np.random.default_rng(0)
    span = max(dl * (n ** (1 / 3)) * 1.5, 50.0)                    # rough block extent (m)
    coord = torch.from_numpy((rng.random((n, 3)) * span).astype(np.float32))
    grid = torch.floor((coord - coord.min(0).values) / dl).int()
    feat = torch.from_numpy(rng.standard_normal((n, fdim)).astype(np.float32))
    labels = torch.from_numpy(rng.integers(0, 2, n).astype(np.int64))
    batch = {"coord": coord, "grid_coord": grid, "feat": feat,
             "offset": torch.tensor([n], dtype=torch.long),
             "labels": labels, "cloud_lengths_0": torch.tensor([n], dtype=torch.long)}
    return {k: v.to(device) for k, v in batch.items()}


def _real_batch(cfg, n, device):
    from meepo_nz.data.scene_dataset import SceneDataset
    from meepo_nz.data.ptv3_collate import PTv3Collate
    cfg.scene_max_points = int(n)                                  # cap the crop to n points
    ds = SceneDataset(cfg._tiles, cfg, split="train")
    collate = PTv3Collate(cfg, device=None, mix_prob=0.0)
    batch = collate([ds[np.random.default_rng(0).integers(0, len(ds))]])
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _one_step(model, batch, opt, amp_dtype):
    model.train()
    ctx = torch.autocast("cuda", dtype=amp_dtype) if (amp_dtype and batch["coord"].is_cuda) else torch.autocast("cpu", enabled=False)
    with ctx:
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits.float(), batch["labels"])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.detach().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=None, help="use a real cropped block (raster included)")
    ap.add_argument("--synthetic", action="store_true", help="fabricate a block (raster off, portable)")
    ap.add_argument("--points", type=int, default=250000)
    ap.add_argument("--granularity", default="all", choices=["all", "none", "stage", "block", "layer"])
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--dl", type=float, default=None)
    ap.add_argument("--iters", type=int, default=3, help="steps per config; peak taken over all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default=None, choices=["meepo", "meepo3", "pointssm", "vm3"],
                    help="Probe a specific backbone (default: cfg default).")
    ap.add_argument("--ssm-backend", default=None, choices=["auto", "cuda", "ssd", "torch", "triton-ssd"])
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.amp]

    if args.tiles and not args.synthetic:
        cfg_path = os.path.join(args.tiles, "config.used.yaml")
        cfg = Config.load(cfg_path) if os.path.exists(cfg_path) else Config()
        cfg._tiles = args.tiles
        use_real = True
    else:
        cfg = Config()
        cfg.use_dtm_raster = False                                # synthetic block has no raster
        use_real = False
    cfg.scene_mode = True
    cfg.spag_learned = False                                      # isolate the backbone cost
    if args.backbone is not None:
        cfg.backbone = args.backbone
    if args.ssm_backend is not None:
        cfg.ssm_backend = args.ssm_backend
    if args.dl is not None:
        cfg.first_subsampling_dl = float(args.dl)
    from meepo_nz.features.shallow_features import expected_feature_dim
    cfg.in_features_dim = int(expected_feature_dim(cfg))          # true per-point feature count

    grans = ["none", "stage", "block", "layer"] if args.granularity == "all" else [args.granularity]
    print(f"[probe] {'REAL block from '+args.tiles if use_real else 'SYNTHETIC block (raster off)'} | "
          f"points={args.points} amp={args.amp} device={device}")
    if device.type != "cuda":
        print("[probe] NOTE: no CUDA -> reporting fwd/bwd correctness only (no VRAM numbers)")

    batch = _real_batch(cfg, args.points, device) if use_real else _synthetic_batch(cfg, args.points, device)
    npts = int(batch["offset"][-1].item())
    print(f"[probe] block has {npts:,} points (in_features_dim={int(getattr(cfg,'in_features_dim',7))})\n")

    rows = []
    for g in grans:
        cfg.grad_checkpointing = (g != "none")
        cfg.checkpoint_granularity = g if g != "none" else "block"
        torch.manual_seed(0)
        model = build_meepo(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        loss = float("nan")
        try:
            for _ in range(max(1, args.iters)):
                loss = _one_step(model, batch, opt, amp_dtype)
            if device.type == "cuda":
                torch.cuda.synchronize()
                alloc = torch.cuda.max_memory_allocated() / 2**30
                resv = torch.cuda.max_memory_reserved() / 2**30
                rows.append((g, f"{alloc:6.2f}", f"{resv:6.2f}", "ok", f"{loss:.3f}"))
                print(f"  {g:6s}: peak alloc={alloc:6.2f} GB   reserved={resv:6.2f} GB   loss={loss:.3f}")
            else:
                fin = np.isfinite(loss)
                rows.append((g, "-", "-", "ok" if fin else "BADLOSS", f"{loss:.3f}"))
                print(f"  {g:6s}: fwd+bwd {'OK' if fin else 'NON-FINITE'}   loss={loss:.3f}")
        except RuntimeError as e:
            oom = "out of memory" in str(e).lower()
            rows.append((g, "-", "-", "OOM" if oom else "ERR", "-"))
            print(f"  {g:6s}: {'OUT OF MEMORY' if oom else 'ERROR: '+str(e)[:80]}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cuda":
        print("\n[probe] 'reserved' is what nvidia-smi shows and what OOMs. For a 16 GB card, "
              "aim for reserved comfortably under ~15 GB on the WORST (densest) tile, not just this one.")


if __name__ == "__main__":
    main()
