#!/usr/bin/env python3
"""
05 - Train MEEPO (PTv3 + sparse MoE) on the preprocessed New Zealand tiles.

Builds the train / val / test ``SphereDataset`` splits, the PTv3 collate, the
MEEPO model (clean PyTorch: no spconv / flash-attn / torch_scatter), and runs
the SAME ``Trainer`` as the original pipeline - so the progress bars, per-epoch
error images, classified-LAZ gallery, and training dashboard are unchanged.

    python scripts/05_train.py --tiles data/nz/tiles --epochs 100
    python scripts/05_train.py --tiles data/nz/tiles --config configs/default.yaml
    # ablations:
    python scripts/05_train.py --tiles data/nz/tiles --no-dtm-raster --name pm_nora
    python scripts/05_train.py --tiles data/nz/tiles --num-experts 4 --moe-topk 1
"""
from __future__ import annotations

import argparse
import os
import sys
# Reduce CUDA fragmentation (the allocator suggests this on OOM); must be set before
# torch initialises CUDA. setdefault so an explicit user value still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meepo_nz.utils.config import Config
from meepo_nz.models import build_meepo
from meepo_nz.data.dataset import SphereDataset
from meepo_nz.data.ptv3_collate import PTv3Collate
from meepo_nz.training.trainer import Trainer


def main():
    ap = argparse.ArgumentParser(description="Train MEEPO for ground extraction.")
    ap.add_argument("--tiles", required=True, help="Preprocessed tile dir (step 04).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--epoch-steps", type=int, default=None,
                    help="Optimiser steps per epoch (KPConv fixed-step epoch; the config default was "
                         "sized for the ~2800-tile NZ corpus). 0 = ONE FULL PASS over all train tiles "
                         "per epoch -- use this for small corpora (with more --epochs) to avoid "
                         "recycling the same tiles hundreds of times per epoch at full LR.")
    ap.add_argument("--val-size", type=int, default=None,
                    help="Validation steps per epoch (default 50; 0 = full validation pass).")
    ap.add_argument("--log-every", type=int, default=None,
                    help="Print a running loss/lr/throughput/eta line every N optimiser steps "
                         "(default: ~10 logs/epoch). First and last step of each epoch always log.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--batch-num", type=int, default=None)
    ap.add_argument("--norm", choices=["bn", "ln"], default=None,
                    help="Backbone (+raster CNN) normalization. 'bn' = BatchNorm (MEEPO Tab.3c "
                         "default). 'ln' = LayerNorm (backbone) + GroupNorm (raster), both "
                         "batch-independent -> required for micro-batch 1 / heavy grad-accum, where "
                         "BN stats would otherwise be computed from a single scene per forward.")
    ap.add_argument("--accum-steps", type=int, default=None,
                    help="Gradient accumulation: micro-batches per optimiser step. Effective batch = "
                         "batch_num * accum_steps. Use --batch-num 4 --accum-steps 4 to train at the "
                         "paper's batch 16 when 16 blocks OOM on one GPU (= the paper's cross-GPU averaging).")
    ap.add_argument("--sphere-mode", action="store_true",
                    help="Use the LEGACY KPConv sphere dataset instead of full-scene PTv3.")
    ap.add_argument("--scene-mode", action="store_true",
                    help="Force PTv3-native whole-scene training (the default). Use to override a stale "
                         "data/<tiles>/config.used.yaml snapshot still set to scene_mode=false, without re-running stage 04.")
    ap.add_argument("--first-subsampling-dl", "--dl", dest="dl", type=float, default=None,
                    help="Finest voxel grid in metres (default 0.05). Changing this -> re-run stage 04.")
    ap.add_argument("--scene-block-size", type=float, default=None,
                    help="Side length (m) of a large block when a tile exceeds scene-max-points.")
    ap.add_argument("--scene-max-points", type=int, default=None,
                    help="Hard cap on points per forward pass (raise on big GPUs, lower on OOM).")
    ap.add_argument("--scene-vote-step", type=float, default=None,
                    help="SparseGF soft-voting grid step (m) between disc centres at val/test/inference "
                         "(default 50). Each disc classifies its circular central region of radius "
                         "(sqrt2/2)*step; overlapping regions are soft-voted (averaged softmax).")
    ap.add_argument("--scene-val-tiles", type=int, default=None,
                    help="How many val tiles to score per epoch with the deployed full-res "
                         "fragment inference (predict_scene), matching the reference SemSegTester. "
                         "Fixed evenly-spaced subset. Default 32; lower if eval is slow, 0 = all tiles. "
                         "The final test always scores the full split.")
    ap.add_argument("--n-vis-tiles", type=int, default=None,
                    help="How many validation areas to render in the per-epoch gallery (default 12).")
    ap.add_argument("--refine", choices=["spag_dc", "off"], default=None,
                    help="Post-classification ground refinement. 'spag_dc' (default) = the SPAG-DC "
                         "misclassification corrector (IEEE Sensors 2025); 'off' = raw argmax.")
    ap.add_argument("--tta", action="store_true",
                    help="Test-time augmentation at the FINAL test: average softmax over z-rotations "
                         "0/90/180/270 (Pointcept SemSegTester). Rotates cloud + prior raster together. "
                         "Not applied to per-epoch validation.")
    ap.add_argument("--spag-alpha", type=float, default=None,
                    help="SPAG-DC region-growing curvature coeff alpha (tau_c=alpha*mu_c). Paper 0.5 (recommend 0.5-0.7).")
    ap.add_argument("--spag-beta", type=float, default=None,
                    help="SPAG-DC angle-relaxation coeff beta (theta_d=theta0*(1+beta*var_c)), in [0,1). Paper 0.7 (recommend 0.6-0.8).")
    ap.add_argument("--spag-theta0", type=float, default=None,
                    help="SPAG-DC initial normal-angle threshold theta0 in degrees. Paper 10.")
    ap.add_argument("--spag-k", type=int, default=None,
                    help="SPAG-DC region-growing kNN neighbourhood. Paper ~20-25.")
    ap.add_argument("--spag-n-sigma", type=float, default=None,
                    help="SPAG-DC correction cut: residual > mu2 + n*sigma2 -> non-ground. Paper-consistent n=3 (3-sigma normal-tail).")
    ap.add_argument("--spag-learned", dest="spag_learned", action="store_true", default=None,
                    help="LEARNED SPAG-DC: train a per-scene regime head that predicts the SPAG-DC globals "
                         "(theta0/alpha/beta/n_sigma/base_res/min_floor) from backbone features, supervised by a "
                         "GT-ground terrain oracle; predicted globals drive the corrector at inference (default OFF = geometry).")
    ap.add_argument("--no-spag-learned", dest="spag_learned", action="store_false",
                    help="Force geometry-only SPAG-DC (fixed cfg globals).")
    ap.add_argument("--spag-regime-weight", type=float, default=None,
                    help="Weight of the learned-SPAG-DC regime smooth-L1 auxiliary on the total loss (default 0.1).")
    ap.add_argument("--spag-rl", dest="spag_rl", action="store_true", default=None,
                    help="Co-train the regime head on DTM-RMSE via self-critical REINFORCE (implies --spag-learned). "
                         "Replaces the oracle smooth-L1: the head learns SPAG-DC globals that MINIMISE DTM-RMSE-vs-GT "
                         "(penalises cliff destruction, no slope rule). Strided + subsampled so it barely slows training.")
    ap.add_argument("--spag-rl-every", type=int, default=None,
                    help="Run the REINFORCE update every N micro-steps (default 10). Lower = more head updates, more CPU.")
    ap.add_argument("--spag-rl-weight", type=float, default=None,
                    help="Weight of the REINFORCE term on the total loss (default 1.0).")
    ap.add_argument("--spag-rl-sigma", type=float, default=None,
                    help="Policy std in logit space for global sampling (default 0.5). Higher = more exploration.")
    ap.add_argument("--spag-rl-max-points", type=int, default=None,
                    help="Per-cloud subsample for the corrector reward (default 30000; 0 = ALL). Corrector cost is "
                         "~linear in points (~0.7s @20k, ~3s @80k, ~8s @200k per call), and the 1 m DEM is "
                         "resolution-limited, so a 20-40k cap is far faster with no accuracy cost.")
    ap.add_argument("--spag-rl-res", type=float, default=None,
                    help="DTM-RMSE grid resolution (m) for the RL reward (default cfg.dtm_rmse_res or 1.0).")
    ap.add_argument("--spag-rl-reward", choices=["rmse", "p95", "p99", "max"], default=None,
                    help="Reward aggregation over per-pixel DEM error (default p95). Tail metrics (p95/p99/max) "
                         "weight the worst cells (cliffs/edges) far more than mean rmse, countering flat-area dilution.")
    ap.add_argument("--spag-rl-eval-every", type=int, default=None,
                    help="Held-out cliff-set eval every N OPTIMISER steps (default 200; 0 disables). Logs greedy-params "
                         "DTM-RMSE vs no-refine on a fixed cliff-heavy val subset -- the real head-progress signal.")
    ap.add_argument("--spag-rl-eval-tiles", type=int, default=None,
                    help="Number of cliff-heavy val tiles in the held-out eval set (default 12).")
    ap.add_argument("--spag-rl-eval-max-points", type=int, default=None,
                    help="Per-cloud subsample cap for the held-out eval (default 80000; 0 = all). The 1 m DEM is "
                         "resolution-limited so this is accuracy-safe and keeps the eval fast.")
    ap.add_argument("--fixed-batch", action="store_true",
                    help="Disable variable point-budget batching; use a fixed batch_num spheres.")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="Disable PTv3 activation checkpointing (uses more VRAM, somewhat faster).")
    ap.add_argument("--checkpoint-granularity", choices=["stage", "block", "layer"], default=None,
                    help="Recompute granularity when checkpointing is on (default block). 'layer' recomputes each "
                         "block's xCPE/Mamba/MLP separately = lowest activation peak, for small-VRAM cards (e.g. 16 GB); "
                         "'stage' recomputes a whole stage at once = least recompute but highest peak (needs headroom).")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="Force activation checkpointing ON in sphere mode (only needed for very large in_radius).")
    ap.add_argument("--batch-limit", type=int, default=None,
                    help="Total input points per variable batch (0 = auto-calibrate).")
    ap.add_argument("--mix-prob", type=float, default=None,
                    help="MEEPO Mix3D probability (default 0.8): merge clouds pairwise per train batch. 0 disables.")
    ap.add_argument("--no-mix3d", action="store_true", help="Disable Mix3D (sets mix_prob=0).")
    ap.add_argument("--no-dropout", action="store_true",
                    help="Disable RandomDropout (sets augment_dropout_prob=0). [default ON = MEEPO RandomDropout(0.2,0.2)]")
    ap.add_argument("--augment-tilt", action="store_true",
                    help="Enable MEEPO x/y micro-tilt (sets augment_tilt_xy=pi/64). OFF by default: desyncs the 2D georeferenced prior.")
    ap.add_argument("--augment-elastic", action="store_true",
                    help="Enable MEEPO ElasticDistortion. OFF by default: warps the ground surface (~1.6 m) and desyncs the prior.")
    ap.add_argument("--scene-cache-tiles", type=int, default=None,
                    help="Per-worker LRU tile cache size (default 4). Lower (1-2) to cut host RAM; each cached tile is large.")
    ap.add_argument("--prefetch-factor", type=int, default=None,
                    help="DataLoader batches prefetched per worker (default 2). Set 1 to roughly halve the prefetch-buffer RAM.")
    ap.add_argument("--no-lovasz", action="store_true", help="CE only (drop the Lovasz term). [default: Lovasz ON = the MEEPO loss, CE + Lovasz]")
    ap.add_argument("--lovasz", action="store_true", help="Enable the Lovasz-softmax term (ON by default; this IS the MEEPO loss).")
    ap.add_argument("--groundiff-regression", dest="groundiff_regression", action="store_true", default=None,
                    help="Enable GrounDiff nDSM regression L1+L2 (ON by default; prevents majority-class collapse). [Dhaouadi et al. 2025]")
    ap.add_argument("--no-groundiff-regression", dest="groundiff_regression", action="store_false",
                    help="Disable GrounDiff regression -> pure MEEPO classification (WILL collapse to all-ground on imbalanced binary).")
    ap.add_argument("--groundiff-l1", dest="groundiff_l1", type=float, default=None, help="GrounDiff lambda1 weight on L1 nDSM loss (default 1.0).")
    ap.add_argument("--groundiff-l2", dest="groundiff_l2", type=float, default=None, help="GrounDiff lambda2 weight on L2 nDSM loss (default 1.0).")
    ap.add_argument("--groundiff-cls-weight", dest="groundiff_cls_weight", type=float, default=None,
                    help="Weight on the CE(+Lovasz) mask term (default 1.0; set 0.1 for strict-GrounDiff weighting).")
    ap.add_argument("--ndsm-scale", dest="ndsm_scale", type=float, default=None, help="Normalise the nDSM target by this many metres so L1/L2 are O(1) (default 10).")
    ap.add_argument("--block-lr-scale", type=float, default=None, help="Backbone LR multiplier (MEEPO uses 0.1; 1.0 disables).")
    ap.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay (paper outdoor 5e-3).")
    ap.add_argument("--warmup-epochs", type=int, default=None, help="Linear LR warmup epochs before cosine.")
    ap.add_argument("--no-intensity-log", action="store_true", help="Skip log1p on intensity (plain z-score).")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    # feature switches (Deviation B lives in these channels)
    ap.add_argument("--no-mean-elev", action="store_true")
    ap.add_argument("--no-curvature", action="store_true")
    ap.add_argument("--no-moments", action="store_true")
    ap.add_argument("--no-return-features", action="store_true",
                    help="disable the laser-return count channel")
    ap.add_argument("--no-return-ratio", action="store_true",
                    help="disable the normalised return-ratio channel")
    ap.add_argument("--no-intensity", action="store_true",
                    help="disable the per-point intensity channel")
    ap.add_argument("--no-xyz-feature", action="store_true",
                    help="drop the xyz coordinate input channels")
    # Deviation A (prior-year classification raster branch)
    ap.add_argument("--no-dtm-raster", action="store_true",
                    help="Disable the previous-year-classification raster branch (Deviation A).")
    ap.add_argument("--prev-dtm-scalar", action="store_true",
                    help="Also add the legacy per-point z - prior_DTM scalar channel.")
    ap.add_argument("--no-prev-dtm", action="store_true")
    ap.add_argument("--no-raster-gating", action="store_true",
                    help="Disable the GrounDiff-style confidence gating head in the raster branch.")
    ap.add_argument("--no-augment", action="store_true")
    # MEEPO knobs
    ap.add_argument("--backbone", choices=["meepo"], default="meepo",
                    help="Backbone: meepo (CNN-Mamba; the only backbone -- PTv3/LitePT+MoE were stripped).")
    ap.add_argument("--ssm-backend", choices=["auto", "cuda", "ssd", "torch"], default=None,
                    help="MEEPO selective-scan backend: auto=fused mamba_ssm kernel if importable else pure-torch (default); cuda=require the kernel; torch=force pure-torch (exact, slower).")
    ap.add_argument("--no-moe", action="store_true", help="Disable MoE (dense PTv3).")
    ap.add_argument("--num-experts", type=int, default=None)
    ap.add_argument("--moe-topk", type=int, default=None)
    ap.add_argument("--moe-aux-alpha", type=float, default=None,
                    help="Load-balancing aux-loss weight (paper best = 0; aux loss hurts).")
    ap.add_argument("--class-balance", choices=["none", "inverse"], default=None)
    ap.add_argument("--lr-schedule", choices=["onecycle", "cosine", "exp", "onecycle_kpx"], default=None,
                    help="LR schedule. 'onecycle' = MEEPO OneCycleLR (default in config). 'cosine'/'exp'/'onecycle_kpx'.")
    ap.add_argument("--compile", action="store_true",
                    help="Wrap the model in torch.compile (CUDA forward speedup).")
    args = ap.parse_args()

    cfg_path = args.config
    if cfg_path is None:
        auto_cfg = os.path.join(args.tiles, "config.used.yaml")
        if os.path.exists(auto_cfg):
            cfg_path = auto_cfg
            print(f"[05] loading resolved config from {auto_cfg}")
            print("[05] NOTE: this snapshot was written by stage 04 and OVERRIDES configs/default.yaml. "
                  "If it predates recent changes (scene_max_points, loss_lovasz, ...), either re-run "
                  "scripts/04_preprocess.py to refresh it or override on the CLI (e.g. --scene-max-points, --no-lovasz).")
    cfg = Config.load(cfg_path) if cfg_path else Config()

    # The stage-04 snapshot (config.used.yaml) must pin DATA-LAYOUT params (dl, in_radius,
    # feature flags, block size, experts) so the model matches how the tiles were built -- but
    # it must NOT freeze TRAINING hyperparameters, or a stale snapshot silently overrides the
    # current recipe (exactly how cosine + wd 0.005 + block lr x0.1 came back). So when
    # we auto-loaded the snapshot, take the optimizer / LR-schedule / loss / refine settings from
    # the live config.py defaults instead; the per-flag CLI overrides below still win. (Skipped
    # when you pass an explicit --config.)
    _used_snapshot = (args.config is None and cfg_path is not None)
    if _used_snapshot:
        _live = Config()
        for _hp in ("optimizer", "adamw_lr", "adamw_weight_decay", "adamw_betas", "adamw_eps",
                    "block_lr_scale", "warmup_epochs", "lr_schedule", "onecycle_pct_start",
                    "onecycle_div_factor", "onecycle_final_div_factor", "kpx_lr_start",
                    "kpx_lr_max", "kpx_lr_warmup_epochs", "kpx_lr_plateau_epochs",
                    "kpx_lr_decay10_epochs", "lr_decay", "grad_clip_norm", "loss_lovasz",
                    "lovasz_weight", "use_groundiff_regression",
                    "groundiff_l1_weight", "groundiff_l2_weight", "groundiff_cls_weight",
                    "ndsm_scale", "ndsm_dtm_res", "ndsm_min_ground",
                    "refine_method", "spag_theta0_deg",
                    "spag_alpha", "spag_beta", "spag_k", "spag_n_sigma", "spag_min_grid_diff",
                    "spag_learned", "spag_regime_weight", "spag_regime_hidden",
                    "spag_tps_kmin", "spag_tps_kmax"):
            if hasattr(_live, _hp):
                setattr(cfg, _hp, getattr(_live, _hp))
        print("[05] training hyperparameters (optimizer / LR schedule / loss / refine) taken from "
              "current config.py, NOT the stale stage-04 snapshot (data-layout params kept from it). "
              "Override per-flag on the CLI, or pass --config to use a file verbatim.")
    if args.epochs is not None:      cfg.epochs = args.epochs
    if args.epoch_steps is not None: cfg.epoch_steps = args.epoch_steps
    if getattr(args, "log_every", None) is not None: cfg.log_every_steps = args.log_every
    if args.val_size is not None:    cfg.validation_size = args.val_size
    if args.out_dir is not None:     cfg.out_dir = args.out_dir
    if args.name is not None:        cfg.name = args.name
    if args.batch_num is not None:   cfg.batch_num = args.batch_num
    if args.norm is not None:        cfg.norm = args.norm
    if args.accum_steps is not None: cfg.grad_accum_steps = args.accum_steps
    if args.sphere_mode:             cfg.scene_mode = False
    if args.scene_mode:              cfg.scene_mode = True   # override a stale config.used.yaml snapshot
    if bool(getattr(cfg, "scene_mode", True)):
        # The prior-raster branch (Deviation A) is INTEGRATED into whole-scene mode: the prior
        # is cropped per block and run through the GrounDiff CNN inside the forward, training
        # end-to-end. It stays ON by default; pass --no-dtm-raster to disable it.
        # Scene mode feeds one whole tile / large block per item. The KPConv sphere-style
        # variable-batch (a total-POINTS budget) over-packs and OOMs, so use plain batching:
        # batch_num clouds per forward (config default 6 x accum 2 = effective 12 = the MEEPO
        # total batch_size; 6 x scene_max_points=204800 ~= 47 GB on the 96 GB card + grad checkpointing).
        # Lower --batch-num (raise --accum-steps to keep effective 12) if OOM on a smaller card.
        cfg.variable_batch = False
        # MEEPO is ITERATION-budgeted, not full-pass-per-epoch, and that budget is
        # dataset-size INDEPENDENT: indoor+outdoor = 180k optimiser iters over 120 epochs =
        # 1500 steps/epoch (indoor = 140k = ~1167). Each step randomly samples batch_num blocks
        # from the corpus (RandomSampler with replacement), so this is correct whether you have
        # 3k tiles or 300k. A full pass (epoch_steps=0) would instead scale with your dataset
        # and over/under-shoot the paper. Default to the paper budget; --epoch-steps overrides.
        if args.epoch_steps is None:
            cfg.epoch_steps = 1500
        if args.epochs is None:
            cfg.epochs = 120
        # The paper batch_size=16 of 204800-pt blocks OOMs a single 96 GB card, so default to the
        # same EFFECTIVE batch via gradient accumulation (4 blocks/forward x 4 = 16, ~25 GB) unless
        # batch/accum are set explicitly. This is the paper's cross-GPU averaging done on one device.
        if args.batch_num is None and args.accum_steps is None:
            cfg.batch_num = 4
            cfg.grad_accum_steps = 4
        # Whole-scene samples are HEAVY (a full tile loaded per __getitem__), unlike sphere mode's
        # tiny crops. 16 forked workers each holding a tile cache + pinned prefetch buffers exhaust
        # host RAM -> swap -> the GPU starves (3-4k pts/s, CPU ~2%). Default to a handful of workers
        # in scene mode (CPU is not the bottleneck here); raise with --num-workers if RAM is flat.
        if args.num_workers is None:
            cfg.num_workers = 6
    else:
        # sphere mode: spheres are small, so activation checkpointing is pure overhead
        # (~20-30% slower, no memory benefit). Off unless explicitly forced.
        if not args.grad_checkpoint:
            cfg.grad_checkpointing = False
    if args.dl is not None:          cfg.first_subsampling_dl = float(args.dl)
    if args.scene_block_size is not None:  cfg.scene_block_size = float(args.scene_block_size)
    if args.scene_max_points is not None:  cfg.scene_max_points = int(args.scene_max_points)
    if args.scene_vote_step is not None:   cfg.scene_vote_step_m = float(args.scene_vote_step)
    if args.scene_val_tiles is not None:   cfg.scene_val_tiles = int(args.scene_val_tiles)
    if args.n_vis_tiles is not None:       cfg.n_vis_tiles = int(args.n_vis_tiles)
    if args.refine is not None:            cfg.refine_method = args.refine
    if args.tta:                           cfg.tta = True
    if args.spag_alpha is not None:        cfg.spag_alpha = float(args.spag_alpha)
    if args.spag_beta is not None:         cfg.spag_beta = float(args.spag_beta)
    if args.spag_theta0 is not None:       cfg.spag_theta0_deg = float(args.spag_theta0)
    if args.spag_k is not None:            cfg.spag_k = int(args.spag_k)
    if args.spag_n_sigma is not None:      cfg.spag_n_sigma = float(args.spag_n_sigma)
    if getattr(args, "spag_learned", None) is not None:  cfg.spag_learned = bool(args.spag_learned)
    if getattr(args, "spag_regime_weight", None) is not None: cfg.spag_regime_weight = float(args.spag_regime_weight)
    if getattr(args, "spag_rl", None):
        cfg.spag_rl = True
        cfg.spag_learned = True                               # RL trains the regime head; it must exist
    if getattr(args, "spag_rl_every", None) is not None:       cfg.spag_rl_every = int(args.spag_rl_every)
    if getattr(args, "spag_rl_weight", None) is not None:      cfg.spag_rl_weight = float(args.spag_rl_weight)
    if getattr(args, "spag_rl_sigma", None) is not None:       cfg.spag_rl_sigma = float(args.spag_rl_sigma)
    if getattr(args, "spag_rl_max_points", None) is not None:  cfg.spag_rl_max_points = int(args.spag_rl_max_points)
    if getattr(args, "spag_rl_res", None) is not None:         cfg.spag_rl_res = float(args.spag_rl_res)
    if getattr(args, "spag_rl_reward", None) is not None:      cfg.spag_rl_reward = str(args.spag_rl_reward)
    if getattr(args, "spag_rl_eval_every", None) is not None:  cfg.spag_rl_eval_every = int(args.spag_rl_eval_every)
    if getattr(args, "spag_rl_eval_tiles", None) is not None:  cfg.spag_rl_eval_tiles = int(args.spag_rl_eval_tiles)
    if getattr(args, "spag_rl_eval_max_points", None) is not None: cfg.spag_rl_eval_max_points = int(args.spag_rl_eval_max_points)
    if args.fixed_batch:             cfg.variable_batch = False
    if args.no_grad_checkpoint:      cfg.grad_checkpointing = False
    if getattr(args, "checkpoint_granularity", None) is not None:
        cfg.checkpoint_granularity = str(args.checkpoint_granularity)
    if args.batch_limit is not None: cfg.batch_limit = args.batch_limit
    if args.mix_prob is not None:    cfg.mix_prob = float(args.mix_prob)
    if args.no_mix3d:                cfg.mix_prob = 0.0
    if args.no_dropout:              cfg.augment_dropout_prob = 0.0
    if args.augment_tilt:            cfg.augment_tilt_xy = 0.04908738521234052   # MEEPO RandomRotate x/y = pi/64
    if args.augment_elastic:         cfg.augment_elastic = True
    if args.scene_cache_tiles is not None: cfg.scene_cache_tiles = int(args.scene_cache_tiles)
    if args.prefetch_factor is not None:   cfg.dataloader_prefetch = int(args.prefetch_factor)
    if bool(getattr(cfg, "use_augmentation", True)):
        print(f"[aug] z-rot scale[{cfg.augment_scale_min:g},{cfg.augment_scale_max:g}](iso) flip-xy "
              f"jitter(sigma={cfg.augment_noise:g},clip={cfg.augment_noise_clip:g}) "
              f"dropout({cfg.augment_dropout_ratio:g}@p{cfg.augment_dropout_prob:g}) "
              f"tilt_xy={cfg.augment_tilt_xy:g}rad elastic={bool(cfg.augment_elastic)} | Mix3D p={float(getattr(cfg,'mix_prob',0.0)):g}  [MEEPO recipe]")
    if bool(getattr(cfg, "scene_mode", True)):
        print(f"[05] scene cfg: scene_max_points={int(getattr(cfg,'scene_max_points',600000)):,} "
              f"block_size={float(getattr(cfg,'scene_block_size',64.0)):g}m "
              f"dl={float(getattr(cfg,'first_subsampling_dl',0.1)):g} "
              f"variable_batch={bool(getattr(cfg,'variable_batch',False))} "
              f"grad_checkpointing={bool(getattr(cfg,'grad_checkpointing',True))} "
              f"(per-forward points are capped by scene_max_points; lower it if OOM)")
    if args.lr is not None:          cfg.learning_rate = args.lr
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.no_mean_elev:    cfg.use_mean_elevation = False
    if args.no_curvature:    cfg.use_curvature = False
    if args.no_moments:      cfg.use_higher_moments = False
    if args.no_return_features: cfg.use_return_features = False
    if args.no_return_ratio: cfg.use_return_ratio = False
    if args.no_intensity:    cfg.use_intensity = False
    if args.no_xyz_feature:  cfg.use_xyz_in_features = False
    if args.no_dtm_raster:   cfg.use_dtm_raster = False
    if args.prev_dtm_scalar: cfg.use_prev_dtm = True
    if args.no_prev_dtm:     cfg.use_prev_dtm = False
    if args.no_raster_gating: cfg.prior_raster_gating = False
    if args.no_augment:      cfg.use_augmentation = False
    if args.backbone is not None:
        cfg.backbone = args.backbone
    if args.ssm_backend is not None:
        cfg.ssm_backend = args.ssm_backend
    if args.no_lovasz:
        cfg.loss_lovasz = False
    if args.lovasz:
        cfg.loss_lovasz = True
    if getattr(args, "groundiff_regression", None) is not None:
        cfg.use_groundiff_regression = bool(args.groundiff_regression)
    if getattr(args, "groundiff_l1", None) is not None:
        cfg.groundiff_l1_weight = float(args.groundiff_l1)
    if getattr(args, "groundiff_l2", None) is not None:
        cfg.groundiff_l2_weight = float(args.groundiff_l2)
    if getattr(args, "groundiff_cls_weight", None) is not None:
        cfg.groundiff_cls_weight = float(args.groundiff_cls_weight)
    if getattr(args, "ndsm_scale", None) is not None:
        cfg.ndsm_scale = float(args.ndsm_scale)
    # SPAG-DC ground-misclassification corrector: echo the active config.
    if str(getattr(cfg, "refine_method", "spag_dc")).lower() not in ("off", "none", ""):
        print(f"[refine] SPAG-DC corrector ENABLED (IEEE Sensors 2025): theta0={cfg.spag_theta0_deg:g}deg "
              f"alpha={cfg.spag_alpha:g} beta={cfg.spag_beta:g} k={cfg.spag_k} n_sigma={cfg.spag_n_sigma:g} "
              f"min_grid_diff={cfg.spag_min_grid_diff:g}m TPS_k=[{cfg.spag_tps_kmin},{cfg.spag_tps_kmax}]. "
              f"Region-growing core -> adaptive seed grid -> MCS purification -> local-TPS surface; reclassifies "
              f"predicted-ground points whose residual to the surface exceeds mu2+n*sigma2 (deterministic, "
              f"geometry-only). Watch the first validation line for the raw->refined RMSE/IoU delta.")
    else:
        print("[refine] SPAG-DC DISABLED (--refine off): deployed labels are the raw argmax.")
    if args.block_lr_scale is not None:
        cfg.block_lr_scale = args.block_lr_scale
    if args.weight_decay is not None:
        cfg.adamw_weight_decay = args.weight_decay
    if args.warmup_epochs is not None:
        cfg.warmup_epochs = args.warmup_epochs
    if args.no_intensity_log:
        cfg.intensity_log = False
    if args.no_moe:          cfg.use_moe = False
    if args.num_experts is not None: cfg.num_experts = args.num_experts
    if args.moe_topk is not None:    cfg.moe_topk = args.moe_topk
    if args.moe_aux_alpha is not None: cfg.moe_aux_loss_alpha = args.moe_aux_alpha
    if args.class_balance is not None: cfg.loss_class_balance = args.class_balance
    if args.lr_schedule is not None: cfg.lr_schedule = args.lr_schedule

    # in_features_dim is implied by the active feature switches; build_meepo
    # recomputes it too, but set it here so prints / norm_stats match.
    from meepo_nz.features.shallow_features import expected_feature_dim
    cfg.in_features_dim = expected_feature_dim(cfg)

    # norm_stats.json (per-channel mean/std) must match the active channel count;
    # tiles are unchanged (features assembled at load), so refresh in place if stale.
    import json as _json
    from meepo_nz.data.preprocess import compute_norm_stats
    _ns = os.path.join(args.tiles, "norm_stats.json")
    _stale = True
    if os.path.exists(_ns):
        try:
            _stale = (_json.load(open(_ns)).get("n_features") != cfg.in_features_dim)
        except Exception:
            _stale = True
    if _stale:
        print(f"[05] norm_stats.json missing/stale for in_features_dim={cfg.in_features_dim}; "
              f"recomputing in place (tiles unchanged)...")
        compute_norm_stats(args.tiles, cfg)

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        # cudnn.benchmark stays OFF: it autotunes cuDNN per DISTINCT input shape, but our
        # per-cloud sequence lengths vary every batch (and Mix3D's pairwise merge makes the
        # merged length = voxels(a)+voxels(b), which essentially never repeats). With benchmark
        # ON, cuDNN re-benchmarks the BiMamba depthwise conv1d at every pooling stage on nearly
        # every step -> minutes/step and no steady-state speedup. The fixed-size raster CNN's
        # small autotune gain is not worth that; heuristic algo selection is fine here.
        torch.backends.cudnn.benchmark = False
        print("[05] TF32 matmul enabled")

    # PTv3-native full-scene dataset by default; legacy sphere dataset when --sphere-mode.
    if bool(getattr(cfg, "scene_mode", True)):
        from meepo_nz.data.scene_dataset import SceneDataset as _DS
        print("[05] dataset: full-scene (PTv3-native: GridSample + point-budget crop, whole tile / "
              "large block per sample; no spheres)"
              + (f"  | prior-raster branch INTEGRATED (per-block GrounDiff CNN)"
                 if bool(getattr(cfg, "use_dtm_raster", True)) else "  | raster OFF"))
    else:
        from meepo_nz.data.dataset import SphereDataset as _DS
        print("[05] dataset: LEGACY sphere mode (KPConv in_radius cylinders)")
    train_set = _DS(args.tiles, cfg, split="train")
    val_set = _DS(args.tiles, cfg, split="val")
    test_set = _DS(args.tiles, cfg, split="test")
    if len(test_set):
        gal_split, gal_src = "test", test_set
    elif len(val_set):
        gal_split, gal_src = "val", val_set
    else:
        gal_split, gal_src = "train", train_set
    vis_idx = gal_src.gallery_center_indices(n_want=int(getattr(cfg, 'n_vis_tiles', 6)))
    if len(val_set) == 0:
        val_set = test_set if len(test_set) else train_set
    if vis_idx:
        vis_set = _DS(args.tiles, cfg, split=gal_split,
                      center_subset=vis_idx, augment=False)
        cfg.n_vis_tiles = len(vis_set)
        print(f"[05] per-epoch gallery: {len(vis_set)} scenes (2 per scene type)")
    else:
        vis_set = test_set if len(test_set) else val_set

    collate = PTv3Collate(cfg, device=None, mix_prob=float(getattr(cfg, "mix_prob", 0.0)))
    model = build_meepo(cfg)

    if args.compile:
        try:
            import torch._dynamo as _dyn
            _dyn.config.capture_scalar_outputs = True
            _dyn.config.capture_dynamic_output_shape_ops = True
            _dyn.config.recompile_limit = 64
            # Point clouds have data-dependent shapes (variable points/patches, voxel
            # pooling, expert routing). suppress_errors makes Dynamo FALL BACK TO EAGER
            # on any region it can't trace instead of crashing the run; it then compiles
            # only the graph-able compute. The serialization, SerializedPooling and
            # SerializedUnpooling forwards are @torch.compiler.disable, so their
            # scatter_/unique/sort never reach inductor.
            _dyn.config.suppress_errors = True
            # Safety net: torch 2.12 inductor's joint-graph pattern matcher raises
            # ("Not all inputs to pattern found ... {'x': scatter}") on some scatter graphs.
            # Disabling the optional rewrite patterns sidesteps that crash; core lowering and
            # scheduler fusion still run, so the compiled dense islands are unaffected.
            try:
                import torch._inductor.config as _ind
                _ind.pattern_matcher = False
            except Exception as _e:
                print(f"[05] note: could not set inductor.pattern_matcher ({_e})")
            model = torch.compile(model, dynamic=True)
            print("[05] torch.compile enabled (dynamic=True; pooling/unpooling eager; inductor pattern_matcher off)")
        except Exception as e:
            print(f"[05] torch.compile unavailable ({e}); continuing uncompiled.")

    nparam = model.num_parameters() if hasattr(model, "num_parameters") \
        else sum(p.numel() for p in model.parameters())
    print(f"[05] device={device} in_features_dim={cfg.in_features_dim} params={nparam:,}")
    print(f"[05] model=MEEPO (CNN-Mamba) state_dim={getattr(cfg,'mamba_state_dim',1)} "
          f"conv={getattr(cfg,'mamba_conv_dim',4)} expand={getattr(cfg,'mamba_expand_factor',3)} "
          f"dirs={getattr(cfg,'mamba_directions',2)} ssm_backend={getattr(cfg,'ssm_backend','auto')} "
          f"raster={cfg.use_dtm_raster} gating={getattr(cfg,'prior_raster_gating',True)} "
          f"prev_dtm_scalar={cfg.use_prev_dtm}")
    print(f"[05] tiles: train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    if len(train_set) == 0:
        sys.exit("[05] ERROR: 0 training tiles. Run stages 01-04 first.")
    _acc = int(getattr(cfg, "grad_accum_steps", 1) or 1)
    print(f"[05] epochs={cfg.epochs} batch_num={cfg.batch_num}"
          + (f" x grad_accum {_acc} (EFFECTIVE batch {cfg.batch_num * _acc})" if _acc > 1 else "")
          + f" optimizer={getattr(cfg,'optimizer','adamw')}")

    trainer = Trainer(model, cfg, train_set, val_set, collate, vis_set=vis_set, device=device)
    trainer.train(cfg.epochs)

    if len(test_set):
        import json
        best = os.path.join(trainer.out_dir, "model_best.pt")
        if os.path.exists(best):
            ck = torch.load(best, map_location=device, weights_only=False)
            sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
            sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
            getattr(model, "_orig_mod", model).load_state_dict(sd)
            print("[05] scoring held-out TEST set with model_best.pt ...")
        t_loss, tm = trainer.evaluate_split(test_set)
        print(f"[05] HELD-OUT TEST (model_best): OA={tm.get('OA', float('nan')):.2f} "
              f"mIoU={tm.get('mIoU', float('nan')):.2f} "
              f"IoU_ground={tm.get('IoU1', float('nan')):.2f} "
              f"IoU_nonground={tm.get('IoU2', float('nan')):.2f} loss={t_loss:.4f}")
        try:
            with open(os.path.join(trainer.out_dir, "test_metrics.json"), "w") as fh:
                json.dump({"loss": t_loss, **tm}, fh, indent=2)
        except Exception:
            pass

    print(f"[05] done. outputs in {os.path.join(cfg.out_dir, cfg.name)}")


if __name__ == "__main__":
    main()
