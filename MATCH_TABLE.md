# Faithfulness audit

This project now implements **MEEPO** (PTv3 + sparse Mixture-of-Experts, ICLR 2026),
not an earlier model. The architecture, the parts kept faithful to the paper
(serialization, serialized attention, top-2 / 8-expert MoE on the attention projection,
BatchNorm, no aux loss, CrossEntropy + Lovász loss), the clean-PyTorch backend swaps (spconv / flash-attn /
torch_scatter / timm all removed), and the two accepted deviations (GrounDiff-style
previous-year classification raster; intensity + return-count features) are documented in:

→ [`proof/DESIGN.md`](proof/DESIGN.md)

---

## Verified hyperparameters (re-checked against the paper figures/tables)

Confirmed by reading MEEPO **Table 3** (MoE-design ablations), **Table 10** (training
settings), **Tables 12/13** (results / param count), and **Figure 1** (architecture), plus
GrounDiff **Eq. 5** + its ablations. "Config" = this repo's default.

| Item | Paper (MEEPO-L) | Config | Match |
|---|---|---|---|
| Experts | 8 (Tab. 3h best) | `num_experts=8` | ✓ |
| Top-k | 2 (Tab. 3b best) | `moe_topk=2` | ✓ |
| Shared experts | 0 (Tab. 3g best) | `moe_use_residual=False` | ✓ |
| Aux load-balance loss | 0 (Tab. 3a best) | `moe_aux_loss_alpha=0.0` | ✓ |
| Normalization | BatchNorm (Tab. 3c best) | `use_batch_norm=True` | ✓ |
| MoE placement | attention Proj (Tab. 3d best) | `use_moe_proj=True`, MLP off | ✓ |
| Activation | ReLU (Tab. 3e best) | `moe_act_fn="relu"` | ✓ |
| Expert width | 2H (Tab. 10) | `moe_n_intermediate_size=2.0` | ✓ |
| Optimizer | AdamW (Tab. 10) | AdamW | ✓ |
| Weight decay | 0.05 (Tab. 10) | `adamw_weight_decay=0.05` | ✓ |
| LR schedule | OneCycleLR (Tab. 10) | cosine + warmup (≈ OneCycle) | ≈ |
| Peak LR | base 0.005; OneCycle peak 0.002 (head) / 0.0006 (block) | `adamw_lr=2e-3`, `block_lr_scale=0.3` | ✓ (effective peak) |
| Params | 100M total / 60M activated (Tab. 12/13) | smoke = 99,892,172 | ✓ |
| Loss | CrossEntropy + Lovász, weight 1.0 each | `loss_lovasz=True`, CE+Lovász | ✓ |
| Total batch size | **16** (indoor+outdoor, "total bs in all gpus") | `batch_num=16` | ✓ |
| Per-scene point budget | SphereCrop `point_max=204800` | `scene_max_points=204800` | ✓ |
| LR warmup | OneCycle `pct_start=0.05` (~6 of 120 ep) | `warmup_epochs=6` | ≈ |
| Iters / epoch | 1500 (indoor+outdoor: 180k iters / 120 ep) | `--epoch-steps 1500` | ✓ |
| Epochs | 120 | `--epochs 120` | ✓ |

The peak LR (`0.002` head / `0.0006` block) is the OneCycle setting **for batch_size 16** — so
`batch_num=16` and the LR are matched as a pair; running `batch_num=1` (the old scene-mode
default) silently mismatched the LR. 16 is the *total* batch across GPUs; 16 blocks of 204800 pts OOM one 96 GB card, so use `--batch-num 4 --accum-steps 4` — gradient accumulation to **effective batch 16**, which is exactly the paper's cross-GPU gradient averaging (verified: the accumulated gradient equals a true 16-block gradient, and BN-over-points stays stable since each forward still has ~800k points). The iteration budget is **dataset-size independent** and is the thing to copy: indoor+outdoor is
**180k optimiser iters** = 120 epochs × 1500 steps/epoch (indoor = 140k = ~1167). Each step
`RandomSampler`s 16 blocks from the corpus, so 180k iters is the same training whether the corpus is
3k tiles or 300k — only the per-tile reuse changes (heavy augmentation + random large-block crops
keep the draws distinct). A *full pass* per epoch (`epoch_steps=0`) would instead scale with the
dataset and over/under-shoot the paper, so the fixed `epoch_steps=1500` is the faithful choice. At
batch 16 this is the paper's full ~1–2 GPU-day budget on one card; lower `--epochs` (the cosine/
OneCycle horizon scales with it) for a shorter but complete schedule.

**Steps per epoch.** MEEPO is **iteration-budgeted, not fixed-steps-per-epoch**:
**140k** total optimiser iters (indoor joint) / **180k** (indoor-outdoor) over 120 epochs ⇒
**≈1167–1500 steps/epoch** (data-determined: `len(loader) = num_samples / batch`). This repo
keeps KPConv's fixed-step epoch (`epoch_steps=500`) because the NZ sphere corpus is far larger
than ScanNet, so a full pass × hundreds of epochs is impractical. The **total** budget still
lands in the paper's range — e.g. `--epochs 500 × epoch_steps 500 = 250k` optimiser steps
(≥ 140–180k). To match the paper's epoch granularity exactly, use `--epochs 120` with
`epoch_steps≈1200` (~144k, indoor) or `≈1500` (~180k, indoor-outdoor); set `epoch_steps=0` for a
true full pass on small tile sets. **In `scene_mode` one "step" is a whole tile / large block**
(far heavier than a KPConv sphere), so the same iteration budget is a fraction of the wall-clock
spheres would need; with ~3k tiles a full pass is `epoch_steps≈3000`, and `--epochs 50 × 3000 ≈
150k` lands in-range. Tune `--epochs × --epoch-steps` toward 140–180k total.

**GrounDiff prior-raster branch.** 5 channels `[DTM, DSM, nDSM, ground_prob, coverage]`;
confidence-gated **residual** head implements Eq. 5 `ĝ = σ(ℓ)·DTM + (1−σ(ℓ))·(DTM − r̂)`
(`prior_raster_encoder.py:137`). Matches the paper's ablations (residual ≈ +17%, removing the
gate ≈ 12× worse).

**SPAG-DC ground-misclassification corrector.** Post-classification ground refinement is a faithful
implementation of **SPAG-DC** (Zhu, Tang et al., *IEEE Sensors Journal* 25(21):40399-40411, 2025,
doi:10.1109/JSEN.2025.3615605): a deterministic, non-learned closed-loop post-filter on the model's
initial ground points that detects and removes Type-II errors (non-ground misclassified as ground =
DTM spikes). Region-growing core (per-point normal+curvature, adaptive normal-angle/curvature
thresholds) → density/gradient adaptive seed grid → Maximum-Consistent-Set seed purification → local
thin-plate-spline surface → reclassify any candidate whose vertical residual to the surface exceeds
`mu2 + n*sigma2` (with a 0.1 m per-grid floor). Parameter values are the paper's (theta0=10°, alpha=0.5,
beta=0.7, k≈20, n=3); the only learned input is the MEEPO ground mask. See `SPAG_DC.md` for the
equation mapping, parameter sources, and the documented deviations (reclassify-only; core sources seeds
while the TPS residual is the detector; elevation-robust MCS; per-cell TPS). This replaces the earlier
PointCVaR refiner, which used a fixed-fraction CVaR tail that demoted ~5% of predicted ground on every
tile and raised DTM-RMSE. The old hand-tuned "SPAG-DC" and the RegimeHead difficulty field were removed.


**Data pipeline — now PTv3-native (verified against the MEEPO/Pointcept repo).**
PTv3 / MEEPO does **NOT** train on KPConv input spheres. Re-reading the official configs
(`configs/point_moe/indoor+outdoor.py`) and `pointcept/datasets/transform.py`, the real pipeline is:

| Stage | MEEPO / Pointcept | This repo (`scene_mode=True`, the default) |
|---|---|---|
| Train | augment (rotate-z + small tilt, scale, flip, jitter, elastic) → **GridSample** (one point/voxel) → **SphereCrop**(`point_max≈204800`) → Collect — *one whole scene, voxelized, cropped to a point budget* | augment (`augment_tile`) → `SceneDataset` whole tile / large block, count-capped to `scene_max_points` → `PTv3Collate` GridSample at `first_subsampling_dl` | ✓ |
| Val | GridSample whole scene, **no crop, no voting** | `SceneDataset` (eval window) → GridSample | ✓ |
| Test | GridSample(`mode=test`) fragments + TTA, votes summed; huge scenes tiled | `predict_scene` block-tiled whole-scene, GridSample per block | ≈ (block-tiled, no TTA) |

The legacy **KPConv overlapping-sphere** path (50M candidate cylinders + sphere voting) is **not**
the MEEPO method; it remains available only behind `--sphere-mode`. Flipping the default to
`scene_mode=True` is also what fixes the start-up hang: the sphere dataset eagerly builds a
~50M-entry candidate index (opening every tile in `__init__`), whereas `SceneDataset` is one
mmap'd sample per tile. Grid size: the paper uses 0.02 m (dense indoor) / ~0.05 m (outdoor); this
repo uses `first_subsampling_dl=0.1` for 10 cm aerial LiDAR — a deliberate, resolution-matched
choice. `point_max` (paper 204800) maps to `scene_max_points` (a per-GPU memory cap).

**Prior-raster branch (Deviation A) integrated into the whole-scene pipeline.** The GrounDiff
raster branch is **no longer dropped in scene mode** and is **no longer a per-sphere bolt-on**.
For every whole-scene block the previous-year 5-channel prior is cropped to the block window
(`scene_block_size`), resampled to `raster_scene_patch_size` px (height channels offset by the
block-centre z), and run **once** through the fully-convolutional confidence-gated CNN *inside the
model forward* — so the raster CNN trains **end-to-end** with the backbone (verified: its 21
parameters receive gradient). Each point bilinearly samples the resulting feature map
(`raster_tile_size = scene_block_size`). `SceneDataset.__getitem__` (train/val), `PTv3Collate`,
and `predict_scene` (inference) all use the **identical** path, so there is no train/test mismatch
and no grid-snapping approximation. `09 --inference auto` uses whole-scene for every model;
`--inference spheres` keeps the legacy overlapping-sphere voting path.