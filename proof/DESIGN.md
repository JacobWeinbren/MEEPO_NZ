# MEEPO for NZ LiDAR ground extraction — design & rationale

This document explains what was built, why it was built this way, and how to verify it.
It is the engineering record behind the swap from the original model to
**MEEPO** while keeping the surrounding New Zealand LiDAR pipeline (download,
preprocess, skew-balanced sampling, training loop, per-epoch visual gallery, metrics,
DTM/raster scaffolding) intact.

---

## 1. What the model is

**MEEPO** (Point Transformer V3 backbone + sparse Mixture-of-Experts), introduced
in *MEEPO: Towards Cross-Domain Generalization in 3D Semantic Segmentation via
Mixture-of-Experts* (ICLR 2026, arXiv:2505.23926). The backbone serialises points along
space-filling curves (Z-order + Hilbert, plus their transposes), runs windowed
("serialized") attention over the 1-D order, and pools/unpools through a U-Net of
encoder/decoder stages. The MoE replaces a dense projection with a top-k–routed bank of
expert MLPs, so different experts can specialise without any domain/terrain labels — the
routing is **label-free** and the experts self-organise.

Here the network is configured as **MEEPO-L** and produces a **binary** per-point
prediction (ground = 1, non-ground = 0). The final decoder stage unpools back to the input
voxel resolution, so the logits align one-to-one with the input voxels.

**Parameter count (full MEEPO-L, as built): 99,920,172.**

### Backbone shape (MEEPO-L)
- orders `(z, z-trans, hilbert, hilbert-trans)`, shuffled each layer
- encoder depths `(2, 2, 2, 6, 2)`, channels `(32, 64, 128, 256, 512)`, heads `(2, 4, 8, 16, 32)`
- decoder depths `(2, 2, 2, 2)`, channels `(64, 64, 128, 256)`
- stem submanifold conv kernel 5; per-block xCPE submanifold conv kernel 3; drop-path 0.3
- **MoE: 8 experts, top-2 routing, BatchNorm, `n_intermediate_size = 2`, on the attention
  output projection, `aux_loss_alpha = 0`.**

These match the paper's MEEPO-L and its ablations. Two choices are worth calling out
because they are counter-intuitive but are what the paper found best:
- **BatchNorm** beats LayerNorm for this model (Tab. 3c).
- **No load-balancing auxiliary loss** (`alpha = 0`): the aux loss *hurts* final accuracy
  in their sweep, so it is disabled by default (exposed as `--moe-aux-alpha` for ablation).

---

## 2. Why a clean-PyTorch reimplementation (and not the official repo)

The target machine is a single **RTX PRO 6000 (Blackwell, sm_120, 96 GB)**. The official
MEEPO code is built on Pointcept and depends on `spconv`, `flash-attn`, and
`torch_scatter`. On Blackwell sm_120 none of these install cleanly from wheels at the time
of writing, and all three are impossible to exercise on a CPU-only box, so they cannot be
smoke-tested before committing GPU hours. To get a model that (a) builds on Blackwell and
(b) is verifiable on CPU, every CUDA-only dependency was replaced with a pure-PyTorch
equivalent:

| Official dependency        | Replaced with (clean PyTorch)                                   |
|----------------------------|-----------------------------------------------------------------|
| `spconv.SubMConv3d`        | `models/submanifold_conv.py` — packed-key hashing + `searchsorted`, O(N·K³) gathers |
| `flash_attn`               | `torch.nn.functional.scaled_dot_product_attention` (`serialized_attention.py`) |
| `torch_scatter.segment_csr`| `torch.Tensor.scatter_reduce_` (`ptv3_moe.py::_segment_reduce`) |
| `timm.layers.DropPath`     | vendored in `models/layers.py`                                  |
| `addict.Dict`              | a tiny `AttrDict` in `models/point_structure.py`                |

The Z-order and Hilbert serialisation code and the MoE layer are ported essentially
verbatim from the official implementation (same bit-twiddling, same routing math), so the
*algorithm* is faithful; only the *backends* changed. The result runs on CPU and on
Blackwell with stock PyTorch (CUDA 12.8+ wheels).

---

## 3. The two accepted deviations from the paper

### Deviation A — previous-year classification raster (GrounDiff-informed)

Each current-year cloud is given a **spatial prior** from the previous survey of the same
footprint. We rasterise the previous year's *classified* cloud at 1 m into a **5-channel**
prior raster and feed it through a small 2-D CNN whose per-point–sampled output is
concatenated to the point features before the backbone.

The 5 channels are chosen to mirror what **GrounDiff** (*GrounDiff: Diffusion-based Ground
Surface Generation / DSM→DTM denoising*, arXiv:2511.10391) consumes, because the user's
instruction for this branch was explicitly "whatever GrounDiff does":

```
[ DTM, DSM, nDSM (= DSM − DTM residual), ground_prob (observed ground confidence), coverage ]
```

`build_prior_raster_from_prev` (in `data/dtm.py`) produces these from the prior cloud's
ground vs non-ground returns. The encoder (`models/prior_raster_encoder.py`) mean-centres
the three height channels, then has a **dual head**:
1. a terrain-feature head (the context features sampled per point), and
2. a **GrounDiff-style confidence-gated refinement** of the ground height,
   the faithful Eq. 5 form `g_ref = σ(ℓ)·DTM + (1 − σ(ℓ))·(DTM − resid) =
   DTM − (1 − σ(ℓ))·resid`, where `ℓ` is a learned per-cell gate. The prior-DTM
   anchor is always present; the residual only corrects it where the cell is not
   confidently unchanged ground.

This gating is the part GrounDiff shows is critical: their ablations report the residual
formulation beats predicting absolute height (~+17 %) and that removing the gate makes the
result ~12× worse. Each point bilinearly samples this map and also receives the scalar
`z − prior_DTM`. The branch is toggleable (`--no-dtm-raster`) and the gate is toggleable
(`--no-raster-gating`) for ablation. When a tile has no usable previous-year twin, a
sensible "ground-everywhere" prior is synthesised so the branch still runs.

### Deviation B — intensity + return-count per-point features

Same sensor across years, so no per-sensor normalisation is applied. Per point we add
`number_of_returns`, the normalised `return_number / number_of_returns` ratio, and
`intensity` (these live in `features/shallow_features.py`).

The **non-deviation** per-point inputs match the paper: PTv3/MEEPO for outdoor LiDAR feed
coordinates (+ intensity) and let the serialized conv/attention learn geometry, so the
hand-crafted shallow features (mean elevation, surface curvature, higher-order
moments) are **disabled by default** — they are not part of PTv3/MEEPO. The default input
width is therefore **6** = xyz(3) + return_count(1) + return_ratio(1) + intensity(1); the prior
raster (Deviation A) is a separate branch. Each channel is individually ablatable
(`--no-intensity`, `--no-return-features`, `--no-return-ratio`), and the shallow features can be
re-enabled for experiments via their config flags.

### What was dropped from the paper
The paper's CLIP/text segmentation head is removed; this is a 2-class problem, so the model
ends in a plain `Linear(dec_channels[0] → 2)`. Routing is label-free
(`moe_domain_guided = False`): the gate sees only point features, never a terrain-type
label, so experts self-organise exactly as in the paper.

---

## 4. How the pipeline fits together (data contract)

The original pipeline was reused as-is wherever possible; only the collate, the model, and
the raster branch were swapped.

1. **01 download** — NZ open LiDAR year-pairs (current + previous-year twin), unchanged.
2. **02 build prior raster** (`02_build_prior_raster.py`, new) — rasterise each previous-year
   *classified* twin into the 5-channel prior `.npz`; manifest gains `prior_rasters`.
3. **03 skew report** — tile-class statistics, unchanged.
4. **04 preprocess** — grid-subsample each cloud, build candidate sphere centres, crop the
   per-tile prior (`prior_data`/`prior_geo`, 5-ch) and derive the legacy DTM channel; write
   one `.npz` per cloud + per-channel `norm_stats.json`.
5. **05 train** — spheres are sampled UNIFORMLY at random (regional diversity from stage
   01's round-robin over areas; no scene-type up-weighting). `SphereDataset` → **`PTv3Collate`** (voxelise to coord/grid_coord/feat/offset,
   majority label per voxel, carry the per-sphere prior patch) → **`build_meepo`** →
   the **same `Trainer`** (progress bars, per-epoch combined per-scene reports (inputs + gap-free TIN DSM/DTM/predicted-DTM + classification + profiles) and one clean LAZ per scene,
   metric dashboard, held-out test scoring).
6. **06 infer / 07 gallery / 08 grid / 09 large-scene** — two inference paths, both at
   **dl = 0.1**: (a) **PTv3-native whole-scene** (`inference/voting.py::predict_scene`) —
   grid-subsample, tile into `scene_block_size`-m blocks with a context ring, voxelise, one
   forward per block, expand voxel->point (the Pointcept whole-scene test scheme); (b)
   **sphere voting** (`predict_cloud_spheres`) — voxelise each sphere, run the model, map
   per-voxel probabilities back to points, vote. `09 --inference auto` selects (a) when the
   model has no prior-raster branch, else falls back to (b) (the raster branch is per-sphere
   and incompatible with point-only whole-scene inference).

`PTv3Collate`'s final logits are voxel-aligned, and `batch["points"][0]`/`labels` are the
same voxels, so the reused metrics and visualiser line up without changes.

---

## 5. Verification — CPU smoke test

The whole stack is exercised on tiny synthetic clouds (no GPU, no real data), runs in
seconds, and is the gate before spending GPU time:

```
PYTHONPATH=. python3 scripts/smoke_test.py
```

It checks, end to end:
1. prior-raster build → crop+downsample → per-sphere multi-channel patch crop;
2. multi-channel augmentation (in-plane warp + height-channel vertical scale);
3. `PTv3Collate` → MEEPO **forward + backward** (TRAIN, dense MoE path) and a forward in
   **EVAL** (the MoE inference scatter path);
4. full **MEEPO-L instantiation** (99,920,172 params);
5. PTv3 **sphere-voting inference** (+ `return_proba`).

Observed result (CPU):

```
[1/5] prior raster: (5, 60, 60)  nDSM_max=12.1  gprob=[0.00,1.00]  patch=(5, 24, 24)  PASS
[2/5] augment: patch(5, 24, 24) local(2000, 3)  PASS
[3/5] model: params=82,302 voxels=1877 loss=0.830 grad_norm=1.19  train+eval  PASS
[4/5] full MEEPO-L instantiated: 99,920,172 params  PASS
[5/5] voting: pred(5000,) classes=[0, 1] proba(5000, 2)  PASS
SMOKE TEST PASSED
```

(The full MEEPO-L also runs a real fwd+bwd on CPU in ~1.4 s for ~1,200 voxels; the
smoke test uses a shrunk config for stage 3 so the loop stays fast.)

---

## 6. File map (new / changed)

```
meepo_nz/models/
  serialization/            z_order.py, hilbert.py (verbatim) + wrappers
  point_structure.py        Point / PointModule / PointSequential, offset<->batch helpers
  submanifold_conv.py       pure-PyTorch SubMConv3d (subclasses PointModule)
  serialized_attention.py   SDPA-based serialized attention (+ optional MoE projection)
  moe_layer.py              MoEMLP / MoEGate / MoELayer (ported verbatim) + make_moe_config
  ptv3_moe.py               PointTransformerMoE backbone (BatchNorm, MoE on attn proj)
  layers.py                 vendored DropPath
  prior_raster_encoder.py   Deviation A: 2-D CNN, GrounDiff-style gated dual head + sampler
  segmentation_model.py     MeepoSeg (+ build_meepo) — raster concat + 2-class head
data/
  ptv3_collate.py           PTv3Collate (voxelise -> PTv3 batch dict)
  dtm.py                    + MultiRaster, build_prior_raster_from_prev, crop_* , load_prior_raster
  dataset.py                _load_prior + multi-channel patch crop
  augment.py                multi-channel patch augmentation
  preprocess.py             crop/store 5-ch prior (prior_data/prior_geo) + legacy DTM channel
  batch.py                  generic move_batch
inference/voting.py         PTv3 voxel-based sphere voting (same signature)
scripts/
  02_build_prior_raster.py  build the 5-ch prior from prev classified twin
  04_preprocess.py          load prior MultiRaster, thread prev_prior
  05_train.py               PTv3Collate + build_meepo, MoE/ablation flags
  06_infer.py, 08, 09       PTv3 inference; load prior via load_prior_raster
  smoke_test.py             the CPU end-to-end test above
configs/default.yaml        MEEPO-L defaults + prior-raster fields
```

---

## 7. References
- MEEPO — arXiv:2505.23926 (ICLR 2026).
- Point Transformer V3 — Wu et al., CVPR 2024 (serialization + serialized attention).
- GrounDiff (DSM→DTM denoising, residual + gating) — arXiv:2511.10391.
