# MEEPO — LiDAR ground-point extraction (clean PyTorch), trained on New Zealand

Binary **ground / non-ground** segmentation of New Zealand aerial LiDAR with **MEEPO**
(Point Transformer V3 + sparse Mixture-of-Experts, ICLR 2026), reimplemented in **clean
PyTorch** so it runs on **Blackwell (sm_120)** GPUs and on CPU — with **no spconv /
flash-attn / torch_scatter / timm**.

The model self-organises experts **without terrain labels** (label-free routing), and is
given two extra signals beyond raw geometry: a **previous-year classification raster**
(a GrounDiff-style spatial prior) and **per-point intensity + return counts**. The full
network is **MEEPO-L (≈99.9 M params)**.

> Engineering details, the faithful-vs-deviation breakdown, and the GrounDiff rationale are
> in [`proof/DESIGN.md`](proof/DESIGN.md).

---

## Why clean PyTorch

The official MEEPO depends on `spconv`, `flash-attn`, and `torch_scatter`, none of which
install cleanly on Blackwell sm_120 or can be tested on CPU. Each was swapped for a
pure-PyTorch equivalent (submanifold conv via hashed `searchsorted`; attention via
`scaled_dot_product_attention`; CSR reductions via `scatter_reduce_`; vendored `DropPath`).
The serialization (Z-order/Hilbert) and the MoE routing are ported verbatim, so the
algorithm is faithful — only the CUDA-only backends changed.

---

## The two deviations from the paper

1. **Previous-year classification raster (GrounDiff-informed).** The prior survey of the same
   footprint is rasterised at 1 m into 5 channels `[DTM, DSM, nDSM, ground_prob, coverage]`
   and encoded by a small 2-D CNN with a GrounDiff-style **confidence-gated** ground-height
   head (`g_ref = sigma(l)*DTM - (1-sigma(l))*resid`). Per-point features are sampled from it.
   Toggle with `--no-dtm-raster` (and `--no-raster-gating`).
2. **Intensity + return-count features.** Same sensor across years (no per-sensor
   normalisation): each point gets `number_of_returns`, the `return_number/number_of_returns`
   ratio, and `intensity`. The **other** per-point inputs match the paper: PTv3/MEEPO
   for outdoor LiDAR feed coordinates + intensity and learn geometry via the serialized
   conv/attention, so the hand-crafted shallow features (curvature, mean
   elevation, higher-order moments) are **off by default**. Default input width = **6**
   (xyz + return-count + return-ratio + intensity). Toggle each per channel.

The paper's CLIP/text head is dropped in favour of a plain 2-class linear head. The training
loss is the **MEEPO loss = CrossEntropy + Lovász-softmax** (both weight 1.0), matching
`configs/point_moe/indoor.py`. (A SparseGF height-aware HAG loss is available as an
off-by-default ablation via `--height-aware-loss`.)

**Backbone choice.** `backbone=point_moe` (default) is PTv3 + MoE (MEEPO-L, ~100M). `backbone=litept` swaps in a **LitePT** backbone (sparse-conv early stages, PointROPE rotary attention in the deep stages) with the same Proj-MoE — lighter and more rotation/transfer-friendly. A/B them on the same pipeline: `--backbone litept`.

**`--backbone vm3` (VoxelMamba-3, recommended for Mamba-3).** A group-free,
whole-scene Mamba-3 U-Net that replaces the MEEPO-3 retrofit: the OFFICIAL
multi-head Mamba-3 mixer (native anatomy: d_state 64, headdim 64, per-head
everything, whole-scene cumulative RoPE, packed varlen `cu_seqlens`) inside a
Voxel-Mamba-style host (dual-scale fwd/bwd branches, implicit window
embeddings) with UniMamba-style conv locality outside the mixer. 60.7M params
by default. Requires the official mamba package on GPU (Triton JIT; see
`VM3.md`). Not `--init-from` compatible with the other backbones. Full design,
provenance, flags and the smoke gate: **`VM3.md`**; smoke:
`PYTHONPATH=. python3 scripts/smoke_vm3.py`.

---

## Install (Blackwell or any CUDA box)

```bash
bash setup.sh                          # Blackwell: installs cu128 PyTorch wheels by default
# or: TORCH_CUDA=cu130 bash setup.sh   # pick a CUDA wheel
# or: TORCH_CUDA=cpu    bash setup.sh   # CPU-only (for the smoke test)
```

`setup.sh` installs a Blackwell-capable PyTorch (CUDA 12.8+), the Python deps, and this
package. **Do not** install spconv / flash-attn / torch_scatter / timm — they are not used.

---

## Smoke test (no data, no GPU)

```bash
PYTHONPATH=. python3 scripts/smoke_test.py
```

Runs the full stack (prior-raster build -> augment -> collate -> MEEPO train+eval -> full
MEEPO-L instantiation -> sphere-voting inference) on synthetic clouds in seconds. This is
the gate before spending GPU time.

---

## Run the whole pipeline

```bash
bash run_all.sh
# tune via env vars, e.g.:
DATA_ROOT=data/nz BUDGET_GB=40 EPOCHS=500 BATCH_NUM=10 NUM_WORKERS=8 DEVICE=cuda \
  CONFIG=configs/default.yaml bash run_all.sh
# dry-run the download plan only:
LIST_ONLY=1 bash run_all.sh
```

Stages: **01** download year-pairs -> **02** build the 5-channel previous-year prior raster ->
**03** skew report -> **04** preprocess to tiles -> **05** train. Or run scripts individually:

```bash
python3 scripts/01_download_data.py     --out data/nz --budget-gb 40 --source opentopography --workers 16
python3 scripts/02_build_prior_raster.py --root data/nz --workers 16
python3 scripts/04_preprocess.py        --root data/nz --out data/nz/tiles --dl 0.1 --in-radius 6 --workers 16
python3 scripts/03_classify_and_sample.py --tile-dir data/nz/tiles
python3 scripts/05_train.py             --tiles data/nz/tiles --scene-mode --batch-num 4 --accum-steps 4 --scene-max-points 204800 --num-workers 6 --epoch-steps 1500 --epochs 120 --device cuda --out-dir runs --name meepo_nz_ground
```
Training now defaults to the **PTv3-native whole-scene pipeline** (GridSample whole tile +
point-budget crop, the MEEPO/Pointcept method), with the previous-year prior-raster branch
(Deviation A) **trained end-to-end per block**. The flags above match the paper exactly:
`--batch-num 16` (MEEPO total batch_size), `--scene-max-points 204800` (SphereCrop `point_max`),
and `--epoch-steps 1500` (MEEPO is iteration-budgeted: indoor+outdoor = **180k optimiser iters** over 120 epochs = 1500 steps/epoch, and that budget is **dataset-size independent** — each step randomly samples 16 blocks from the corpus, so it's right whether you have 3k tiles or 300k; a full pass would instead scale with your dataset and miss the paper). The
2e-3 / 0.3× LR is the OneCycle setting *for batch 16*, so batch and LR are matched as a pair. batch_size=16 is the paper's *total* batch ("bs in all gpus"): 16 blocks of 204800 pts OOM a single 96 GB card, so `--batch-num 4 --accum-steps 4` keeps the **effective batch 16** by gradient accumulation (4 blocks/forward × 4) — mathematically the paper's cross-GPU gradient averaging, done on one device. Memory ≈ 4 blocks (~25 GB); bump to `--batch-num 8 --accum-steps 2` if you have headroom.
**Throughput note (whole-scene mode).** Two regimes to tell apart with `nvidia-smi` + `free -g`:
(1) *RAM maxed (low `available`) + GPU-util low* → host swap; cut `--num-workers` (6 is the scene
default — each sample loads a whole tile, so 16+ workers thrash). (2) *RAM fine + GPU-util 100% but
power well under cap (e.g. 130 W / 600 W)* → the model is memory/launch-bound (the MoE expert dispatch
and serialized-attention gather/scatter are many small kernels), **not** data-bound — so preprocessing
into smaller tiles will NOT speed it up, but these will: `--no-grad-checkpoint` (you have VRAM; the
default checkpointing recomputes the memory-bound forward in the backward pass — pure waste here, the
single biggest win), `--compile` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fuses the
small kernels), and a larger micro-batch (`--batch-num 8 --accum-steps 2`, same effective 16, better
bandwidth utilisation). Add levers one at a time and watch GPU power climb toward the cap; the
first-step ETA is startup-dominated and not the real rate. The
legacy KPConv overlapping-sphere path is behind `--sphere-mode`. Pass `--scene-mode` to force the
whole-scene path even if an older `data/nz/tiles/config.used.yaml` snapshot still has
`scene_mode: false` (otherwise re-run stage 04). Add `--log-every 20` for denser logging; add
`--compile` to try `torch.compile` (set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with it —
the inductor scatter fallback fragments the allocator).
Stage 01 keeps captures in a **2–9 pts/m²** band and round-robins across NZ areas, so the
corpus is regionally diverse. Stage 03 (`scripts/03_classify_and_sample.py --tile-dir
data/nz/tiles`) prints the per-region sphere distribution.

Inference on a new cloud:

```bash
python scripts/06_infer.py \
    --checkpoint runs/meepo_nz_ground/model_best.pt \
    --input scene.laz --out scene_classified.laz \
    --tiles data/nz/tiles \
    --prev-dtm data/nz/prior/<region>_<prevyear>/<cloud>.npz \
    --error-image err.png
```

(`--prev-dtm` accepts the 5-channel prior from stage 02; a legacy single-channel DTM is also
accepted and auto-promoted.)

**Large scenes.** `scripts/09_infer_large_scenes.py` runs over the held-out preprocessed tiles
(optionally merging contiguous tiles into bigger blocks) and emits a clean classified LAZ + an
error image + a review panel per scene. It picks the inference path automatically:

```bash
python scripts/09_infer_large_scenes.py \
    --checkpoint runs/meepo_nz_ground/model_best.pt \
    --tiles data/nz/tiles --out-dir runs/meepo_nz_ground/scenes \
    --split test --num-scenes 3 --merge --max-tiles 9 --inference auto
```

`--inference auto` (default) uses **PTv3-native whole-scene inference** — grid-subsample the
scene, tile it into `scene_block_size`-m blocks with a context ring, voxelise each block, run one
forward, and expand voxel→point — exactly how PTv3/Pointcept tests whole scenes. The
**previous-year prior-raster branch is integrated into this path**, identically to training: the
prior is cropped to each block window, resampled to `raster_scene_patch_size` px, and run through
the confidence-gated CNN inside the forward; each point bilinearly samples the result. So there is
**no train/test mismatch** — the raster CNN sees blocks at train and test time the same way. Both
paths keep **dl = 0.1** (voxelisation/subsampling grid). Force a path with `--inference scene` /
`--inference spheres` (the latter uses the legacy KPConv overlapping-sphere voting); add
`--refine off` for the raw argmax (no ground post-correction); the default `--refine spag_dc` runs the SPAG-DC misclassification corrector (IEEE Sensors 2025; see `SPAG_DC.md`). Tune within the paper's ranges via `--spag-alpha` (0.5-0.7), `--spag-beta` (0.6-0.8), `--spag-theta0`, `--spag-k`, `--spag-n-sigma`.

---

## What you get every epoch

Every epoch the training loop emits, for a **region-diverse** set of ~200 m scenes (spread
across distinct NZ clouds/surveys, not any terrain taxonomy):

* **one combined image per scene** (`scene_<tile>.png`) with five bands —
  (1) **inputs**: elevation relief, return count, return ratio, normalised intensity and the
  previous-year prior DTM; (2) **gap-free TIN DEMs**: **DSM**, **true DTM** and **predicted
  DTM** (Delaunay-triangulated, hole-filled); (3) **classification**: ground truth / prediction
  / errors; (4) **cross-section profiles**: truth / prediction / errors; with a metrics header
  (OA, ground IoU, DTM RMSE);
* **one clean LAZ per scene** (`scene_<tile>.laz`) holding only standard LAS fields
  (classification + intensity + return counts), so it opens in any QGIS / CloudCompare / PDAL build;

plus a refreshed **metrics/loss/ETA dashboard**, a rolling checkpoint + `model_best.pt`, and a
final held-out **test** score (`test_metrics.json`).

## Metrics
Binary IoU (ground / non-ground), overall accuracy, and mIoU, via the reused
`ConfusionAccumulator`.

## Ablations
```bash
python scripts/05_train.py --tiles data/nz/tiles --no-moe              --name pm_dense
python scripts/05_train.py --tiles data/nz/tiles --num-experts 4 --moe-topk 1
python scripts/05_train.py --tiles data/nz/tiles --moe-aux-alpha 0.01  # paper best = 0
python scripts/05_train.py --tiles data/nz/tiles --no-dtm-raster       --name pm_nora
python scripts/05_train.py --tiles data/nz/tiles --no-raster-gating
python scripts/05_train.py --tiles data/nz/tiles --no-intensity --no-return-features
```

## Layout
```
meepo_nz/      models/ (MEEPO), data/ (prior raster + PTv3 collate), inference/, training/, utils/
scripts/           01..09 pipeline + smoke_test.py
configs/           default.yaml (MEEPO-L defaults)
proof/DESIGN.md    architecture, deviations, GrounDiff basis, Blackwell rationale, verification
arcgis/            ArcGIS Pro integration
```

## References
MEEPO (arXiv:2505.23926, ICLR 2026) - Point Transformer V3 (Wu et al., CVPR 2024) -
GrounDiff (arXiv:2511.10391).
