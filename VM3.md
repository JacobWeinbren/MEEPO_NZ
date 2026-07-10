# VM3 — VoxelMamba-3 (`--backbone vm3`)

Group-free, whole-scene **Mamba-3** backbone for the NZ ground-extraction
pipeline. Replaces the MEEPO-3 retrofit (`models/meepo3.py`), which kept
Mamba-1 anatomy (per-channel scan, dt_rank bottleneck, d_state 1–4,
per-direction convs) around the Mamba-3 math and could not close the layout
residuals documented in its header. VM3 is the multi-head rebuild: the mixer
is the **official** `mamba_ssm.modules.mamba3.Mamba3` (Lahoti et al.,
arXiv:2603.15569) at native anatomy, and the host is designed around it.

## Design (provenance per component)

| Component | Source | What VM3 does |
|---|---|---|
| Whole-scene sequences, no patches, no padding | Voxel Mamba (NeurIPS 2024, 2406.10700) | every cloud = ONE serialized sequence; batch = packed varlen `cu_seqlens` (official kernels support fwd+bwd varlen) |
| Mixer | Mamba-3 (2603.15569) | SISO, d_state 64, headdim 64, expand 2, per-head dt/A/λ, B/C biases after BCNorm, data-dependent RoPE (fraction 0.5) **cumulative over the whole scene**, heavy-tail A, **no conv in the mixer**, MIMO off |
| Bidirectionality + hierarchy | Voxel Mamba DSB (Eq. 4) | fwd branch full-res, bwd branch on code-downsampled flipped sequence (`--vm3-dsb-down`, default 1,2,4,4), LN per branch, broadcast back |
| Positional encoding | Voxel Mamba IWP | implicit window embedding MLP (z, window idx, in-window offsets, half-window shift), shared per stage; complementary to RoPE |
| Spatial locality outside the mixer | UniMamba SLM (CVPR 2025, 2503.12009) + MEEPO xCPE | k5 subconv stem + optional k3 xCPE per block (`--vm3-no-cpe` to ablate). UniMamba Tab. 4: with conv locality even random ordering ≈ Hilbert |
| Local–global | replaces UniMamba LGSA | per-head **decay banding**: dt_bias initialized log-spaced over [dt_min, dt_max] so heads span local→global horizons; data-dependent per-head A learns the split (`--vm3-no-decay-bands` to ablate) |
| Skeleton | this repo | U-Net over `SerializedPooling`/`SerializedUnpooling`, channels 128→256→384→512, heads 4/8/12/16, decoder 384→256→128; CE + Lovász as before |

Default model: **60.7M params**, `out_channels=128`.

## Requirements

The GPU path needs the official mamba package (Triton JIT — no CUDA
extension is built by default; works on Blackwell sm_120):

```bash
pip install einops transformers packaging   # mamba_ssm import-chain deps
pip install --no-deps -e ~/mamba-main
python -c "from mamba_ssm.modules.mamba3 import Mamba3; print('mamba3 OK')"
```

`--ssm-backend cuda` **requires** the official module (clear error otherwise);
`auto` falls back to a pure-torch reference (`Mamba3TorchRef`, same math and
state-dict layout, CPU-safe, slow) — the startup banner prints which one is
live. Never train on the reference.

## Smoke gate (run before GPU time)

```bash
PYTHONPATH=. python3 scripts/smoke_vm3.py
```

Checks: flip involution; forward/backward finiteness; **packed-varlen ==
per-cloud exactness** (validates serialization/cu_seqlens/flip/DSB-pool/Up
plumbing, max err ~1e-5); decay-band spacing; MeepoSeg(vm3) end-to-end.

## Caveats

* **Not `--init-from` compatible** with `meepo`, `meepo3`, or `pointssm`
  checkpoints (disjoint parameter shapes).
* Data/tiles/prior-raster preprocessing is **unchanged and reusable** — the
  input feature contract is identical to the other backbones.
* Stage widths must satisfy `(expand·C) % headdim == 0` (all defaults do).
* Grad checkpointing: VM3 supports block granularity ('stage'/'layer' map to
  'block'); enabled whenever the trainer's `grad_checkpointing` is on.
* The mixer holds no conv and no per-direction parameters; the two DSB
  branches are two independent Mamba-3 blocks per VM3Block.

## Ablation flags

`--vm3-state 32|64|128` (state-size ladder), `--vm3-no-cpe`,
`--vm3-no-decay-bands`, `--vm3-dsb-down 1,1,1,1` (plain bidirectional, no
dual-scale), `--vm3-iwe-window`, `--vm3-enc-depths/-channels` (scale).

## r2 changes (2026-07-10)

* `scripts/smoke_vm3.py` sets `POINT_MOE_DISABLE_SPCONV=1` itself, so it runs
  unmodified on GPU boxes (the CUDA implicit-GEMM conv path rejects CPU
  tensors; training is unaffected).
* `training/trainer.py` now honors `_no_weight_decay` flags (Mamba-3 dt_bias
  and D): param groups split into decay / no-decay halves at both LR tiers,
  wd applied per group, OneCycle max_lr follows the group LRs. Protects the
  local/global decay bands from wd=0.05 pressure over 80k-step schedules.
  Applies to all backbones; fresh runs only (old optimizer checkpoints have a
  different group layout).

## Fast profile

If wall clock matters more than the last fraction of quality:
`--vm3-no-cpe --vm3-enc-depths 1,1,2,2`. The per-block k3 subconv is the
single largest MAC term at full resolution (~27C^2/point vs ~27C^2 for
mixerx2+SwiGLU combined) and is a planned ablation anyway (UniMamba Tab. 4:
with Hilbert ordering, SLM is worth ~0.2-0.3); halving the two full/half-res
stages' depth removes most of the rest. Locality still comes from the k5 stem,
IWE, and the Hilbert/Z complementary orders.

## r3 changes (2026-07-10)

* `--resplit-seed N` (+ `--resplit-val-frac/--resplit-test-frac`): runtime
  override of the stage-04 split via a deterministic per-cloud hash — no
  re-preprocessing, order-independent, stable across runs/machines. Changes
  val AND test; check the [diag] train/val class-balance lines agree, then
  freeze the seed for that dataset. Implemented in `data/splitting.py`.
* `probe_vram.py` accepts `--backbone vm3 --ssm-backend cuda` so the 16 GB
  feasibility probe exercises the real VM3 stack.
