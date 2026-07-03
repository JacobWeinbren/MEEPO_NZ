# GrounDiff nDSM regression — the fix for majority-class (predict-all-ground) collapse

## The symptom
Training collapsed to predicting **everything is ground**:

| epoch | IoU1 (non-ground) | IoU2 (ground) | OA | Kappa | mIoU | val CE |
|------:|------------------:|--------------:|-----:|------:|-----:|-------:|
| 1 | 39.53 | 61.60 | 69.31 | 33.27 | 50.57 | 0.644 |
| 2 | 32.18 | 62.01 | 67.81 | 27.19 | 47.10 | 0.645 |
| 3 |  3.02 | 61.69 | 62.14 |  3.49 | 32.36 | 0.853 |

Non-ground IoU 39→32→3 (→0); ground IoU pinned ~62; OA → the ground fraction
(~62%); Kappa → 3; val CE *rises* as the model becomes confidently all-ground.
This is the algebra of "predict the majority class everywhere": IoU2 = ground
fraction, IoU1 ≈ 0, OA = ground fraction, base-rate CE ≈ 0.66. RMSE even
"improves" because an all-ground surface fits the actual ground points.

## Why it happens (paper-grounded, not a guess)
The loss is **exactly** MEEPO's criteria — `CrossEntropyLoss(weight=1)` +
`LovaszLoss(multiclass, weight=1)`, `classes="present"`, per-batch, **unweighted**
(verified against `configs/point_moe/indoor.py` and `pointcept/models/losses/lovasz.py`).
No reference config uses class weights, even on imbalanced outdoor data — so
class weighting is *not* the reference answer.

The problem is a **regime mismatch**: MEEPO / LitePT are *multi-class*
semantic-segmentation recipes. On a *binary, imbalanced* task (~62/38 here, with
a strong previous-year DTM prior), unweighted mean-CE has a strong
"predict-majority" local minimum (mean CE ≈ the base rate ≈ 0.66) that is far
more attractive than in multi-class (where predicting one class is catastrophic
for CE). Lovász (weight 1.0) opposes it but is overwhelmed as the OneCycle LR
ramps to its 2e-3 peak — which is exactly when the collapse accelerates.

**This is not the SparseGF HAG loss.** There is no SparseGF paper in the
provided set, so re-enabling the HAG bin-classifier was never "matching the
papers." The ground paper that *is* in the set is **GrounDiff**.

## The fix: GrounDiff's regression-dominant objective (Dhaouadi et al. 2025)
GrounDiff explicitly contrasts *classification* formulations of ground
extraction against *regression* formulations, and deliberately makes the
problem **regression-dominant**. Its denoiser is dual-output — it predicts an
nDSM `r̂` (height above bare earth) **and** classification logits `ℓ` — trained with

```
L = λ1·L1 + λ2·L2 + λ∇·L∇ + λc·Lc                                  (Eq. 11)
L1 = ‖ĝ − g‖₁,   L2 = ‖ĝ − g‖₂²                                    (Eq. 12)
L∇ = | ‖∇ĝ‖₂ − ‖∇g‖₂ |₁     (edge-aware, raster grid)              (Eq. 13)
Lc = BCE(σ(ℓ), Mα)          (ground-mask classification)          (Eq. 14)
G  = σ(ℓ)·s + (1−σ(ℓ))·(s − r̂)   (gating fusion)                   (Eq. 5)
λ1 = λ2 = 1.0,   λ∇ = 0.1,   λc = 0.1
```

Our loss was the **inverse** of this: classification only (CE+Lovász = nothing
but GrounDiff's Lc), at full weight, with **no regression at all**. GrounDiff
doesn't collapse because the dense, continuous per-point height target has **no
majority shortcut** — predicting 0 for a 15 m tree costs ‖15‖₁ / ‖15‖₂² — so the
shared features are *forced* to encode height-above-ground, which is precisely
what separates ground from non-ground.

## What we implemented (faithful adaptation to a point classifier)
A per-point regression head predicts the continuous nDSM `r = z − DTM(x,y)`
(height above the **GT** bare-earth surface, built from the GROUND-labelled
points), trained with GrounDiff's **L1 + L2** (Eq. 12). The classification term
stays MEEPO's CE+Lovász (this is GrounDiff's Lc / the IoU metric):

```
L = groundiff_cls_weight · (CE + Lovász) + λ1·L1 + λ2·L2   on  (r̂ − r)/ndsm_scale
```

The regression head is **train-only** (it shapes the shared features and is
detached at eval, so inference and the IoU metric are unchanged — still argmax
of the classification logits).

### Documented deviations from GrounDiff (stated honestly)
1. **L∇ (Eq. 13) is omitted** — it is a raster-grid gradient term with no
   analogue on unstructured points, and the smallest term (λ∇=0.1).
2. **The diffusion process and gating fusion G (Eq. 5) are not transferred** —
   they are GrounDiff's raster-generative machinery; we transfer only the
   regression-dominant *loss*, which is the part that prevents the collapse.
3. **cls weight kept at 1.0** (vs the paper's λc=0.1) because per-point IoU is
   *our* deliverable metric and the mask head must train well. Exposed as
   `--groundiff-cls-weight`; set `0.1` for the strict-GrounDiff weighting (which
   makes the regression dominate even more strongly).
4. **The nDSM target is normalised by `ndsm_scale` (default 10 m)** so L1/L2 are
   O(1) and balanced against CE — GrounDiff likewise regresses normalised
   elevations.

### Residual risk
The heads are parallel (no gating), so the regression shapes the features but
does not *directly* constrain the classification logits. In practice the dense
regression gradient dominates the backbone and makes the discriminative
("use the height") solution far easier than the collapse minimum. **If the
classification still collapses, the next step is to lower
`--groundiff-cls-weight` toward 0.1** (regression-dominant, GrounDiff's balance)
and/or add tighter coupling via the gating of Eq. 5.

## Files
- `data/dtm.py::height_above_ground` — continuous per-point nDSM target (NaN where invalid).
- `training/losses.py::GrounDiffLoss` — CE(+Lovász) + L1 + L2.
- `models/segmentation_model.py` — train-only `reg_head` (gated by `use_groundiff_regression`).
- `data/scene_dataset.py`, `data/ptv3_collate.py` — emit + NaN-aware per-voxel aggregate the nDSM target.
- `training/trainer.py` — builds `GrounDiffLoss`, passes `(reg_pred, ndsm)`.
- `utils/config.py`, `scripts/05_train.py` — config fields + CLI flags (in the snapshot-reset list).
- `scripts/smoke_test.py` — step `[8/9]` verifies target/head/loss/gradient/detach.

## Toggle
On by default. Pure MEEPO classification (which collapses) is
`--no-groundiff-regression`.
