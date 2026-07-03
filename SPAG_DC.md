# SPAG-DC ground-misclassification correction

Faithful implementation of **Zhu, Tang, Yang, Li, Xue, Su, Yi**, *An Adaptive
Ground-Point-Cloud Misclassification Detection and Correction Algorithm Based on
Seed-Point Guidance for High-Precision DEM Construction in Complex Terrain*,
**IEEE Sensors Journal 25(21):40399-40411, 2025** (doi:10.1109/JSEN.2025.3615605).

SPAG-DC is the paper's own name: **S**eed-**P**oint-guided **A**daptive **G**round
misclassification **D**etection and **C**orrection. It is a *deterministic, non-learned*
closed-loop post-filter that runs on the initial ground points produced by an upstream
classifier (here: our MEEPO model) and removes **Type-II errors** -- non-ground points
misclassified as ground, i.e. the DTM "spikes". This is the module's only learned input: the
MEEPO model produces the initial ground mask; SPAG-DC itself has no trained parameters.

Code: `meepo_nz/inference/spag_dc.py`. Pure numpy/scipy (CPU), so it adds no GPU risk and
is fully testable offline (`scripts/smoke_test.py` step 8; `/tmp/test_spag.py`).

## The three stages (paper Fig. 2) and where they live

| Paper stage | Equations | Function |
|---|---|---|
| A. Dynamic region growing -> core ground | c = l3/(l1+l2+l3) (1); theta_d = theta0*(1+beta*var_c) (2-3); tau_c = alpha*mu_c (4-5) | `_normals_and_curvature`, `region_growing_core` |
| B. Seed extraction via grid optimisation + MCS purification | grid index (6); weighted gradient (7); refine/merge by density d | `_adaptive_seed_indices`, `_mcs_purify` |
| C. TPS surface fit + distance-threshold correction | f = sum l_i U(r)+a1 x+a2 y+a3, U(r)=r^2 log r (8); k=clip(rho*alpha,kmin,kmax) (9); cut at mu2+n*sigma2 | `_tps_fit`, `_tps_eval`, top-level residual test |

Top-level entry point: `spag_dc_refine(sub_xyz, sub_raw, cfg, return_info=False)` -> refined
labels (1=ground, 0=non-ground). It operates on the predicted-ground subset (`sub_raw==1`).

## Parameter values (the paper's; no guessed values)

| Symbol | Meaning | Value (cfg field) | Source in paper |
|---|---|---|---|
| theta0 | initial normal-angle threshold | 10 deg (`spag_theta0_deg`) | "we adopt a smaller value (theta0 = 10 deg)" |
| alpha | region-growing curvature coeff (tau_c=alpha*mu_c) | 0.5 (`spag_alpha`) | gate opens at 0.4; recommended 0.5-0.7 (Sec. V-B, Table III) |
| beta | angle-relaxation coeff (in [0,1)) | 0.7 (`spag_beta`) | recommended 0.6-0.8 (Sec. V-B, Table III) |
| k | region-growing kNN neighbourhood | 20 (`spag_k`) | "adaptive K ~ 20-25" (Sec. V-C, Table IV) |
| kmin,kmax | local-TPS adaptive neighbourhood bounds | 10,30 (`spag_tps_kmin/kmax`) | tested K range 10-30 (Table IV) |
| n | mu2 + n*sigma2 correction multiplier | 3.0 (`spag_n_sigma`) | ground residuals ~ normal (Bartels & Wei); 3 = the 3-sigma normal-tail (paper does not state n explicitly, so the canonical 3-sigma value is used) |
| min floor | per-grid min elevation-difference | 0.1 m (`spag_min_grid_diff`) | "set to 0.1 m in this article" |
| block | prepartition block size | 20-100 m | "block size is set between 20 and 100 m" (here driven by the scene block tiling) |

The seed-grid base resolution is **density-adaptive** (the paper grids by the average density
`d`, refining a cell to 2x2 when `N_ij > d`): we derive the base cell side from the local 2-D
point density so each cell holds ~8 points, then apply the refine rule. There is no single magic
distance to "guess" -- the rule is the parameter.

## Deviations from the paper (documented, deliberate)

1. **Reclassify, never move/drop.** Detected spikes are relabelled ground->non-ground and kept
   in the cloud (our pipeline is reclassify-only); the paper discards them. Same decision, our
   bookkeeping.
2. **Region growing sources seeds; the TPS residual is the detector.** On rough airborne ground
   the region-growing core is a *subset* of true ground (continuous ground with locally noisy
   normals is not all admitted). We therefore use the core only to build clean seeds (paper
   Stage B draws seeds from it) and let the **TPS-surface residual** (Stage C, the paper's
   "detect and correct misclassified points based on distance thresholds") make the actual
   ground/non-ground decision over all non-seed candidates. This avoids demoting continuous
   ground while still removing spikes, and matches the paper's framing ("seed points + unclassified
   candidates -> residual classification").
3. **MCS purification is elevation-robust.** The seed outlier test fits a local plane and rejects
   seeds whose vertical residual is a high outlier (median + 3*MAD). This reliably removes an
   isolated spike that became a grid-cell minimum (a non-robust SVD-orthogonal plane is pulled by
   the very outlier it should reject); it is faithful in intent to Nurunnabi's Maximum Consistent
   Set (largest consistent subset by distance to a robust local surface).
4. **Per-grid-cell local TPS.** One TPS is fit per occupied grid cell from its k nearest seeds and
   evaluated for all candidates in the cell (the paper's "local TPS interpolation approach"),
   which keeps scene-scale runtime bounded.

## Verified behaviour

Standalone (sloped + rough ground, 12 giant + 12 subtle spikes): **giant 12/12, subtle 12/12,
clean-ground wrongly demoted 0.0%**. Smoke step 8 (60x60 slope, 10 spikes): **10/10 caught,
clean-ground demoted 0.0%**. This is the property the previous fixed-tail refiner violated -- it
demoted ~5% of predicted ground on every tile (spikes or not), raising DTM-RMSE; SPAG-DC only
demotes points that are genuinely above the fitted ground surface.

## Cost

SPAG-DC is heavier than a fixed-tail refiner (paper: ~580 s for ~1M points on CPU; here the
per-cell TPS keeps it bounded). It runs per validation block in training (raw-vs-refined chart)
and at final inference. If per-epoch validation is too slow, run with `--refine off` during
training and apply SPAG-DC only at inference (`scripts/09_infer_large_scenes.py --refine spag_dc`).
