"""Configuration for MEEPO (PTv3 + sparse MoE) training / data / model."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class Config:
    # ---------------- experiment ----------------
    name: str = "meepo_nz_nz_ground"
    out_dir: str = "runs"
    seed: int = 1

    # ---------------- model (Figure 1 / 2) ----------------
    num_classes: int = 2                     # ground (1) / non-ground (0)
    in_features_dim: int = 6                  # PAPER-FAITHFUL input: xyz(3) [PTv3 outdoor uses coord in feat] + return count(1) + return ratio(1) + intensity(1) [the sanctioned Deviation B]. Hand-crafted shallow features are OFF (not in PTv3/MEEPO). Recomputed from flags by expected_feature_dim() at runtime
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 13, 15])  # legacy sphere-mode branch kernel sizes (unused by the PTv3/LitePT backbones; kept for the sphere data path)
    first_subsampling_dl: float = 0.10       # finest grid size (m); 10 cm. PTv3-native full-scene default. CHANGING THIS -> re-run stage 04 (preprocess subsamples the raw cloud at this grid).
    # --- PTv3 full-scene mode (replaces KPConv sphere cropping) ------------------
    scene_mode: bool = True                   # DEFAULT = PTv3-native whole-scene (GridSample+point-budget crop, the MEEPO/Pointcept method). --sphere-mode for the legacy KPConv path (KPConv-style in_radius cylinders + variable batch). FAST (a batch is ~batch_num small spheres) and the prior-DTM raster branch (use_dtm_raster) is active. The previous-year DTM also rides along as the per-point use_prev_dtm channel. Pass --scene-mode for the PTv3-native whole-scene path (receptive field = scene_block_size, but at full 10 cm resolution one block is huge -> needs grad_checkpointing and is much slower per epoch). Receptive field in sphere mode = in_radius (below).
    scene_max_points: int = 204_800           # = MEEPO SphereCrop point_max (Tab/transform: dict(type="SphereCrop", point_max=204800)). Hard cap on points per scene/block per forward. WITH grad_checkpointing ON (default) batch_num=16 of these (~3.3M pts total) fits in ~66 GB of the 96 GB card. At dl=0.1 a 200-300 m block holds >=204800 pts so this caps it (paper-faithful); raise only if you want bigger scenes and have VRAM. Lower if OOM. With --compile set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (the inductor scatter fallback fragments the allocator).
    scene_block_size: float = 300.0           # m; side length of the contiguous block fed per forward (also the PTv3-native inference tiling grid). THIS is the spatial receptive field. At dl=0.10 (kept everywhere) a full 300 m block at ~13 pts/m^2 is ~1.17M pts, so it is count-cropped to scene_max_points (~190 m effective per block); blocks tile the scene with a margin ring so coverage is complete. Raise for more reach if VRAM allows.
    scene_block_margin: float = 8.0           # m; ABSOLUTE floor for the context ring around each inference block.
    scene_block_margin_frac: float = 0.10     # context ring is at least this fraction of the block size (>=10%); effective margin = max(scene_block_margin, frac*scene_block_size).
    scene_vote_step_m: float = 50.0           # SparseGF soft-voting: grid step (m) between disc centres at val/test/inference. Each disc classifies its circular CENTRAL region of radius (sqrt2/2)*step (circumscribes the cell -> full coverage + edge/corner overlap), with the nearest scene_max_points as context. Overlapping central regions are soft-voted (averaged softmax). SparseGF uses 50 m. Smaller = more overlap/votes but slower; must stay below the context disc radius.
    scene_min_points: int = 100               # blocks/tiles with fewer points fall back to the nearest-N points.
    scene_val_tiles: int = 32                  # per-epoch validation scores ONE scene_block_size window per sampled tile (KPConv's bounded validation_size regions, region size = our training region), via the deployed full-res inference (predict_scene), on a fixed evenly-spaced subset of this many tiles -> a stable, fast, paper-faithful number. KPConv never validates on whole scenes; full-tile coverage is reserved for the FINAL test (evaluate_split, max_tiles=None). Lower if eval is slow; 0/None = all tiles (whole-tile, slow).
    tta: bool = False                          # test-time augmentation: average softmax over z-rotations 0/90/180/270 (Pointcept SemSegTester aug_transform). Applied at the FINAL test (evaluate_split) and inference (--tta), NOT per-epoch val (kept fast). Rotates BOTH the cloud and the georeferenced prior raster together (rot90 + georef), matching how augment_tile rotates the prior patch at train time.
    scene_cache_tiles: int = 2                # per-worker LRU cache of loaded tiles. Small on purpose: random sampling rarely re-hits a tile, and in scene mode each cached tile is a whole (large) cloud, so a big cache x many workers exhausts host RAM.
    # --- SPAG-DC ground-misclassification correction (post-classification, at inference) ---
    # SPAG-DC post-filter (Zhu, Tang et al., IEEE Sensors J. 25(21):40399-40411, 2025,
    # doi:10.1109/JSEN.2025.3615605): a deterministic, non-learned closed-loop module on
    # the model's initial ground points that detects & corrects Type-II errors (non-ground
    # misclassified as ground = DTM spikes). Region-growing core -> density/gradient adaptive
    # seed grid -> Maximum-Consistent-Set seed purification -> local thin-plate-spline surface
    # -> reclassify any candidate whose vertical residual to the surface exceeds mu2 + n*sigma2.
    # Values below are the paper's (no guessed params); see SPAG_DC.md for the eq mapping.
    refine_method: str = "spag_dc"            # 'spag_dc' | 'off' (off = raw argmax, no post-processing)
    spag_theta0_deg: float = 10.0             # theta0, initial normal-angle threshold (paper: "we adopt a smaller value, theta0 = 10 deg").
    spag_alpha: float = 0.5                   # alpha, region-growing curvature coeff tau_c=alpha*mu_c (eq 4). Gate opens at 0.4; paper recommends 0.5-0.7.
    spag_beta: float = 0.7                    # beta, region-growing angle-relaxation coeff theta_d=theta0*(1+beta*var_c) (eq 2), in [0,1). Paper recommends 0.6-0.8.
    spag_k: int = 20                          # k, region-growing kNN neighbourhood (paper: adaptive K ~ 20-25).
    spag_n_sigma: float = 3.0                 # n in the mu2+n*sigma2 correction cut. Ground residuals ~ normal (Bartels & Wei); 3 = the 3-sigma rule (99.7% ground retained).
    spag_min_grid_diff: float = 0.1           # m; minimum per-grid elevation-difference floor (paper: "set to 0.1 m"). Cells whose max residual < this are kept all-ground.
    spag_tps_kmin: int = 10                   # TPS adaptive-neighbourhood lower bound k=clip(rho*scale, kmin, kmax) (eq 9; paper tested K 10-30).
    spag_tps_kmax: int = 30                   # TPS adaptive-neighbourhood upper bound.
    # LEARNED SPAG-DC: a per-scene regime head (MLP on pooled backbone features)
    # predicts the SPAG-DC control globals (theta0/alpha/beta/n_sigma/base_res/min_floor),
    # supervised by an "oracle" target from each scene's GT-ground terrain stats. At
    # inference the predicted globals replace the fixed spag_* defaults above (per scene).
    spag_learned: bool = False                # OFF by default -> the geometry SPAG-DC above. --spag-learned turns on the learned regime head.
    spag_regime_weight: float = 0.1           # weight of the (train-only) regime smooth-L1 auxiliary on the total loss.
    spag_regime_hidden: int = 64              # hidden width of the regime MLP head.
    conv_radius: float = 2.1                  # KPConv neighbour radius in units of dl. KPConvX (Thomas 2024) finds 2.1 OPTIMAL: larger radii make each kernel point cover too much area -> "less descriptive, missing finer details". KPConv-rigid default is 2.5. Receptive field is meant to come from DEPTH (num_layers), NOT a wide radius. At 2.1 there are only ~20 neighbours/point at every layer, so neighbor_limit ~24 barely truncates and per-sphere VRAM is tiny (leaves room for big batch + many layers). Was 8.0 (a wide-radius compromise that hurt fidelity and forced ~1000-neighbour caps).
    kp_extent: float = 1.0                   # sigma: kernel-point influence = Sigma*dl, Sigma=1.0 (KPConv paper, cross-validated); units of dl
    # ---- depth-driven receptive field (KPConvX philosophy: small radius + many layers) ----
    num_layers: int = 5                       # number of encoder RESOLUTION LEVELS (= decoder stages + 1). Sets the PTv3/LitePT encoder pyramid depth. Pyramid (points/neighbors/lengths = num_layers; pools/upsamples = num_layers-1) and calibrate_neighbor_limit are threaded with this. NOTE for airborne data: 5 levels at dl0=0.10 -> coarsest grid 1.6 m; spatial context is set by in_radius (below), NOT auto conv-reach. To widen the spatial receptive field beyond KPConvX-L, extend layer_blocks (e.g. [3,3,9,12,12,9,3,3]) — a documented deviation. CHANGING in_radius -> re-run stage 04; run scripts/smoke_test.py after any depth change.
    init_channels: int = 64                   # width of the first encoder layer. KPConvX uses 64.
    channel_scaling: float = 1.41             # per-layer width growth: ch[l] = round(init_channels * channel_scaling**l). KPConvX uses 1.41 (~sqrt 2) so going deep does NOT explode width (x2 would give 4096+ channels at 7 layers). 1.41 @ 7 layers, init 64 -> [64,90,127,179,253,357,503]. Set 2.0 with num_layers 5 to reproduce the original [64,128,256,512,1024] exactly.
    use_batch_norm: bool = True
    norm: str = "bn"                          # backbone (PTv3/LitePT) + raster-CNN normalization: 'bn' = BatchNorm (MEEPO Tab.3c default, best for multi-domain). 'ln' = LayerNorm (backbone) + GroupNorm (raster CNN), both batch-independent -> use with micro-batch 1 / heavy grad-accum, where BN statistics would otherwise collapse to a single scene per forward.
    batch_norm_momentum: float = 0.1        # KPConvX-L bn_momentum=0.1 (PyTorch convention; train_S3DIS.py). Was 0.02 (KPConv-era).
    # Normalisation inside the KPConv blocks. 'bn' = KPConv's BatchNorm1d (paper,
    # default). 'gn' = batch-statistics-free GroupNorm: eval==train, which removes
    # the transient validation-mIoU collapses caused by BatchNorm running-stat
    # drift under small-batch decimation (the Point-SCT BN->norm lesson). Opt-in.
    norm_type: str = "bn"                    # 'bn' | 'gn'
    num_norm_groups: int = 8                 # GroupNorm groups (clamped to a divisor of each width)
    dropout: float = 0.5
    drop_path_rate: float = 0.3              # stochastic depth (per-cloud DropPath), ramped 0->this across the encoder blocks. PTv3/LitePT use 0.3.
    grid_scaling: float = 2.2                # sphere-mode multiscale-pyramid grid growth per level (legacy data path). CHANGING THIS -> re-run stage 04.
    head_dropout: float = 0.0                # dropout before the final seg linear. KPConvX seg head has none (0.0); set >0 to regularize.

    # ---------------- shallow features (Section 3.1) ----------------
    feature_knn: int = 16                    # k nearest neighbours for curvature / mean elevation
    use_mean_elevation: bool = False         # OFF: hand-crafted geometric feature, NOT in PTv3/MEEPO. (--use-mean-elevation to ablate)
    use_curvature: bool = False              # OFF: hand-crafted geometric feature, NOT in PTv3/MEEPO
    use_higher_moments: bool = False         # OFF: hand-crafted geometric feature, NOT in PTv3/MEEPO
    use_return_features: bool = True         # DEVIATION (not in the paper): per-point laser-return cue - number_of_returns (a single count channel). --no-return-features to disable
    use_return_ratio: bool = True            # DEVIATION (not in the paper): normalised return ratio return_number/number_of_returns in (0,1]; ~1 = last/only return (likely ground), <1 = earlier return (canopy). Complements the count; computed per raw point then averaged. --no-return-ratio to disable
    use_intensity: bool = True               # per-point intensity channel (not in the paper); radiometric ground/vegetation cue, standardised by norm_stats. --no-intensity to disable
    intensity_log: bool = True               # log1p-compress intensity BEFORE standardising. LiDAR intensity is heavy-tailed (16-bit: most ~0-2000, tail to ~60000); a plain z-score is dominated by the tail and squashes the bulk to ~0. log1p restores resolution. Applied in assemble_features so norm_stats + train/val/test/inference all match. --no-intensity-log to disable
    use_rgb: bool = False                    # RGB ablation only (off; aerial LiDAR has no RGB)
    # The paper's input is [n, 11] = xyz coordinates + 8 shallow features (Sec 3.2:
    # "the model utilizes xyz coordinates and shallow features ... as inputs ...
    # mapped from [n, 11] to [n, d]"). So the 3 local coordinates are fed as the
    # first input channels (--no-xyz-feature drops them for the KPConv-style setup).
    use_xyz_in_features: bool = True
    # KPConv's constant-1 channel. The paper uses xyz instead of this, so it is OFF
    # by default; --constant-feature re-enables it (KPConv-PyTorch issue #90 setup).
    use_constant_feature: bool = False
    # the single sanctioned deviation: the previous-year ground DTM, fed to the
    # network as a RASTER through a small 2D CNN branch whose features are
    # bilinearly sampled per point and concatenated to the shallow features.
    use_dtm_raster: bool = True              # the deviation (raster branch); --no-dtm-raster to disable
    raster_scene_patch_size: int = 128       # whole-scene mode: the previous-year prior raster is cropped to each block window and resampled to this many px (the fully-conv GrounDiff CNN then runs once per block; per-point bilinear sample). ~1 m/px over a 128 m block.
    dtm_patch_size: int = 64                 # DTM raster patch (px); resolves the 1 m prev-year DTM over a 50 m tile (50 cells) with power-of-2 headroom
    dtm_feat_dim: int = 8                     # DTM CNN output channels; matches the 8 shallow features (comparable-width terrain-context feature)
    # legacy point-based alternative (z - DTM_prev as one scalar channel); off by
    # default now that the raster branch exists, but still selectable.
    use_prev_dtm: bool = True                 # ON: per-point height above the previous-year DTM (z - prevDTM, "GrounDiff/Ground-Awareness-like" relative elevation). Baked into each tile at stage 04 from the matched prior raster, so it works in scene mode (where the sphere-only raster branch is off). Zeros where no prior DTM covers a point.
    dtm_resolution: float = 1.0              # m (1 m DTM, matching LINZ dem_1m)
    min_dtm_coverage: float = 0.0            # 0 = keep all tiles. >0 = at stage 04, DROP any tile whose sphere centres are <this fraction covered by the previous-year DTM (e.g. 0.9 keeps only tiles >=90% covered). Safety net to guarantee prev-DTM overlap; with spatially-matched downloads coverage is already ~1.0.

    # ---------------- data / tiling ----------------
    data_root: str = "data"
    tile_size: float = 50.0                  # m; used only for the DTM-raster patch extent fallback / legacy
    # ---- KPConv input-sphere sampling (replaces tiling; faithful to KPConv KP-FCNN) ----
    auto_in_radius: bool = False             # KPConvX sizes the input region by a fixed in_radius/sub_size ratio (~50: S3DIS 2.1 m @ 4 cm), NOT by conv reach. So leave OFF for the KPConvX network and set in_radius below. (When True, resolve_geometry() instead sets in_radius = conv_radius*dl0*2^(num_layers-1) - the old depth-driven sizing, suited to the original MultiKPFCNN.)
    in_radius: float = 6.0                    # m; radius of each input CYLINDER (sphere mode). PTv3 mixes information across the WHOLE cylinder via serialised attention + grid-pool levels, so the radius only needs enough points for context, not a wide ball - 6 m (12 m across) keeps spheres small/fast (~1.5-2k pts/sphere at dl=0.1) and yields more samples per epoch. The raster patch auto-covers 2*in_radius. CHANGING THIS -> re-run stage 04 (candidate cylinders cand_idx are baked at in_radius); --in-radius N. Ignored when scene_mode=True.
    sphere_center_spacing: float = 5.0        # m; [LEGACY sphere mode only] grid spacing of candidate cylinder centres.
    tile_cache_size: int = 4                 # max preprocessed tiles kept in each loader's LRU cache (each ~ one tile's arrays). Bounds RAM; raise if you have headroom and want fewer reloads.
    sphere_min_points: int = 100             # cylinders with fewer subsampled points fall back to the nearest-N points
    infer_batch_spheres: int = 16            # inference: spheres per forward pass (batched as a PTv3 multi-cloud batch). Higher = faster on a big GPU; lower if VRAM-bound. --infer-batch
    tile_overlap: float = 5.0                # m; legacy (unused by the sphere pipeline)
    max_points_per_tile: int = 400000        # raw safety bound only; preprocessing grid-subsamples each tile to first_subsampling_dl (full voxel coverage, no random decimation), so this rarely binds and does not set the resolution
    max_train_tiles: int = 0                  # 0 = use ALL train tiles. >0 = train on only this many train tiles (deterministic, seeded), to shrink the working set so it fits host RAM/page-cache. Each kept tile is full-fidelity (no compression); only val/test always use all tiles. Lossless per-sample alternative to dtype/compression.
    val_fraction: float = 0.10        # 80/10/10 split (train/val/test)
    test_fraction: float = 0.10
    # first_subsampling_dl should be ~ the survey's NOMINAL point spacing. With
    # --auto-dl, stage 04 estimates it from density (sqrt(area / n_points)), which
    # is robust to coincident/overlapping returns - the literal nearest-neighbour
    # minimum can be ~0 (duplicate points) and must NOT be used. min_subsampling_dl
    # floors the estimate so it can never collapse toward zero.
    auto_subsampling_dl: bool = False        # stage 04 derives first_subsampling_dl from data
    min_subsampling_dl: float = 0.10         # hard floor (m) for the estimate

    # NZ download (scripts/01).
    #  - point clouds come from OpenTopography's anonymous bulk S3 (no API key);
    #    nz-elevation has DEM/DSM rasters ONLY and is just an optional DTM source.
    download_budget_gb: float = 100.0
    download_source: str = "opentopography"   # opentopography (NZ point clouds, default) | pnoa_es (Spanish PNOA) | nz_elevation_dem
    tiles_per_pair: int = 6                    # LAZ tiles fetched per survey per round-robin visit (opentopography only)
    download_workers: int = 16                 # parallel S3 download streams (I/O-bound; can exceed core count)
    # ---- OpenTopography density scan (download_source="opentopography") ----
    #   Keep only year-pairs whose CURRENT (training) capture is post-ot_min_year
    #   AND whose sampled point density (read from LAS headers) falls in the band.
    #   This selects naturally low-density captures (so dl0=0.10 / in_radius=5.0
    #   stay valid) that ALSO have a previous-survey twin for the prev-year DTM.
    ot_min_year: int = 2020
    ot_min_density: float = 2.0                # pts/m^2 (inclusive)
    ot_max_density: float = 9.0                # pts/m^2 (inclusive) - 2-9 band spans a wide REGIONAL range of NZ captures while excluding the densest recent/urban surveys (>9). Diversity comes from regional spread (round-robin over areas in stage 01), not from a single dense region.
    ot_density_sample_tiles: int = 6           # LAS headers sampled per dataset to estimate density
    # ---- Spanish PNOA-LiDAR source (download_source="pnoa_es") ----
    #   flai-ai open-lidar-data bucket, COPC, EPSG:25829 (UTM 29N = Galicia/NW Spain),
    #   2.8 pts/m^2, CC BY 4.0. Each tile is self-paired (its own prior-DTM source).
    es_prefix: str = "data/ES/CNIG/Lidar_2015-2021_epsg25829/copc/"   # 2nd coverage = training clouds (2.8 pts/m^2)
    es_prev_prefix: str = "data/ES/CNIG/Lidar_2008-2015_epsg25829/copc/"  # 1st coverage = previous-survey DTM (0.9 pts/m^2)
    es_pairing: str = "cross"                  # "cross" = pair each training tile to the 2008-2015 tile at the same UTM grid cell; tiles with NO prior tile are SKIPPED (not downloaded), never self-paired. "self" = each tile is its own prior-DTM source (single-coverage).
    es_epsg: int = 25829
    es_region_filter: str = ""                 # keep only CURRENT tiles whose basename contains this (e.g. "GAL-W"); "" = every tile under es_prefix. Prior coverage is matched by grid key, not filtered.
    es_sample_seed: int = 1234                 # deterministic shuffle before the budget cut -> spatially-spread, resumable selection
    regions: List[str] = field(default_factory=list)   # OPTIONAL area-key substring filter for the OT density scan and the DEM downloader; empty = scan every region (the density band is the selector). Use --regions <name...> to restrict.

    # Sphere sampling is UNIFORM at random over every candidate cylinder across all
    # downloaded clouds. Regional diversity comes from stage 01's round-robin
    # interleave over areas (so no single region dominates the corpus) - there is no
    # terrain-type up-weighting. Train / val / test /
    # inference therefore all see the same natural sphere distribution.
    use_region_balanced_sampler: bool = False  # True -> weight spheres so every SOURCE CLOUD (region) contributes equally per epoch (make_region_balanced_sampler). Default False = uniform over all spheres.

    # data augmentation (KPConv recipe: vertical rotation, anisotropic scaling,
    # X-symmetry, Gaussian jitter). Applied per training tile; features are
    # recomputed and the DTM patch is transformed to stay aligned.
    mix_prob: float = 0.8                    # MEEPO Mix3D probability (configs/scannet/semseg-meepo.py: mix_prob=0.8). Pointcept point_collate_fn merges clouds PAIRWISE (offset rewrite only -> backbone treats each pair as one scene; points/labels untouched). TRAIN ONLY (val uses 0). --mix-prob N / --no-mix3d.
    use_augmentation: bool = True
    augment_rotation_z: bool = True          # random rotation about the vertical axis
    augment_anisotropic: bool = False        # MEEPO RandomScale([0.9,1.1]) is ISOTROPIC (one factor for all axes)
    augment_scale_min: float = 0.9           # MEEPO/PTv3 RandomScale [0.9, 1.1] (configs/point_moe/indoor.py)
    augment_scale_max: float = 1.1           # MEEPO/PTv3 RandomScale [0.9, 1.1]
    augment_flip_x: bool = True              # MEEPO/PTv3 RandomFlip p=0.5 (x-axis)
    augment_flip_y: bool = True              # MEEPO/PTv3 RandomFlip p=0.5 also flips the y-axis
    augment_noise: float = 0.005             # m, Gaussian jitter -- MEEPO/PTv3 RandomJitter sigma=0.005
    augment_noise_clip: float = 0.02         # m, jitter clip -- MEEPO/PTv3 RandomJitter clip=0.02
    augment_dropout_ratio: float = 0.2       # MEEPO RandomDropout dropout_ratio=0.2 (drop this fraction of points)
    augment_dropout_prob: float = 0.2        # MEEPO RandomDropout dropout_application_ratio=0.2 (per-sample prob it fires). Density-robustness; does NOT move (x,y) so the per-point prior raster stays consistent.
    augment_tilt_xy: float = 0.0             # rad; MEEPO RandomRotate x/y = +-pi/64 (~0.049). DEFAULT 0 (off): a tilt mixes z into (x,y) by a z-dependent amount the 2D georeferenced prior (Deviation A) cannot follow, desyncing the prior by ~relief*sin(angle). Set to 0.04908 to match MEEPO exactly.
    augment_elastic: bool = False            # MEEPO ElasticDistortion. DEFAULT off: warps the ground SURFACE by up to ~1.6 m AND moves (x,y), desyncing the prior. Enable (--augment-elastic) to match MEEPO exactly.
    augment_elastic_params: tuple = ((0.2, 0.4), (0.8, 1.6))  # MEEPO distortion_params=[[0.2,0.4],[0.8,1.6]] (granularity_m, magnitude_m) pairs

    # ---------------- training ----------------
    # Training recipe below follows KPConv (Thomas et al. 2019); the paper states
    # no hyperparameters. Values match the KPConv author's guidance for low-density
    # outdoor data (KPConv-PyTorch issue #90): lr 1e-2, momentum 0.98, grad-clip 100,
    # lr_decay = 0.1**(1/100) per epoch, batch_num for KP-FCNN segmentation.
    epochs: int = 800                        # MEEPO: epoch=800 (configs/scannet/semseg-meepo.py). With epoch_steps=100 -> 80k optimiser steps, matching MEEPO ScanNet (~1201 scenes / batch 12 ~= 100 steps/epoch x 800). best-val checkpoint (model_best.pt) guards late overfit.
    # STEPS PER EPOCH (verified vs MEEPO Table 10): MEEPO is ITERATION-budgeted,
    # not fixed-steps-per-epoch. It trains 140k total optimiser iters (indoor joint) /
    # 180k (indoor-outdoor) over 120 epochs => ~1167-1500 steps/epoch (data-determined:
    # len(loader) = num_samples / batch). Pointcept's true convention is a full pass per
    # epoch. Here the sphere corpus is far larger than ScanNet, so a full pass x epochs is
    # impractical; we keep KPConv's fixed-step epoch. Note the TOTAL budget still lands in
    # the paper's range: e.g. --epochs 500 x epoch_steps 500 = 250k optimiser steps
    # (>= the paper's 140-180k). To match the paper's epoch granularity exactly use
    # ~epochs 120 x epoch_steps 1200 (~144k, indoor) or 1500 (~180k, indoor-outdoor);
    # set epoch_steps 0 for a true full pass if your tile set is small.
    epoch_steps: int = 100                   # MEEPO match: 100 optimiser steps/epoch x 800 epochs = 80,000 total -> reproduces MEEPO's OneCycle horizon (ScanNet ~100 steps/epoch). total iters = epochs*epoch_steps. 0 = full pass.
    log_every_steps: int = 0                 # step-log cadence in the train loop. 0 (default) = adaptive ~10 logs/epoch (every n_iter//10 steps); set e.g. 20 to print a running-loss/lr/throughput/eta line every 20 optimiser steps. The first and last step of each epoch always log.
    validation_size: int = 50                # validation steps per epoch (KPConv guidance); 0 = full
    grad_accum_steps: int = 2                # gradient accumulation: micro-batches per optimiser step. Effective batch = batch_num * grad_accum_steps = 12 (default 6x2) = MEEPO batch_size ("total bs in all gpus", configs/scannet/semseg-meepo.py). On ONE GPU this reproduces MEEPO's cross-GPU gradient averaging because the backbone is LayerNorm/RMSNorm-only -- no BatchNorm to desync across micro-batches. Scheduler steps per optimiser step so the LR schedule is unaffected.
    batch_num: int = 6                       # clouds/blocks per forward. With grad_accum_steps=2 the EFFECTIVE batch is 12 = MEEPO batch_size (configs/scannet/semseg-meepo.py: batch_size=12, "total bs in all gpus"). Scene mode: 6 x scene_max_points(204800) ~= 1.2M pts/forward ~= 47 GB of 96 with grad_checkpointing (12-in-one-forward ~= 91 GB is borderline, hence 6x2). Sphere mode: spheres are tiny so this fits trivially.
    # KPConv variable batch size (supplementary, Sec. A): pack input spheres into a
    # constant total-points budget so a batch AVERAGES batch_num spheres. This is how
    # KPConv copes with clouds of varying size/density - dense spheres -> fewer per
    # batch, sparse -> more, total points (hence VRAM / step time) ~constant. Keeps the
    # paper's batch_num while absorbing NZ density variation. --fixed-batch to disable.
    variable_batch: bool = True
    batch_limit: int = 0                     # total input points per variable batch; 0 = auto-calibrate so the average batch = batch_num spheres
    neighbor_limit: int = 20                 # SCALAR fallback cap on radius neighbours/point. Used only when neighbor_limits (below) is empty, or by legacy/inference call sites. 0 = uncapped.
    # Per-level neighbour caps = KPConvX-L's calibrated neighbor_limits (train_S3DIS.py).
    # Faithful AND faster: fewer neighbours at the dense fine levels -> smaller M x H
    # neighbour matrices and cheaper KD-tree queries. build_multiscale_batch uses
    # neighbor_limits[l] per level when this list is non-empty; set [] to fall back to
    # the scalar neighbor_limit (e.g. when --calibrate-neighbors derives a single cap).
    neighbor_limits: List[int] = field(default_factory=lambda: [12, 16, 20, 20, 20])
    neighbor_percentile: float = 98.0        # --calibrate-neighbors sets neighbor_limit to this percentile of the per-layer neighbour distribution (cap = p98 of the data)
    neighbor_limit_max: int = 0              # fixed ceiling on the calibrated cap; 0 = none (cap = the p{neighbor_percentile} value, no artificial clamp). Set > 0 only to hard-bound VRAM on a smaller card.
    num_workers: int = 16                     # parallel DataLoader workers (multiscale-batch build is CPU-bound); raise toward core count to keep the GPU fed
    dataloader_persistent: bool = False       # keep workers alive across epochs. False (default) re-forks them each epoch, which RESETS the gradual per-worker copy-on-write RAM creep so it can't compound into an OOM over a long run. Set True only if RAM is comfortably flat.
    dataloader_prefetch: int = 2              # batches prefetched per worker; lower to 1 to cut host/pinned RAM if memory is tight.
    gpu_neighbors: bool = False               # build the multiscale neighbour/pool/upsample index tensors ON THE GPU (workers only grid-subsample). Keeps the large index arrays off host RAM and out of the worker->main pipe. Output is identical to the CPU path (validated). Default False = CPU path.
    gpu_neighbor_qchunk: int = 4096           # (gpu_neighbors only) query rows per cdist tile in the on-device neighbour search. Bounds the (nq, ns) distance matrix to (qchunk, ns) so a dense sphere (in_radius/dl0 can pack tens of thousands of points) can't spike GPU memory. Output-identical to any chunk size (per-row topk/argmin). Lower if you still see memory spikes; raise for marginally fewer kernel launches.
    learning_rate: float = 1e-2              # KPConv (SGD)
    momentum: float = 0.98                   # KPConv (SGD)
    weight_decay: float = 1e-3               # KPConv (SGD)
    # Optimizer: MEEPO reference recipe (configs/point_moe/indoor.py):
    # AdamW lr=0.005 (optimizer base; OneCycle OVERRIDES it) wd=0.05; OneCycleLR
    # max_lr=[0.002, 0.0006] (head, block) pct_start=0.05 div_factor=10 final_div_factor=1000.
    # The EFFECTIVE peak LR is the OneCycle max_lr (head 0.002), NOT the 0.005 base.
    optimizer: str = "adamw"                 # 'adamw' (MEEPO) | 'sgd' (legacy)
    adamw_lr: float = 0.006                   # OneCycle head max_lr (effective peak). MEEPO: max_lr=[0.006 head, 0.0006 block]; AdamW base overridden by OneCycle.
    adamw_weight_decay: float = 0.05         # MEEPO: weight_decay 0.05 (both indoor and indoor+outdoor configs).
    adamw_betas: tuple = (0.9, 0.999)        # KPConvX-L AdamW betas
    adamw_eps: float = 1e-8                  # KPConvX-L AdamW eps
    block_lr_scale: float = 0.1              # MEEPO: backbone ("block") trains at 0.1x the head peak -> max_lr=[0.006, 0.0006] (param_dicts keyword="block", lr=0.0006). 1.0 disables.
    block_lr_keyword: str = "block"          # MEEPO param_dicts: only params whose qualified name contains this keyword get the block LR (the enc/dec Mamba blocks).
    warmup_epochs: int = 6                   # [cosine schedule only] linear LR warmup. OneCycle uses pct_start instead.
    # OneCycleLR (MEEPO configs/point_moe/indoor.py): the faithful schedule.
    onecycle_pct_start: float = 0.05         # fraction of training spent ramping up (MEEPO pct_start=0.05).
    onecycle_div_factor: float = 10.0        # initial_lr = max_lr / div_factor (MEEPO div_factor=10).
    onecycle_final_div_factor: float = 1000.0  # min_lr = initial_lr / final_div_factor (MEEPO final_div_factor=1000).
    loss_lovasz: bool = True                 # MEEPO loss = CrossEntropy + Lovasz-softmax, both loss_weight 1.0 (configs/point_moe/indoor.py, verified vs paper). ON by default to match MEEPO. --no-lovasz for plain CE.
    lovasz_weight: float = 1.0               # weight on the Lovasz-softmax term (if enabled)
    grad_checkpointing: bool = True           # recompute PTv3 encoder/decoder block activations in backward instead of storing them (~10x less activation memory, ~20-30% slower). THE way to train large full-resolution scene blocks (keep dl=0.1) within 96 GB. --no-grad-checkpoint to disable if you have VRAM headroom and want speed.
    checkpoint_granularity: str = "block"     # recompute granularity when grad_checkpointing is on: 'stage' (whole stage as one segment: least recompute, most VRAM), 'block' (per block; default = current behaviour), 'layer' (each block's xCPE/Mamba/MLP separately: most recompute, least VRAM -> for small-VRAM cards). Overridden to 'none' when grad_checkpointing is off.
    mask_uncovered_prev_dtm: bool = True      # when a previous-year prior does not cover a point (NoData hole or outside the raster extent), zero that point's z-prevDTM feature instead of extrapolating the raster's edge value. Keeps a partial / hand-crafted prior raster from injecting phantom deviation signal; the raster branch still sees the coverage channel. Set False for the legacy extrapolate-to-edge behaviour.
    # --- GrounDiff nDSM regression (Dhaouadi et al. 2025, Eqs. 11-12) ----------
    # ON by default: the ground problem is regression-dominant. A per-point head
    # regresses the continuous nDSM r = z - DTM(x,y) (height above bare earth)
    # with L1+L2; the dense height target has no majority-class shortcut, so it
    # prevents the predict-all-ground collapse of a classification-only loss.
    # Total loss = groundiff_cls_weight*(CE[+Lovasz]) + l1*L1 + l2*L2 on nDSM/scale.
    use_groundiff_regression: bool = False   # OFF: over-corrected to all-non-ground (regression dominated the shared features without GrounDiff's gating to couple it to the mask). Pure MEEPO CE+Lovasz is the active loss. The GrounDiff head/loss is kept in the code (gated) for ablation; --groundiff-regression re-enables it.
    groundiff_l1_weight: float = 1.0         # lambda1 (Eq. 11; paper 1.0) -- L1 preserves sharp ground
    groundiff_l2_weight: float = 1.0         # lambda2 (Eq. 11; paper 1.0) -- L2 smooths homogeneous regions
    groundiff_cls_weight: float = 1.0        # weight on the CE(+Lovasz) mask term. Paper uses lambda_c=0.1; kept at 1.0 here because per-point IoU is OUR metric. Set 0.1 for strict-GrounDiff weighting.
    ndsm_scale: float = 10.0                 # m; nDSM is normalised by this so L1/L2 are O(1) and balanced vs CE (GrounDiff regresses normalised elevations)
    ndsm_dtm_res: float = 1.0                # m; resolution of the per-sample GT-ground DTM used to derive the nDSM target
    ndsm_min_ground: int = 8                 # min GT-ground points in a sample to compute the nDSM target (else skipped)
    dtm_rmse_res: float = 1.0                # m; DTM-RMSE grid resolution. DEMs of GT-ground and predicted-ground built by triangulation (Delaunay-linear, as in OpenGF's "LAS Dataset To Raster"); RMSE taken between them. (OpenGF uses 0.5 m; 1.0 m kept here by choice.)
    # --- KPConvX-L 1-cycle LR schedule (train_S3DIS.py cyc_*): start 1e-4, ramp x10 to
    #     peak 5e-3 over 30 epochs, 5-epoch plateau, then /10 every 120 epochs. ----
    kpx_lr_start: float = 1e-4               # cyc_lr0 (start = minimum LR)
    kpx_lr_max: float = 5e-3                 # cyc_lr1 (peak LR)
    kpx_lr_warmup_epochs: int = 30           # cyc_raise_n (epochs to ramp lr0 -> lr1)
    kpx_lr_plateau_epochs: int = 5           # cyc_plateau (epochs held at the peak)
    kpx_lr_decay10_epochs: int = 120         # cyc_decrease10 (epochs to divide LR by 10)
    lr_decay: float = 0.9772372              # = 0.1**(1/100) per epoch, KPConv's literal schedule
    # LR schedule: 'onecycle' = MEEPO's OneCycleLR (DEFAULT, faithful; max_lr=[0.002,0.0006],
    #   pct_start=0.05, cos anneal). 'cosine' = warmup+cosine approximation. 'onecycle_kpx' =
    #   KPConvX 1-cycle. 'exp' = KPConv per-epoch multiplicative decay.
    lr_schedule: str = "onecycle"               # MEEPO: AdamW + OneCycleLR. Stepped per optimiser step.
    grad_clip_norm: float = 1.0              # MEEPO / PTv3: clip_grad=1.0 (configs set clip_grad=1.0). (KPConv legacy used 100 = effectively no clip.)
    class_weights: Optional[List[float]] = None   # explicit per-class weights override
    # KPConv's segmentation_loss is an *unweighted* softmax cross-entropy (reduce_mean),
    # and no reference config (PTv3, MEEPO, OpenGF) uses class weighting. Inverse-frequency
    # weighting up-weights the minority class and, since ground is usually the
    # majority in NZ terrain, biases the model toward predicting non-ground (it
    # collapses ground recall). So the default is "none" (faithful to KPConv);
    # "inverse" re-enables inverse-frequency weighting if a dataset truly needs it.
    loss_class_balance: str = "none"               # "none" (KPConv) | "inverse"
    amp: bool = True                          # mixed precision (good on Blackwell)
    amp_dtype: str = "bf16"                    # "bf16" (fp32 range, no overflow) or "fp16"

    # ---------------- per-epoch outputs ----------------
    save_checkpoint_every_epoch: bool = True
    render_errors_every_epoch: bool = True
    write_laz_every_epoch: bool = True
    n_vis_tiles: int = 12                     # how many areas/spheres to render / classify each epoch
    vis_full_area: bool = True               # per-epoch gallery renders voted full AREAS (sphere-voting over a window) instead of single 16 m input spheres - far more legible at scene scale (costs more: ~ (area/spacing)^2 forward passes per area)
    vis_area_size: float = 200.0             # m; side of each per-epoch gallery area window. 200 m (4x the area of the old 100 m) so scenes show real landscape context; sphere-voting cost scales ~(area/spacing)^2, but only n_vis_tiles scenes are rendered per epoch (eval, no grad).
    vis_sphere_spacing: float = 10.0         # m; sphere-centre spacing used for the per-epoch GALLERY voting ONLY (sphere-mode models). The gallery renders to a ~240 px relief over vis_area_size (~1 m/px), so 10 m voting is visually identical to the 5 m training spacing but ~4x cheaper. 0 = reuse sphere_center_spacing. (Ignored for scene-mode models, which render via one-forward-per-block PTv3-native inference.)

    # ------------------------------------------------------------------ #
    # MEEPO (PTv3 + sparse Mixture-of-Experts) backbone
    # ------------------------------------------------------------------ #
    # Space-filling-curve orders used by serialized attention (PTv3 default).
    moe_order: tuple = ("z", "z-trans", "hilbert", "hilbert-trans")
    shuffle_orders: bool = True
    stem_kernel_size: int = 5            # embedding stem SubMConv kernel (paper: 5)
    # Encoder / decoder shape (MEEPO-L defaults).
    enc_stride: tuple = (2, 2, 2, 2)
    enc_depths: tuple = (2, 2, 2, 6, 2)
    enc_channels: tuple = (32, 64, 128, 256, 512)
    enc_num_head: tuple = (2, 4, 8, 16, 32)
    enc_patch_size: tuple = (1024, 1024, 1024, 1024, 1024)
    dec_depths: tuple = (2, 2, 2, 2)
    dec_channels: tuple = (64, 64, 128, 256)
    dec_num_head: tuple = (4, 4, 8, 16)
    dec_patch_size: tuple = (1024, 1024, 1024, 1024)
    mlp_ratio: float = 3.0                   # MEEPO: MLP hidden = 3x channels (semseg-meepo.py)
    drop_path_rate: float = 0.3
    head_dropout: float = 0.0
    # MoE switches (paper-L: MoE on the attention projection, top-2, 8 experts,
    # BatchNorm, NO aux loss - the ablation shows aux loss hurts).
    use_moe: bool = True
    use_moe_proj: bool = True            # experts on attention output projection
    use_moe_mlp: bool = False            # experts on the block MLP (alt placement)
    num_experts: int = 8
    moe_topk: int = 2
    moe_use_residual: bool = False       # shared (always-on) expert
    moe_layers: Optional[list] = None    # None = MoE in every block
    moe_aux_loss_alpha: float = 0.0
    moe_n_intermediate_size: float = 2.0      # expert width = 2x channels: matches MEEPO-L (100M total params); Tab.3f trend is mixed
    moe_act_fn: str = "relu"
    moe_domain_guided: bool = False      # label-free routing (no terrain context)
    moe_context_channels: int = 16

    # ------------------------------------------------------------------ #
    # Backbone selector + LitePT variant (A/B against MEEPO-L)
    # ------------------------------------------------------------------ #
    backbone: str = "meepo"              # "meepo" (CNN-Mamba, default) -- PTv3/LitePT + MoE were stripped.
    # MEEPO (CNN-Mamba) backbone -- hyperparameters from the MEEPO ScanNet config
    # (semseg-meepo.py). Each block is SubMConv3d(cpe) -> RMSNorm -> Bidirectional
    # Mamba -> LayerNorm -> MLP; no MoE; LayerNorm/RMSNorm only (micro-batch-1 safe).
    meepo_enc_depths: tuple = (2, 2, 2, 6, 2)
    meepo_enc_channels: tuple = (32, 64, 128, 256, 512)
    meepo_dec_depths: tuple = (2, 2, 2, 2)
    meepo_dec_channels: tuple = (64, 64, 128, 256)
    mamba_state_dim: int = 1             # SSM state dim N (MEEPO config: 1; locality comes from the conv)
    mamba_conv_dim: int = 4              # non-causal depthwise conv width d_conv
    mamba_expand_factor: int = 3         # block expansion (d_inner = expand * d_model)
    mamba_directions: int = 2            # scan directions: 2 = bidirectional (released MEEPO), 4 = + strided
    ssm_backend: str = "auto"            # "auto" = fused mamba_ssm CUDA kernel if importable, else chunked SSD scan (Mamba-2 algorithm, pure torch); "cuda" = require kernel; "ssd" = force SSD scan; "torch" = naive reference loop (validation)
    # LitePT-S geometry (Yue et al. 2025, Tab.12): conv blocks for stages < lc,
    # PointROPE-attention blocks (+ Proj-MoE) for stages >= lc. Channels divisible
    # by 6 in attention stages (head_dim divisible by 6 for the 3-axis rotary split).
    litept_lc: int = 3                   # stages 0,1,2 = sparse conv; 3,4 = PointROPE attention
    litept_enc_depths: tuple = (2, 2, 2, 6, 2)
    litept_enc_channels: tuple = (36, 72, 144, 252, 504)
    litept_enc_num_head: tuple = (2, 4, 8, 14, 28)   # heads only used in attention stages (252/14=504/28=18)
    litept_dec_channels: tuple = (72, 72, 144, 252)  # LitePT-S lightweight (linear) decoder
    litept_rope_freq: float = 100.0      # PointROPE base frequency (paper Tab.6 best)
    litept_decoder_blocks: bool = False  # False = LitePT-S (linear decoder); True = LitePT-S* (mirror enc)

    # ------------------------------------------------------------------ #
    # Previous-year CLASSIFICATION raster branch (Deviation A)
    # ------------------------------------------------------------------ #
    prior_raster_channels: int = 5       # dtm, dsm, ndsm, ground_prob, coverage
    prior_raster_gating: bool = True     # GrounDiff-style confidence gating head
    dtm_cnn_mid: int = 32                # 2D-CNN width

    def resolve_geometry(self, num_layers: int = None) -> "Config":
        """When ``auto_in_radius`` is set, size the input cylinder to the coarsest
        conv reach (``conv_radius * first_subsampling_dl * 2**(num_layers-1)``) and
        keep ``sphere_center_spacing`` equal to it. ``num_layers`` defaults to
        ``self.num_layers`` (matches the architecture). Idempotent; call after
        load()/CLI overrides and BEFORE preprocessing (stage 04) or training.
        Returns self."""
        if num_layers is None:
            num_layers = int(getattr(self, "num_layers", 5))
        # For the KPConvX network, the number of resolution levels is fixed by the
        # blocks-per-level list; keep num_layers (which the batch pyramid uses) in sync.
        if bool(getattr(self, "use_kpconvx", False)):
            lb = list(getattr(self, "layer_blocks", []))
            if lb:
                num_layers = len(lb)
                self.num_layers = num_layers
        if getattr(self, "auto_in_radius", False):
            gs = float(getattr(self, "grid_scaling", 2.0))
            r = float(self.conv_radius) * float(self.first_subsampling_dl) * (gs ** (num_layers - 1))
            self.in_radius = float(r)
            self.sphere_center_spacing = float(r)
        return self

    def save(self, path: str) -> None:
        if yaml is None:
            raise RuntimeError("pyyaml not installed")
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        if yaml is None:
            raise RuntimeError("pyyaml not installed")
        with open(path) as f:
            data = yaml.safe_load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_overrides(cls, base: Optional["Config"] = None, **overrides) -> "Config":
        cfg = base or cls()
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
