"""
Preprocessing: turn raw New Zealand LAS/LAZ clouds into KPConv-style samples.

Faithful to KPConv's KP-FCNN input pipeline: each cloud is grid-subsampled once
to ``first_subsampling_dl`` (one barycentre per voxel), then the network draws
**input spheres** of radius ``in_radius`` at train/test time (no tiling). We
store, per cloud: the subsampled points, binary labels, laser-return channels,
the previous-year DTM raster (the sanctioned deviation #1, for sphere-centred
patches), and a set of candidate sphere centres - each classified into the seven
candidate sphere centres (with their cylinder point indices) for uniform,
regionally-diverse sphere sampling at train time.

Shallow features are NOT baked here: they are computed per sphere on the
KPConv-centred coordinates at load time (so the higher-order moments are raw
monomials of centred coords, per the paper). ``compute_norm_stats`` samples a
few spheres to estimate per-channel mean/std.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

from ..utils.laz_io import read_points, label_from_classification
from ..features.shallow_features import expected_feature_dim
from .dtm import (Raster, crop_dtm_patch, crop_downsample_raster,
                  MultiRaster, crop_downsample_multiraster, height_above_prev_dtm)
from .subsampling import grid_subsample


def preprocess_points(
    xyz: np.ndarray,
    classification: np.ndarray,
    num_returns: np.ndarray,
    return_number: np.ndarray,
    intensity: np.ndarray,
    cfg,
    out_path: str,
    prev_dtm: Optional[Raster] = None,
    prev_prior: "Optional[MultiRaster]" = None,
    split: str = "train",
) -> int:
    """Grid-subsample one cloud, build candidate sphere centres, and save one
    ``.npz`` per cloud. Returns 1 on success, 0 if the cloud is too small."""
    if cKDTree is None:
        raise RuntimeError("scipy is required for preprocessing")
    n_raw = xyz.shape[0]
    if n_raw < int(cfg.sphere_min_points):
        return 0
    labels_all = label_from_classification(
        classification,
        ground_classes=getattr(cfg, "ground_classes", None),
        unclassified_classes=getattr(cfg, "unclassified_classes", None))

    # ---- whole-cloud grid subsample (KPConv): barycentre + majority label ----
    # carry the per-point attributes (num_returns, return_number, intensity)
    # through subsampling together so the voxel barycentre averages them too.
    # per-raw-point return ratio = return_number / number_of_returns in (0,1],
    # carried through subsampling as a 4th channel so the stored value is the true
    # mean of per-point ratios (not mean(rn)/mean(nr)). ~1 => last/only return.
    nr_raw = np.maximum(num_returns.astype(np.float32), 1.0)
    ratio_pp = (return_number.astype(np.float32) / nr_raw).astype(np.float32)
    feat_pp = np.stack([num_returns.astype(np.float32),
                        return_number.astype(np.float32),
                        intensity.astype(np.float32),
                        ratio_pp], axis=1)
    if getattr(cfg, "first_subsampling_dl", 0) and cfg.first_subsampling_dl > 0:
        sub_xyz, sub_feat, sub_lab = grid_subsample(
            xyz.astype(np.float32), feat_pp, labels_all, float(cfg.first_subsampling_dl))
    else:
        sub_xyz, sub_feat, sub_lab = xyz.astype(np.float32), feat_pp, labels_all.astype(np.int64)
    sub_ret = sub_feat[:, :2]                 # [num_returns, return_number]
    sub_int = sub_feat[:, 2]                  # intensity
    sub_ratio = sub_feat[:, 3]                # mean per-point return ratio
    n = sub_xyz.shape[0]
    if n < int(cfg.sphere_min_points):
        return 0

    # subsampled ASPRS class code (nearest full-res point) - only for scene class
    _, nn = cKDTree(xyz[:, :2]).query(sub_xyz[:, :2], k=1)
    sub_cls = classification[nn]

    file_origin = np.array([sub_xyz[:, 0].min(), sub_xyz[:, 1].min(),
                            sub_xyz[:, 2].min()], dtype=np.float64)
    local = (sub_xyz - file_origin).astype(np.float32)

    # Per-point z - prevDTM (use_prev_dtm): sampled from the matched prior (channel 0
    # of the 5-ch prior, or the legacy single-channel DTM) at world coords. Independent
    # of the sphere-only raster branch, so it survives into scene mode. Saved per tile.
    dtm_height = None
    if bool(getattr(cfg, "use_prev_dtm", False)) and (prev_prior is not None or prev_dtm is not None):
        if prev_prior is not None:
            _dtm_ras = Raster(np.asarray(prev_prior.data[0], dtype=np.float32),
                              float(prev_prior.x_min), float(prev_prior.y_min), float(prev_prior.res))
        else:
            _dtm_ras = prev_dtm
        dtm_height = height_above_prev_dtm(sub_xyz, _dtm_ras).astype(np.float32)
        # Partial-coverage honesty: where the prior does NOT actually cover a point (a
        # NoData hole, or outside the raster's extent) sample_dtm would extrapolate an
        # edge value. Zero those points' deviation feature instead, so an incomplete
        # hand-crafted raster contributes no phantom signal. Coverage still reaches the
        # raster branch through the 5-channel prior's coverage channel.
        if prev_prior is not None and bool(getattr(cfg, "mask_uncovered_prev_dtm", True)):
            from .dtm import prior_coverage_mask
            covered = prior_coverage_mask(prev_prior, sub_xyz[:, :2])
            dtm_height = np.where(covered, dtm_height, np.float32(0.0)).astype(np.float32)
        dtm_height = np.nan_to_num(dtm_height, nan=0.0).astype(np.float32)

    # ---- candidate sphere centres: snap a regular grid to nearest sub point ----
    tree2d = cKDTree(sub_xyz[:, :2])
    R = float(cfg.in_radius)
    s = float(cfg.sphere_center_spacing)
    xs = np.arange(sub_xyz[:, 0].min(), sub_xyz[:, 0].max() + 1e-6, s)
    ys = np.arange(sub_xyz[:, 1].min(), sub_xyz[:, 1].max() + 1e-6, s)
    # ``cyl`` keeps, per accepted candidate, the indices of the sub-points inside
    # its in_radius cylinder. These are computed here anyway for the min-points
    # check; storing them lets __getitem__ slice the cylinder at
    # train time instead of rebuilding a KD-tree over the whole tile per access
    # (and makes candidate_point_counts a free offset lookup).
    centers, cyl, seen = [], [], set()
    for cx in xs:
        for cy in ys:
            ci = int(tree2d.query([cx, cy], k=1)[1])
            if ci in seen:
                continue
            cpt = sub_xyz[ci]
            sph = np.asarray(tree2d.query_ball_point(cpt[:2], R), dtype=np.int64)  # cylinder (XY disc)
            if sph.size < int(cfg.sphere_min_points):
                continue
            seen.add(ci)
            centers.append((cpt - file_origin).astype(np.float32))
            cyl.append(np.sort(sph).astype(np.int32))
    if not centers:                                   # fallback: one centre at centroid
        ci = int(tree2d.query([sub_xyz[:, 0].mean(), sub_xyz[:, 1].mean()], k=1)[1])
        sph = np.asarray(tree2d.query_ball_point(sub_xyz[ci][:2], R), dtype=np.int64)
        if sph.size < int(cfg.sphere_min_points):     # tiny tile: whole cloud is the cylinder
            sph = np.arange(n, dtype=np.int64)
        centers = [(sub_xyz[ci] - file_origin).astype(np.float32)]
        cyl = [np.sort(sph).astype(np.int32)]
    centers = np.stack(centers).astype(np.float32)
    # CSR layout: cand_idx is every cylinder's indices concatenated; cand_off[k]:
    # cand_off[k+1] is candidate k's slice. int32 keeps it compact.
    cand_off = np.zeros(len(cyl) + 1, dtype=np.int64)
    np.cumsum(np.array([a.size for a in cyl], dtype=np.int64), out=cand_off[1:])
    cand_idx = (np.concatenate(cyl).astype(np.int32) if cyl
                else np.zeros(0, dtype=np.int32))

    # ---- previous-year CLASSIFICATION raster (Deviation A), stored per tile ----
    # Crop the whole-region/per-cloud prior raster to THIS tile's extent (+in_radius
    # margin so edge-sphere patches are covered) and block-mean downsample to
    # dtm_store_res (default 1 m); the network samples only a small patch per sphere.
    prior_data = None
    prior_geo = None
    use_rast = bool(getattr(cfg, "use_dtm_raster", False))
    if use_rast and (prev_prior is not None or prev_dtm is not None):
        R = float(cfg.in_radius)
        x0w, y0w = float(file_origin[0]), float(file_origin[1])
        x1w = x0w + float(local[:, 0].max())
        y1w = y0w + float(local[:, 1].max())
        store_res = float(getattr(cfg, "dtm_resolution", 1.0))
        if prev_prior is not None:                       # preferred: 5-channel prior
            pr = crop_downsample_multiraster(prev_prior, x0w - R, y0w - R, x1w + R, y1w + R, store_res)
            prior_data = np.asarray(pr.data, dtype=np.float32)               # (C,H,W)
            prior_geo = np.array([pr.x_min, pr.y_min, pr.res], dtype=np.float64)
            dtm_data = prior_data[0].astype(np.float32)                       # DTM channel
            dtm_geo = prior_geo.copy()                                        # legacy scalar path
        else:                                            # legacy single-channel DTM only
            dtm_t = crop_downsample_raster(prev_dtm, x0w - R, y0w - R, x1w + R, y1w + R, store_res)
            dtm_data = np.asarray(dtm_t.data, dtype=np.float32)
            dtm_geo = np.array([dtm_t.x_min, dtm_t.y_min, dtm_t.res], dtype=np.float64)
    else:
        dtm_data = np.zeros((1, 1), dtype=np.float32)
        dtm_geo = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # ---- optional coverage guarantee: drop tiles whose sphere centres mostly fall
    #      outside the previous-year DTM (cfg.min_dtm_coverage > 0). With spatially
    #      matched downloads this is ~1.0 for every tile; it's a safety net so the
    #      kept set is guaranteed >= the threshold covered. 0 = keep all (default).
    min_cov = float(getattr(cfg, "min_dtm_coverage", 0.0) or 0.0)
    if min_cov > 0.0 and getattr(cfg, "use_dtm_raster", False) and prev_dtm is not None:
        cw = centers[:, :2] + file_origin[:2].astype(np.float32)      # world xy of centres
        Hd, Wd = dtm_data.shape
        col = np.floor((cw[:, 0] - dtm_geo[0]) / dtm_geo[2]).astype(np.int64)
        row = np.floor((cw[:, 1] - dtm_geo[1]) / dtm_geo[2]).astype(np.int64)
        inb = (col >= 0) & (col < Wd) & (row >= 0) & (row < Hd)
        valid = np.zeros(cw.shape[0], dtype=bool)
        if inb.any():
            valid[inb] = ~np.isnan(dtm_data[row[inb], col[inb]])
        coverage = float(valid.mean()) if cw.shape[0] else 0.0
        if coverage < min_cov:
            return 0                                                  # skip: insufficient prev-DTM coverage

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Precompute the expensive KD-tree shallow features ONCE per tile (on the clean
    # tile-local coords) so __getitem__ never rebuilds them per sphere per epoch.
    # Stored only when their config flag is on; legacy tiles without these keys fall
    # back to per-sphere computation in assemble_features.
    save = dict(
        local=local, labels=sub_lab.astype(np.int64), returns=sub_ret.astype(np.float32),
        intensity=sub_int.astype(np.float32), ret_ratio=sub_ratio.astype(np.float32),
        dtm_data=dtm_data, dtm_geo=dtm_geo, file_origin=file_origin,
        centers=centers,
        cand_idx=cand_idx, cand_off=cand_off,
        split=np.array(split),
    )
    if prior_data is not None:
        save["prior_data"] = prior_data            # (C,H,W) previous-year class raster
        save["prior_geo"] = prior_geo
    if dtm_height is not None:
        save["dtm_height"] = dtm_height            # (N,) z - prevDTM, the use_prev_dtm channel
    try:
        from ..features.shallow_features import compute_shallow_geom
        feat_elev, feat_curv = compute_shallow_geom(local, cfg)
        if feat_elev is not None:
            save["feat_mean_elev"] = feat_elev.astype(np.float32)
        if feat_curv is not None:
            save["feat_curvature"] = feat_curv.astype(np.float32)
    except Exception:
        pass                                       # fall back to per-sphere features
    from .tile_io import save_tile
    save_tile(out_path, save)                      # big arrays -> mmap-able .npy sidecars
    return 1


def preprocess_file(
    las_path: str,
    cfg,
    out_dir: str,
    prev_dtm: Optional[Raster] = None,
    prev_prior: "Optional[MultiRaster]" = None,
    split: str = "train",
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Read one LAS/LAZ cloud and write one subsampled, sphere-indexed ``.npz``."""
    os.makedirs(out_dir, exist_ok=True)
    xyz, classification, num_returns, return_number, intensity, rgb, meta = read_points(
        las_path, want_rgb=cfg.use_rgb)
    base = os.path.splitext(os.path.basename(las_path))[0]
    return preprocess_points(xyz, classification, num_returns, return_number, intensity, cfg,
                             os.path.join(out_dir, f"{base}.npz"),
                             prev_dtm=prev_dtm, prev_prior=prev_prior, split=split)


def compute_norm_stats(tile_dir: str, cfg, n_tiles: int = 16, per_tile: int = 20) -> dict:
    """Per-channel feature mean/std over a sample of input spheres.

    Spheres are drawn **tile-grouped**: ``per_tile`` candidates from each of
    ``n_tiles`` randomly chosen tiles, visited tile-by-tile. A uniform random sample
    would touch ~one distinct tile per sphere (hundreds of huge tiles) and blow RAM
    via the loader cache; grouping loads only ``n_tiles`` tiles, each reused, so the
    pass is fast and bounded."""
    from collections import defaultdict
    from .dataset import SphereDataset
    from ..utils.progress import progress
    ds = SphereDataset(tile_dir, cfg, split=None, augment=False)
    ds.mean = None
    ds.std = None                                      # force RAW features
    if len(ds) == 0:
        return {"mean": [], "std": [], "n_points": 0, "n_features": expected_feature_dim(cfg)}

    by_file = defaultdict(list)
    for gi, cand in enumerate(ds.cands):
        by_file[cand[0]].append(gi)                    # cand[0] = file index
    rng = np.random.default_rng(cfg.seed)
    files = list(by_file.keys())
    rng.shuffle(files)
    files = files[:min(int(n_tiles), len(files))]
    sel = []
    for fi in files:                                   # keep grouped so the LRU cache hits
        cand = by_file[fi]
        pick = rng.choice(len(cand), size=min(int(per_tile), len(cand)), replace=False)
        sel.extend(int(cand[p]) for p in pick)

    n_feat = expected_feature_dim(cfg)
    s1 = np.zeros(n_feat); s2 = np.zeros(n_feat); count = 0
    for i in progress(sel, desc="[04] norm-stats"):
        feats = ds[int(i)]["features"].astype(np.float64)
        if feats.shape[1] != n_feat:
            n_feat = feats.shape[1]; s1 = np.zeros(n_feat); s2 = np.zeros(n_feat); count = 0
        s1 += feats.sum(0); s2 += (feats ** 2).sum(0); count += feats.shape[0]
    mean = s1 / max(count, 1)
    std = np.sqrt(np.maximum(s2 / max(count, 1) - mean ** 2, 1e-8))
    stats = {"mean": mean.tolist(), "std": std.tolist(),
             "n_points": int(count), "n_features": int(n_feat)}
    with open(os.path.join(tile_dir, "norm_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    return stats
