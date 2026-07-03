"""PTv3 sphere-voting inference (clean MEEPO).

Same overlapping-sphere voting strategy as the original KPConv pipeline, but the
per-sphere batch is built the PTv3 way (voxelise -> coord/grid_coord/feat/offset)
instead of a KPConv kd-tree neighbour pyramid, and the model returns one
prediction per *voxel*, which we map back to the sphere's sub-points via the
voxelisation inverse before accumulating soft-max probabilities. Signature is
unchanged so the reused Trainer gallery / large-scene driver keep working.

``prev_dtm`` may be a multi-channel :class:`MultiRaster` (previous-year
classification raster, Deviation A) or a legacy single-channel ``Raster``/None;
the per-sphere patch is cropped to (C, ps, ps) and sampled inside the model.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

from ..data.subsampling import grid_subsample
from ..data.batch import move_batch
from ..data.dtm import crop_multiraster_patch, MultiRaster, PRIOR_RASTER_CHANNELS, Raster, height_above_prev_dtm
from ..features.shallow_features import assemble_features, compute_shallow_geom
from .spag_dc import spag_dc_refine             # SPAG-DC ground-misclassification corrector (IEEE Sensors 2025)


def _accumulate_regime(model, acc):
    """LEARNED SPAG-DC: stash this batch's per-scene regime globals (mean over the
    discs in the batch) so they can be averaged into the scene-level globals."""
    rp = getattr(model, "_regime_pred", None)
    if rp is not None:
        acc.append(rp.detach().float().mean(0).cpu().numpy())


def _scene_regime(acc):
    """Mean of the accumulated per-batch regime globals, or None if the head is off."""
    return np.mean(acc, axis=0) if acc else None


# ---------------------------------------------------------------------------
# Test-time augmentation helpers: rotate the cloud AND the georeferenced prior
# raster together by k*90 deg about a shared centre, so the model sees a rotated
# (cloud, prior) pair exactly as it did under augment_tile at train time. The
# transform below is verified exact: sample(rot_raster, rot_xy) == sample(raster, xy)
# (world +90 CCW  <=>  data'=rot90(data,-1), x_min'=cx+cy-y_max, y_min'=x_min+cy-cx).
# Per-point features (intensity, returns, height-above-prev-DTM) are rotation-
# invariant scalars, so they need no change.
# ---------------------------------------------------------------------------
def _rot_xy_k90(xyz: np.ndarray, k: int, c: np.ndarray) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float64).copy()
    d = out[:, :2] - c
    for _ in range(int(k) % 4):
        d = np.column_stack([-d[:, 1], d[:, 0]])      # +90 deg CCW about c
    out[:, :2] = d + c
    return out.astype(np.float32)


def _rot_raster_k90(R: "Raster", k: int, c: np.ndarray) -> "Raster":
    cx, cy = float(c[0]), float(c[1])
    out = R
    for _ in range(int(k) % 4):
        H, W = out.data.shape
        y_max = out.y_min + H * out.res
        out = Raster(np.rot90(out.data, -1).astype(np.float32),
                     cx + cy - y_max, out.x_min + cy - cx, out.res)
    return out


def _rot_multiraster_k90(mr, k: int, c: np.ndarray):
    if mr is None:
        return None
    cx, cy = float(c[0]), float(c[1])
    out = mr
    for _ in range(int(k) % 4):
        Cc, H, W = out.data.shape
        y_max = out.y_min + H * out.res
        new = np.stack([np.rot90(out.data[ch], -1) for ch in range(Cc)]).astype(np.float32)
        out = MultiRaster(new, cx + cy - y_max, out.x_min + cy - cx, out.res,
                          getattr(out, "channels", PRIOR_RASTER_CHANNELS))
    return out


def _grid_center_indices(sub_xyz: np.ndarray, spacing: float):
    tree2d = cKDTree(sub_xyz[:, :2])
    xs = np.arange(sub_xyz[:, 0].min(), sub_xyz[:, 0].max() + 1e-6, spacing)
    ys = np.arange(sub_xyz[:, 1].min(), sub_xyz[:, 1].max() + 1e-6, spacing)
    seen, centers = set(), []
    for cx in xs:
        for cy in ys:
            ci = int(tree2d.query([cx, cy], k=1)[1])
            if ci not in seen:
                seen.add(ci); centers.append(ci)
    return centers


def _as_multiraster(prev_dtm):
    """Coerce ``prev_dtm`` to a MultiRaster (or None). A single-channel Raster is
    promoted to the 5-channel layout with a ground-everywhere prior."""
    if prev_dtm is None or isinstance(prev_dtm, MultiRaster):
        return prev_dtm
    d = np.asarray(prev_dtm.data, dtype=np.float32)
    data = np.stack([d, d, np.zeros_like(d), np.ones_like(d), np.ones_like(d)], 0)
    return MultiRaster(data, float(prev_dtm.x_min), float(prev_dtm.y_min),
                       float(prev_dtm.res), PRIOR_RASTER_CHANNELS)


def _voxel_pack(centered, feats, grid_size):
    """Voxelise one centred sphere into a PTv3 single-cloud batch (numpy)."""
    gmin = centered.min(0)
    gc = np.floor((centered - gmin) / grid_size).astype(np.int64)
    key = gc - gc.min(0)
    span = key.max(0) + 1
    flat = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    uniq, inverse = np.unique(flat, return_inverse=True)
    n_vox = len(uniq)
    vcoord = np.zeros((n_vox, 3), np.float64); cnt = np.zeros((n_vox, 1))
    np.add.at(vcoord, inverse, centered.astype(np.float64)); np.add.at(cnt, inverse, 1.0)
    vcoord = (vcoord / np.maximum(cnt, 1.0)).astype(np.float32)
    vfeat = np.zeros((n_vox, feats.shape[1]), np.float64)
    np.add.at(vfeat, inverse, feats.astype(np.float64))
    vfeat = (vfeat / np.maximum(cnt, 1.0)).astype(np.float32)
    vgrid = np.zeros((n_vox, 3), np.int64); vgrid[inverse] = gc
    return vcoord, vgrid, vfeat, inverse, n_vox


def predict_cloud_spheres(xyz, num_returns, return_number, cfg, model, device,
                          mean=None, std=None, prev_dtm=None, neighbor_limit: int = 50,
                          return_proba: bool = False, progress: int = 0, intensity=None,
                          ret_ratio=None, return_precleanup: bool = False, tta: bool = False):
    """Sphere-vote a whole cloud. Returns a full-resolution (N,) int prediction
    (1=ground, 0=non-ground), or ``(pred, proba_full)`` if ``return_proba``.

    When ``cfg.scene_mode`` is set this dispatches to :func:`predict_scene`
    (PTv3-native large-block inference) so the per-epoch visualiser and the
    inference scripts run full-scene without sphere voting."""
    if bool(getattr(cfg, "scene_mode", False)):
        return predict_scene(xyz, num_returns, return_number, cfg, model, device,
                             mean=mean, std=std, return_proba=return_proba,
                             progress=progress, intensity=intensity, ret_ratio=ret_ratio,
                             return_precleanup=return_precleanup, prev_dtm=prev_dtm, tta=tta)
    if cKDTree is None:
        raise RuntimeError("scipy is required for inference")
    R = float(cfg.in_radius)
    dl = float(getattr(cfg, "first_subsampling_dl", 0.0) or 0.0)
    grid_size = dl if dl > 0 else 0.1
    if intensity is None:
        intensity = np.zeros((xyz.shape[0],), dtype=np.float32)
    if ret_ratio is None:
        ret_ratio = (return_number.astype(np.float32) /
                     np.maximum(num_returns.astype(np.float32), 1.0))
    feat_pp = np.stack([num_returns.astype(np.float32), return_number.astype(np.float32),
                        intensity.astype(np.float32), ret_ratio.astype(np.float32)], 1)
    if dl > 0:
        sub_xyz, sub_feat, _ = grid_subsample(xyz.astype(np.float32), feat_pp, None, dl)
    else:
        sub_xyz, sub_feat = xyz.astype(np.float32), feat_pp
    sub_ret = sub_feat[:, :2]; sub_int = sub_feat[:, 2]; sub_ratio = sub_feat[:, 3]
    n_sub = sub_xyz.shape[0]

    tree2d = cKDTree(sub_xyz[:, :2])
    centers = _grid_center_indices(sub_xyz, float(cfg.sphere_center_spacing))
    min_pts = int(cfg.sphere_min_points)
    ps = int(getattr(cfg, "dtm_patch_size", 64))
    nchan = int(getattr(cfg, "prior_raster_channels", 5))
    prior = _as_multiraster(prev_dtm)
    use_raster = bool(getattr(cfg, "use_dtm_raster", False)) and prior is not None
    _dtm_ras = (Raster(np.asarray(prior.data[0], np.float32), float(prior.x_min), float(prior.y_min),
                       float(prior.res)) if (bool(getattr(cfg, "use_prev_dtm", False)) and prior is not None) else None)
    mean_a = None if mean is None else np.asarray(mean, dtype=np.float32)

    proba = np.zeros((n_sub, 2), dtype=np.float64)
    covered = np.zeros(n_sub, dtype=np.int32)
    model.eval()
    infer_b = max(int(getattr(cfg, "infer_batch_spheres", 16)), 1)

    def _build_pack(ci):
        """Voxelise one sphere -> (idx, inverse, vcoord, vgrid, vfeat, dtm_patch, n_vox)."""
        center = sub_xyz[ci].astype(np.float64)
        idx = tree2d.query_ball_point(center[:2], R)
        if len(idx) < min_pts:
            idx = np.atleast_1d(tree2d.query(center[:2], k=min(min_pts, n_sub))[1])
        idx = np.asarray(idx, dtype=np.int64)
        centered = (sub_xyz[idx] - center).astype(np.float32)
        if use_raster:
            patch = crop_multiraster_patch(prior, center[0] - R, center[1] - R,
                                           2.0 * R, ps, origin_z=float(center[2]))
        else:
            patch = np.zeros((nchan, ps, ps), dtype=np.float32)
        dh = None if _dtm_ras is None else height_above_prev_dtm(sub_xyz[idx], _dtm_ras)
        feats = assemble_features(centered, cfg, num_returns=sub_ret[idx, 0],
                                  return_number=sub_ret[idx, 1], intensity=sub_int[idx],
                                  return_ratio=sub_ratio[idx], dtm_height=dh).astype(np.float32)
        if mean_a is not None and feats.shape[1] == mean_a.shape[0]:
            feats = (feats - mean_a) / std
        vcoord, vgrid, vfeat, inverse, n_vox = _voxel_pack(centered, feats, grid_size)
        return idx, inverse, vcoord, vgrid, vfeat, patch, n_vox

    # Process spheres in batches: each sphere is an independent cloud, so a chunk of
    # them is exactly a PTv3 multi-cloud batch (concatenated points + cumulative offset
    # + stacked rasters). One forward per chunk instead of per sphere amortises the
    # Python/voxel overhead and the kernel launches -> large inference speedup.
    done = 0
    regime_acc = []                                   # learned SPAG-DC regime globals (per batch)
    for start in range(0, len(centers), infer_b):
        packs = [_build_pack(ci) for ci in centers[start:start + infer_b]]
        if not packs:
            continue
        nvox = [int(p[6]) for p in packs]
        offs = np.cumsum(nvox).astype(np.int64)
        batch = {
            "coord": torch.from_numpy(np.concatenate([p[2] for p in packs], 0)),
            "grid_coord": torch.from_numpy(np.concatenate([p[3] for p in packs], 0)),
            "feat": torch.from_numpy(np.concatenate([p[4] for p in packs], 0)),
            "offset": torch.from_numpy(offs),
            "cloud_lengths_0": torch.tensor(nvox, dtype=torch.long),
        }
        if use_raster:
            batch["dtm_patches"] = torch.from_numpy(
                np.stack([p[5] for p in packs], 0).astype(np.float32))
        batch = move_batch(batch, device)
        with torch.no_grad():
            logits = model(batch).float()
            p_all = torch.softmax(logits, dim=1).cpu().numpy()       # (sum_nvox, 2)
            _accumulate_regime(model, regime_acc)
        s = 0
        for j, p in enumerate(packs):
            e = int(offs[j])
            idx, inverse = p[0], p[1]
            proba[idx] += p_all[s:e][inverse]                         # voxel -> sub-point
            covered[idx] += 1
            s = e
        done += len(packs)
        if progress and (start // max(infer_b, 1)) % max(progress, 1) == 0:
            print(f"    spheres {done}/{len(centers)}", flush=True)

    sub_raw = np.zeros(n_sub, dtype=np.int64)
    has = covered > 0
    sub_raw[has] = proba[has].argmax(1)
    # SPAG-DC misclassification correction. With cfg.spag_learned, the regime head's
    # per-scene predicted globals (mean over discs) replace the fixed cfg defaults;
    # otherwise geometry-only (theta0/alpha/beta/n_sigma/base_res from cfg + density).
    sub_clean = spag_dc_refine(sub_xyz, sub_raw, cfg, learned_globals=_scene_regime(regime_acc))
    _, nn = cKDTree(sub_xyz).query(xyz, k=1)
    pred_full = sub_clean[nn]
    raw_full = sub_raw[nn]
    if return_proba and return_precleanup:
        denom = np.maximum(covered[nn, None], 1)
        return pred_full, (proba[nn] / denom), raw_full
    if return_proba:
        denom = np.maximum(covered[nn, None], 1)
        return pred_full, (proba[nn] / denom)
    if return_precleanup:
        return pred_full, raw_full
    return pred_full




def predict_scene(xyz, num_returns, return_number, cfg, model, device,
                  mean=None, std=None, return_proba: bool = False, progress: int = 0,
                  intensity=None, ret_ratio=None, return_precleanup: bool = False, prev_dtm=None,
                  tta: bool = False):
    """PTv3-native full-scene inference (no spheres).

    The cloud is grid-subsampled, then partitioned into a regular grid of large
    ``scene_block_size``-metre blocks that together cover every point exactly
    once. Each block is predicted with a ``scene_block_margin`` ring of context
    points (predicted but discarded), so block interiors see surrounding terrain
    - the cheap analogue of SparseGF's central-region scheme. One model forward
    per block. If the model carries the previous-year prior-raster branch (a
    sphere-trained model), per-voxel raster features are built the training way
    (local 2R windows) and concatenated, so whole-scene inference works WITH the
    raster branch. Returns the full-resolution (N,) labels, or ``(pred, proba_full)``
    if ``return_proba``.
    """
    if cKDTree is None:
        raise RuntimeError("scipy is required for inference")

    # ---- test-time augmentation: average RAW softmax over z-rotations, then
    # argmax, then SPAG-DC once (Pointcept SemSegTester accumulates softmax over
    # the aug_transform views before argmax). Rotation preserves point order, so
    # per-point probabilities accumulate directly. Cloud AND prior raster are
    # rotated together (verified sample-preserving). Per-epoch val passes tta=False.
    if tta:
        rots = [0, 1, 2, 3]
        if len(rots) > 1:
            c = np.asarray(xyz[:, :2], dtype=np.float64).mean(0)
            prior_mr = _as_multiraster(prev_dtm)
            proba_sum = None
            tta_regimes = []                          # learned SPAG-DC: per-rotation scene globals
            for k in rots:
                xyz_k = _rot_xy_k90(xyz, k, c)
                prior_k = _rot_multiraster_k90(prior_mr, k, c) if prior_mr is not None else None
                _, proba_k = predict_scene(
                    xyz_k, num_returns, return_number, cfg, model, device,
                    mean=mean, std=std, return_proba=True, progress=0,
                    intensity=intensity, ret_ratio=ret_ratio, prev_dtm=prior_k, tta=False)
                proba_k = np.asarray(proba_k, dtype=np.float64)
                proba_sum = proba_k if proba_sum is None else proba_sum + proba_k
                _rg = getattr(model, "_scene_regime", None)
                if _rg is not None:
                    tta_regimes.append(np.asarray(_rg, dtype=np.float64))
            proba_avg = proba_sum / float(len(rots))
            raw = proba_avg.argmax(1).astype(np.int64)
            clean = spag_dc_refine(np.asarray(xyz, dtype=np.float64), raw, cfg,
                                   learned_globals=_scene_regime(tta_regimes))
            if return_precleanup and return_proba:
                return clean, proba_avg, raw
            if return_proba:
                return clean, proba_avg
            if return_precleanup:
                return clean, raw
            return clean

    dl = float(getattr(cfg, "first_subsampling_dl", 0.0) or 0.0)
    grid_size = dl if dl > 0 else 0.1            # keep dl=0.1 everywhere (voxelisation default)
    block_cfg = float(getattr(cfg, "scene_block_size", 64.0))   # max receptive field; shrunk to fit max_pts below
    max_pts = int(getattr(cfg, "scene_max_points", 1_500_000))
    if intensity is None:
        intensity = np.zeros((xyz.shape[0],), dtype=np.float32)
    if ret_ratio is None:
        ret_ratio = (return_number.astype(np.float32) /
                     np.maximum(num_returns.astype(np.float32), 1.0))
    feat_pp = np.stack([num_returns.astype(np.float32), return_number.astype(np.float32),
                        intensity.astype(np.float32), ret_ratio.astype(np.float32)], 1)
    if dl > 0:
        sub_xyz, sub_feat, _ = grid_subsample(xyz.astype(np.float32), feat_pp, None, dl)
    else:
        sub_xyz, sub_feat = xyz.astype(np.float32), feat_pp
    sub_ret = sub_feat[:, :2]; sub_int = sub_feat[:, 2]; sub_ratio = sub_feat[:, 3]
    n_sub = sub_xyz.shape[0]
    mean_a = None if mean is None else np.asarray(mean, dtype=np.float32)
    # previous-year DTM as a per-point feature (use_prev_dtm), matching training. This is
    # separate from the raster CNN branch (which now also runs in whole-scene mode below).
    _dtm_ras = None
    if bool(getattr(cfg, "use_prev_dtm", False)) and prev_dtm is not None:
        _dtm_ras = (Raster(np.asarray(prev_dtm.data[0], np.float32), float(prev_dtm.x_min),
                           float(prev_dtm.y_min), float(prev_dtm.res))
                    if isinstance(prev_dtm, MultiRaster) else prev_dtm)

    proba = np.zeros((n_sub, 2), dtype=np.float64)
    covered = np.zeros(n_sub, dtype=np.int32)
    model.eval()
    # Shallow features (mean elevation / curvature): training precomputes these ONCE per tile
    # at native density and re-centres elevation per region. Match that by precomputing on the
    # FULL sub-cloud and indexing per block, rather than recomputing per (count-capped) block --
    # otherwise the model sees a different neighbourhood than it trained on (a train/val gap).
    # Returns (None, None) unless use_mean_elevation / use_curvature is enabled.
    mep_full, mcp_full = compute_shallow_geom(sub_xyz.astype(np.float64), cfg)

    xy = sub_xyz[:, :2]
    lo = xy.min(0); hi = xy.max(0)
    tree2d = cKDTree(xy)
    rng = np.random.default_rng(0)
    # ---- SparseGF overlapping-disc SOFT VOTING (Sec 2.3) --------------------
    # Lay disc centres on a regular grid of step `scene_vote_step_m`. Each disc
    # CLASSIFIES its circular CENTRAL region of radius Rc = (sqrt2/2)*step, which
    # circumscribes the step x step cell -> every point is central in >=1 disc
    # (complete coverage) and central regions overlap at cell edges/corners. The
    # model sees the nearest ~max_pts points as CONTEXT -- the SAME disc unit it
    # trains on (Pointcept SphereCrop point_max). Predictions for points falling in
    # multiple central regions are soft-voted (proba/covered below). If the whole
    # sub-cloud already fits one disc (val's bounded region / small clouds), one
    # forward over all points is the degenerate single-vote case.
    if n_sub <= max_pts:
        jobs = [(np.arange(n_sub), np.arange(n_sub))]
        if progress:
            print(f"    predict_scene: {n_sub} pts -> single disc (<= max_pts), 1 forward", flush=True)
    else:
        step = float(getattr(cfg, "scene_vote_step_m", 50.0))
        if block_cfg > 0:
            step = float(min(step, block_cfg))
        Rc = (np.sqrt(2.0) / 2.0) * step
        _area = float(max((hi[0] - lo[0]) * (hi[1] - lo[1]), 1.0))
        _dens = n_sub / _area
        r_ctx = max(Rc + float(getattr(cfg, "scene_block_margin", 8.0)),
                    float(np.sqrt(max_pts / max(_dens * np.pi, 1e-6))))
        nx = max(int(np.ceil((hi[0] - lo[0]) / step)), 1)
        ny = max(int(np.ceil((hi[1] - lo[1]) / step)), 1)
        if progress:
            print(f"    predict_scene: {n_sub} pts  dens={_dens:.1f}/m^2  step={step:.0f}m  "
                  f"Rc={Rc:.0f}m  ctx_r={r_ctx:.0f}m  {nx * ny} discs (soft-vote)", flush=True)
        jobs = []
        for a in range(nx):
            for b in range(ny):
                cx = lo[0] + (a + 0.5) * step; cy = lo[1] + (b + 0.5) * step
                central = np.asarray(tree2d.query_ball_point([cx, cy], Rc), dtype=np.int64)
                if central.size == 0:
                    continue
                if central.size > max_pts:                   # central alone exceeds the budget
                    central = rng.choice(central, size=max_pts, replace=False)  # (very dense); neighbours + backstop cover the dropped points
                ctx = np.asarray(tree2d.query_ball_point([cx, cy], r_ctx), dtype=np.int64)
                if ctx.size == 0:
                    ctx = central
                if ctx.size > max_pts:                       # keep all central, trim periphery context
                    cset = np.zeros(n_sub, dtype=bool); cset[central] = True
                    per = ctx[~cset[ctx]]
                    keep = max_pts - int(central.size)
                    if keep > 0 and per.size > keep:
                        per = rng.choice(per, size=keep, replace=False)
                    ctx = np.concatenate([central, per]) if keep > 0 else central
                jobs.append((central, ctx))
    done = 0
    regime_acc = []                                   # learned SPAG-DC regime globals (per batch)
    for central, ctx in jobs:
        done += 1
        center = sub_xyz[ctx].mean(0).astype(np.float64)
        centered = (sub_xyz[ctx] - center).astype(np.float32)
        dh = None if _dtm_ras is None else height_above_prev_dtm(sub_xyz[ctx], _dtm_ras)
        mep = None if mep_full is None else (np.asarray(mep_full)[ctx] - np.float32(center[2])).astype(np.float32)
        mcp = None if mcp_full is None else np.asarray(mcp_full)[ctx].astype(np.float32)
        feats = assemble_features(centered, cfg, num_returns=sub_ret[ctx, 0],
                                  return_number=sub_ret[ctx, 1], intensity=sub_int[ctx],
                                  return_ratio=sub_ratio[ctx], dtm_height=dh,
                                  mean_elev_precomp=mep, curvature_precomp=mcp).astype(np.float32)
        if mean_a is not None and feats.shape[1] == mean_a.shape[0]:
            feats = (feats - mean_a) / std
        vcoord, vgrid, vfeat, inverse, n_vox = _voxel_pack(centered, feats, grid_size)
        batch = {
            "coord": torch.from_numpy(vcoord),
            "grid_coord": torch.from_numpy(vgrid),
            "feat": torch.from_numpy(vfeat),
            "offset": torch.tensor([n_vox], dtype=torch.long),
            "cloud_lengths_0": torch.tensor([n_vox], dtype=torch.long),
        }
        if bool(getattr(model, "use_raster", False)):
            # Deviation A, identical to training: crop the prior to a fixed
            # scene_block_size window at the disc centroid, resample to
            # raster_scene_patch_size, run through the raster CNN in the forward.
            ps = int(getattr(cfg, "raster_scene_patch_size", 128))
            T = float(getattr(cfg, "scene_block_size", 64.0))
            patch = crop_multiraster_patch(prev_dtm, float(center[0]) - T / 2.0,
                                           float(center[1]) - T / 2.0, T, ps,
                                           origin_z=float(center[2]))
            batch["dtm_patches"] = torch.from_numpy(patch[None].astype(np.float32))
        batch = move_batch(batch, device)
        if bool(getattr(model, "use_raster", False)):
            batch["raster_tile_size"] = float(getattr(cfg, "scene_block_size", 64.0))
        with torch.no_grad():
            logits = model(batch).float()
            p_all = torch.softmax(logits, dim=1).cpu().numpy()          # (n_vox, 2)
            _accumulate_regime(model, regime_acc)
        p_pts = p_all[inverse]                                          # voxel -> ctx sub-point
        # soft vote: accumulate softmax over the central points; overlapping central
        # regions of adjacent discs give those points >1 vote (averaged via covered).
        ctx_sorted_idx = np.argsort(ctx, kind="stable")
        central_local = ctx_sorted_idx[np.searchsorted(ctx[ctx_sorted_idx], central)]
        proba[central] += p_pts[central_local]
        covered[central] += 1
        if progress and (done % max(progress, 1) == 0):
            print(f"    discs {done}/{len(jobs)}", flush=True)
    # coverage backstop: any sub-point left unvoted inherits its nearest voted one.
    if (covered == 0).any():
        _miss = np.where(covered == 0)[0]; _have = np.where(covered > 0)[0]
        if _have.size:
            _, _nnj = cKDTree(sub_xyz[_have]).query(sub_xyz[_miss], k=1)
            proba[_miss] = proba[_have[_nnj]]
            covered[_miss] = covered[_have[_nnj]]   # inherit vote count too, so proba/covered stays an average

    sub_raw = np.zeros(n_sub, dtype=np.int64)
    has = covered > 0
    sub_raw[has] = proba[has].argmax(1)
    # SPAG-DC misclassification correction (learned globals when cfg.spag_learned).
    _scene_lg = _scene_regime(regime_acc)
    try:
        model._scene_regime = _scene_lg               # expose to TTA wrapper (regime ~ rotation-invariant)
    except Exception:
        pass
    sub_clean = spag_dc_refine(sub_xyz, sub_raw, cfg, learned_globals=_scene_lg)
    _, nn = cKDTree(sub_xyz).query(xyz, k=1)
    pred_full = sub_clean[nn]
    raw_full = sub_raw[nn]
    if return_proba and return_precleanup:
        denom = np.maximum(covered[nn, None], 1)
        return pred_full, (proba[nn] / denom), raw_full
    if return_proba:
        denom = np.maximum(covered[nn, None], 1)
        return pred_full, (proba[nn] / denom)
    if return_precleanup:
        return pred_full, raw_full
    return pred_full
