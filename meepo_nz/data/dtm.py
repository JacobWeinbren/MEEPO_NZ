"""
Previous-year ground DTM - the single sanctioned deviation from the paper.

The paper feeds the network eight shallow channels (Section 3.1).  Our ONE
deviation adds a ninth: for every point we sample a Digital Terrain Model built
from the *previous year's ground returns* and feed the **height of the point
above that previous-year ground surface** (z - DTM_prev(x, y)).

This module:
  * ``build_dtm_from_ground`` - rasterise previous-year ground points to a 1 m
    DTM (mean z per cell, then fill small gaps);
  * ``sample_dtm`` - bilinear-sample the DTM at arbitrary (x, y);
  * ``height_above_prev_dtm`` - the per-point deviation channel.

If a ready-made LINZ ``dem_1m`` raster is available for the tile it can be used
directly (see ``scripts/02_build_dtm.py``); otherwise we build one from the
previous-year point cloud's ground class.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Raster:
    data: np.ndarray          # (H, W) float32, NaN where unknown
    x_min: float
    y_min: float
    res: float

    @property
    def shape(self):
        return self.data.shape


def build_dtm_from_ground(
    ground_xyz: np.ndarray,
    res: float = 1.0,
    bounds: Optional[tuple] = None,
    fill_iterations: int = 8,
    reduce: str = "mean",
) -> Raster:
    """Rasterise ground points to a DTM at ``res`` metres.

    ``reduce`` selects the per-cell aggregator: ``"mean"`` (default) or ``"min"``.
    The minimum is robust to high outliers (a spike in a cell is ignored in favour of
    the true low ground return), which is what the GrounDiff+ classification refiner
    uses so that high-curvature real terrain is preserved while spikes stand proud.

    ``bounds`` = (x_min, y_min, x_max, y_max); inferred from the points if None.
    Gaps are filled to the FULL raster extent (exact Euclidean nearest-valued fill),
    so the DTM has no NaN holes inside its extent."""
    if bounds is None:
        x_min, y_min = ground_xyz[:, 0].min(), ground_xyz[:, 1].min()
        x_max, y_max = ground_xyz[:, 0].max(), ground_xyz[:, 1].max()
    else:
        x_min, y_min, x_max, y_max = bounds

    W = max(int(np.ceil((x_max - x_min) / res)), 1)
    H = max(int(np.ceil((y_max - y_min) / res)), 1)

    col = np.clip(((ground_xyz[:, 0] - x_min) / res).astype(np.int64), 0, W - 1)
    row = np.clip(((ground_xyz[:, 1] - y_min) / res).astype(np.int64), 0, H - 1)

    if str(reduce).lower() == "min":
        dmin = np.full((H, W), np.inf, dtype=np.float64)
        np.minimum.at(dmin, (row, col), ground_xyz[:, 2].astype(np.float64))
        dtm = np.where(np.isfinite(dmin), dmin, np.nan).astype(np.float32)
        nz = np.isfinite(dmin)
    else:
        acc = np.zeros((H, W), dtype=np.float64)
        cnt = np.zeros((H, W), dtype=np.float64)
        np.add.at(acc, (row, col), ground_xyz[:, 2])
        np.add.at(cnt, (row, col), 1.0)
        dtm = np.full((H, W), np.nan, dtype=np.float32)
        nz = cnt > 0
        dtm[nz] = (acc[nz] / cnt[nz]).astype(np.float32)

    # ---- fill ALL gaps to the full raster extent (no NaN holes inside the extent) ----
    nan_mask = np.isnan(dtm)
    if nan_mask.any() and nz.any():
        done = False
        try:                                    # exact nearest-valued fill, O(H*W)
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(nan_mask, return_distances=False,
                                         return_indices=True)
            dtm = dtm[tuple(idx)].astype(np.float32)
            done = True
        except Exception:
            done = False
        if not done:                            # fallback: 4-neighbour average, iterate until full
            it, cap = 0, max(int(fill_iterations), H + W)   # enough passes to cross the raster
            while np.isnan(dtm).any() and it < cap:
                it += 1
                cur = dtm
                acc2 = np.zeros((H, W), dtype=np.float64)
                cnt2 = np.zeros((H, W), dtype=np.float64)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    sh = np.full_like(cur, np.nan)
                    r0s, r1s = max(dr, 0), H + min(dr, 0)
                    c0s, c1s = max(dc, 0), W + min(dc, 0)
                    r0d, r1d = max(-dr, 0), H + min(-dr, 0)
                    c0d, c1d = max(-dc, 0), W + min(-dc, 0)
                    sh[r0d:r1d, c0d:c1d] = cur[r0s:r1s, c0s:c1s]
                    ok = ~np.isnan(sh)
                    acc2[ok] += sh[ok]
                    cnt2[ok] += 1.0
                new = cur.copy()
                gap = np.isnan(cur) & (cnt2 > 0)
                new[gap] = (acc2[gap] / cnt2[gap]).astype(np.float32)
                dtm = new

    return Raster(data=dtm, x_min=float(x_min), y_min=float(y_min), res=float(res))


def sample_dtm(raster: Raster, xy: np.ndarray) -> np.ndarray:
    """Bilinearly sample the DTM at world coordinates ``xy`` (N,2). NaN-safe."""
    H, W = raster.shape
    fx = (xy[:, 0] - raster.x_min) / raster.res
    fy = (xy[:, 1] - raster.y_min) / raster.res
    x0 = np.clip(np.floor(fx).astype(np.int64), 0, W - 1)
    y0 = np.clip(np.floor(fy).astype(np.int64), 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    wx = np.clip(fx - x0, 0, 1)
    wy = np.clip(fy - y0, 0, 1)

    d = raster.data
    q00, q01 = d[y0, x0], d[y0, x1]
    q10, q11 = d[y1, x0], d[y1, x1]

    def nan_to(a, fallback):
        out = a.copy()
        m = np.isnan(out)
        out[m] = fallback[m]
        return out

    # NaN fallback computed from the sampled corners only -- never reduce over the
    # whole raster. All-NaN corner sets (placeholder/zero-coverage tiles) are expected,
    # so suppress the benign "Mean of empty slice" warning and fall back to 0.0.
    _corner = np.concatenate([np.asarray(q00).ravel(), np.asarray(q01).ravel(),
                              np.asarray(q10).ravel(), np.asarray(q11).ravel()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        _mv = np.nanmean(_corner) if _corner.size else np.nan
    mean_valid = float(_mv) if np.isfinite(_mv) else 0.0
    fb = np.full(xy.shape[0], mean_valid, dtype=np.float64)
    q00, q01, q10, q11 = (nan_to(q, fb) for q in (q00, q01, q10, q11))

    top = q00 * (1 - wx) + q01 * wx
    bot = q10 * (1 - wx) + q11 * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def height_above_prev_dtm(xyz: np.ndarray, raster: Optional[Raster]) -> np.ndarray:
    """Per-point z - DTM_prev(x, y).  Zeros if no previous-year DTM available."""
    if raster is None:
        return np.zeros((xyz.shape[0],), dtype=np.float32)
    ground_z = sample_dtm(raster, xyz[:, :2])
    return (xyz[:, 2] - ground_z).astype(np.float32)


def height_above_ground(
    xyz: np.ndarray,
    labels: np.ndarray,
    res: float = 1.0,
    min_ground: int = 8,
) -> np.ndarray:
    """GrounDiff regression target (Dhaouadi et al., 2025): per-point CONTINUOUS
    height above the bare-earth surface, i.e. the nDSM ``r = z - DTM(x, y)`` in
    metres (Eq. 12, ``L1/L2`` are computed on this quantity, not on bins).

    A bare-earth DTM is built from the points labelled GROUND in this sample
    (``build_dtm_from_ground``: mean GROUND z per cell, gaps filled) and every
    point's height above it is returned. GROUND points sit at ~0; a 15 m tree
    sits at ~15 -- there is no majority-class shortcut in this continuous target,
    which is exactly why regressing it prevents the predict-all-ground collapse
    that a pure classification (mask) loss falls into on imbalanced binary data.

    Returns float32 ``(n,)`` in metres. Points are ``NaN`` (skipped by the loss)
    when their own label is the ignore label, or when the sample has fewer than
    ``min_ground`` ground points (no reliable surface to measure against).
    Vertical height is invariant to the pipeline's z-axis rotation augmentation,
    so this may be computed on centred coordinates.
    """
    n = int(xyz.shape[0])
    out = np.full(n, np.nan, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    g = labels == 1
    if int(g.sum()) < int(min_ground):
        return out
    dtm = build_dtm_from_ground(xyz[g], res=float(res))          # mean GT-ground surface
    z_ref = sample_dtm(dtm, xyz[:, :2]).astype(np.float64)
    ndsm = xyz[:, 2].astype(np.float64) - z_ref                  # nDSM r = z - DTM(x,y)
    from ..utils.laz_io import IGNORE_LABEL
    valid = labels != int(IGNORE_LABEL)
    # also drop points where the surface could not be sampled (NaN z_ref)
    valid &= np.isfinite(ndsm)
    out[valid] = ndsm[valid].astype(np.float32)
    return out


def crop_downsample_raster(raster: Optional[Raster], x_min: float, y_min: float,
                           x_max: float, y_max: float, target_res: float) -> Optional[Raster]:
    """Crop ``raster`` to the world box [x_min,x_max]x[y_min,y_max] and block-mean
    (NaN-aware) downsample to ~``target_res`` metres. Returns a small new Raster with
    geo-referencing updated so ``sample_dtm`` still maps world coords correctly.

    Used at stage 04 so each tile stores only its OWN extent of the previous-year DTM
    at a coarse resolution (default 1 m) instead of a full-region cm-grid copy. The
    network only samples a 64x64 patch per sphere, so 1 m is ample.
    """
    if raster is None:
        return None
    H, W = raster.shape
    res = float(raster.res)
    c0 = max(int(np.floor((x_min - raster.x_min) / res)), 0)
    c1 = min(int(np.ceil((x_max - raster.x_min) / res)) + 1, W)
    r0 = max(int(np.floor((y_min - raster.y_min) / res)), 0)
    r1 = min(int(np.ceil((y_max - raster.y_min) / res)) + 1, H)
    if c1 <= c0 or r1 <= r0:                                   # no overlap with the tile
        return Raster(np.full((1, 1), np.nan, np.float32), float(x_min), float(y_min), float(target_res))
    sub = np.asarray(raster.data[r0:r1, c0:c1], dtype=np.float32)
    new_x_min = raster.x_min + c0 * res
    new_y_min = raster.y_min + r0 * res
    factor = max(int(round(float(target_res) / res)), 1)
    if factor > 1:
        h, w = sub.shape
        h2, w2 = (h // factor) * factor, (w // factor) * factor
        if h2 >= factor and w2 >= factor:
            blk = sub[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor)
            import warnings
            with warnings.catch_warnings():                   # all-NaN blocks -> NaN (handled downstream)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sub = np.nanmean(blk, axis=(1, 3)).astype(np.float32)
            res = res * factor
    return Raster(sub.astype(np.float32), float(new_x_min), float(new_y_min), float(res))


def crop_dtm_patch(
    raster: Optional[Raster],
    x0: float,
    y0: float,
    tile_size: float,
    size: int,
    origin_z: float = 0.0,
) -> np.ndarray:
    """Sample the previous-year DTM over a tile into a (size, size) patch.

    The patch spans ``[x0, x0+tile_size] x [y0, y0+tile_size]`` so that a point
    with local coordinate ``x_local in [0, tile_size]`` maps to patch column
    ``x_local / tile_size * (size-1)`` (row index increases with y).  The tile's
    vertical origin ``origin_z`` is subtracted so the patch shares the network's
    local vertical frame.  Returns zeros if no DTM is available.
    """
    if raster is None:
        return np.zeros((size, size), dtype=np.float32)
    xs = np.linspace(x0, x0 + tile_size, size)
    ys = np.linspace(y0, y0 + tile_size, size)
    gx, gy = np.meshgrid(xs, ys)                      # (size,size), row index = y
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    z = sample_dtm(raster, xy).reshape(size, size)
    return (z - origin_z).astype(np.float32)


# =========================================================================== #
# Previous-year CLASSIFICATION raster (Deviation A, GrounDiff-informed)
#
# Generalises the single-channel DTM above to a multi-channel raster built from
# the previous survey's *classified* cloud, giving the network a spatial prior
# on where ground sits (see models/prior_raster_encoder.py). Channels:
#   0 DTM   - mean z of prior GROUND returns (bare earth)
#   1 DSM   - max z of all prior returns (top surface)
#   2 nDSM  - DSM - DTM (non-ground height; GrounDiff's residual prior)
#   3 gprob - fraction of prior returns classified as ground (observed confidence)
#   4 cover - 1 where the cell had any prior return, else 0
# =========================================================================== #
PRIOR_RASTER_CHANNELS = ("dtm", "dsm", "ndsm", "ground_prob", "coverage")


@dataclass
class MultiRaster:
    data: np.ndarray          # (C, H, W) float32
    x_min: float
    y_min: float
    res: float
    channels: tuple = PRIOR_RASTER_CHANNELS

    @property
    def shape(self):
        return self.data.shape


def _nearest_fill(arr: np.ndarray) -> np.ndarray:
    """Fill NaN holes with the nearest valid cell (exact EDT if scipy present)."""
    nan_mask = np.isnan(arr)
    if not nan_mask.any() or nan_mask.all():
        return arr
    try:
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        return arr[tuple(idx)].astype(np.float32)
    except Exception:
        out = arr.copy()
        H, W = out.shape
        it, cap = 0, H + W
        while np.isnan(out).any() and it < cap:
            it += 1
            cur = out
            acc = np.zeros((H, W)); cnt = np.zeros((H, W))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                sh = np.full_like(cur, np.nan)
                r0s, r1s = max(dr, 0), H + min(dr, 0)
                c0s, c1s = max(dc, 0), W + min(dc, 0)
                r0d, r1d = max(-dr, 0), H + min(-dr, 0)
                c0d, c1d = max(-dc, 0), W + min(-dc, 0)
                sh[r0d:r1d, c0d:c1d] = cur[r0s:r1s, c0s:c1s]
                ok = ~np.isnan(sh); acc[ok] += sh[ok]; cnt[ok] += 1.0
            gap = np.isnan(cur) & (cnt > 0)
            cur = cur.copy(); cur[gap] = (acc[gap] / cnt[gap]).astype(np.float32)
            out = cur
        return out


def build_prior_raster_from_prev(xyz: np.ndarray, classification: np.ndarray,
                                 ground_classes, res: float = 1.0,
                                 bounds=None) -> MultiRaster:
    """Rasterise a previous-year *classified* cloud into the 5-channel prior raster."""
    if bounds is None:
        x_min, y_min = float(xyz[:, 0].min()), float(xyz[:, 1].min())
        x_max, y_max = float(xyz[:, 0].max()), float(xyz[:, 1].max())
    else:
        x_min, y_min, x_max, y_max = bounds
    W = max(int(np.ceil((x_max - x_min) / res)), 1)
    H = max(int(np.ceil((y_max - y_min) / res)), 1)

    col = np.clip(((xyz[:, 0] - x_min) / res).astype(np.int64), 0, W - 1)
    row = np.clip(((xyz[:, 1] - y_min) / res).astype(np.int64), 0, H - 1)
    z = xyz[:, 2].astype(np.float64)
    is_g = np.isin(classification, np.asarray(ground_classes))

    # ground mean z (DTM)
    g_acc = np.zeros((H, W)); g_cnt = np.zeros((H, W))
    np.add.at(g_acc, (row[is_g], col[is_g]), z[is_g])
    np.add.at(g_cnt, (row[is_g], col[is_g]), 1.0)
    dtm = np.full((H, W), np.nan, np.float32)
    gnz = g_cnt > 0
    dtm[gnz] = (g_acc[gnz] / g_cnt[gnz]).astype(np.float32)

    # DSM = max z of all returns
    dsm = np.full((H, W), -np.inf, np.float32)
    np.maximum.at(dsm, (row, col), z.astype(np.float32))
    dsm[~np.isfinite(dsm)] = np.nan

    # ground probability + coverage
    all_cnt = np.zeros((H, W)); np.add.at(all_cnt, (row, col), 1.0)
    cover = (all_cnt > 0).astype(np.float32)
    gprob = np.zeros((H, W), np.float32)
    np.divide(g_cnt, np.maximum(all_cnt, 1.0), out=gprob)
    gprob = gprob.astype(np.float32)

    # fill height holes (so DTM/DSM defined inside extent); nDSM after fill
    dtm = _nearest_fill(dtm)
    dsm = _nearest_fill(dsm)
    ndsm = (dsm - dtm).astype(np.float32)
    ndsm = np.clip(ndsm, 0.0, None)  # non-ground height is non-negative

    data = np.stack([dtm, dsm, ndsm, gprob, cover], axis=0).astype(np.float32)
    return MultiRaster(data=data, x_min=x_min, y_min=y_min, res=float(res))


def _to_dtm_raster(mr: MultiRaster) -> Raster:
    """View the DTM channel of a MultiRaster as a plain Raster (for z - DTM_prev)."""
    return Raster(data=mr.data[0].astype(np.float32), x_min=mr.x_min,
                  y_min=mr.y_min, res=mr.res)


def prior_from_dtm_grid(band, x_min, y_min, res, nodata=None) -> MultiRaster:
    """Build the 5-channel prior from ONE bare-earth DTM grid (e.g. a hand-crafted
    previous-year raster). Coverage is taken from the VALID (non-NoData, finite) cells
    BEFORE gaps are nearest-filled, so a partial raster honestly reports where it has no
    data. DSM=DTM and nDSM=0 (a DTM carries no surface/height info); ground_prob=coverage.
    The DTM channel is nearest-filled (finite inside extent) purely so sampling never
    returns NaN; the coverage channel is what tells the model a cell was real vs filled."""
    band = np.asarray(band, dtype=np.float32)
    if nodata is not None:
        band = np.where(band == np.float32(nodata), np.nan, band)
    band = np.where(np.isfinite(band), band, np.nan)              # +/-inf -> NaN as well
    cover = np.isfinite(band).astype(np.float32)                  # honest coverage (pre-fill)
    dtm = _nearest_fill(band).astype(np.float32)
    dtm = np.where(np.isfinite(dtm), dtm, 0.0).astype(np.float32) # all-NoData raster -> zeros
    ndsm = np.zeros_like(dtm)
    data = np.stack([dtm, dtm, ndsm, cover, cover], axis=0).astype(np.float32)
    return MultiRaster(data=data, x_min=float(x_min), y_min=float(y_min),
                       res=float(res), channels=PRIOR_RASTER_CHANNELS)


def prior_from_raster_file(path, res=None) -> Optional[MultiRaster]:
    """Load a hand-crafted previous-year raster into the 5-channel prior.

    GeoTIFF / ASC (any rasterio-readable grid) are read and resampled to ``res`` metres
    (native resolution if ``res`` is None), with the file's NoData marked uncovered. A
    ``.npz`` in our own format (data/x_min/y_min/res[/channels]) is loaded directly: a
    single-channel array is treated as a DTM grid, a multi-channel array as a ready prior.
    Returns None if the file cannot be read."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".npz", ".npy"):
        d = np.load(path, allow_pickle=True)
        data = np.asarray(d["data"], dtype=np.float32)
        if data.ndim == 2:
            return prior_from_dtm_grid(data, float(d["x_min"]), float(d["y_min"]), float(d["res"]))
        chans = tuple(d["channels"]) if "channels" in d else PRIOR_RASTER_CHANNELS
        return MultiRaster(data, float(d["x_min"]), float(d["y_min"]), float(d["res"]), chans)
    import rasterio
    from rasterio.merge import merge
    with rasterio.open(path) as _s:
        nodata = _s.nodata
        native = abs(float(_s.transform.a))
    tgt = float(res) if res else (native or 1.0)
    src = rasterio.open(path)
    try:
        mosaic, transform = merge([src], res=(tgt, tgt))
    finally:
        src.close()
    band = np.flipud(mosaic[0].astype(np.float32))                # rasterio row0=north -> row0=y_min
    x_min = float(transform.c)
    y_min = float(transform.f) - band.shape[0] * tgt
    return prior_from_dtm_grid(band, x_min, y_min, tgt, nodata=nodata)


def prior_from_raster_files(paths, res=None, bounds=None, margin=25.0) -> Optional[MultiRaster]:
    """Build the 5-channel prior from one OR SEVERAL rasterio-readable rasters
    (mosaicked; e.g. two project DTMs covering one survey area). If ``bounds``
    (xmin, ymin, xmax, ymax) is given -- typically the cloud's extent -- the mosaic is
    cropped to it plus ``margin`` metres, so a project-wide DTM yields a compact
    per-cloud prior. NoData stays honest: uncovered cells -> coverage 0."""
    paths = [p for p in (paths or []) if p]
    if not paths:
        return None
    import rasterio
    from rasterio.merge import merge
    with rasterio.open(paths[0]) as _s:
        nodata = _s.nodata
        native = abs(float(_s.transform.a))
    tgt = float(res) if res else (native or 1.0)
    srcs = [rasterio.open(p) for p in paths]
    try:
        kw = dict(res=(tgt, tgt))
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            kw["bounds"] = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
        if nodata is not None:
            kw["nodata"] = nodata
        mosaic, transform = merge(srcs, **kw)
    finally:
        for s in srcs:
            s.close()
    band = np.flipud(mosaic[0].astype(np.float32))                # rasterio row0=north -> row0=y_min
    x_min = float(transform.c)
    y_min = float(transform.f) - band.shape[0] * tgt
    return prior_from_dtm_grid(band, x_min, y_min, tgt, nodata=nodata)


def prior_coverage_mask(prior: "MultiRaster", xy: np.ndarray) -> np.ndarray:
    """Per-point boolean: is (x,y) INSIDE the raster extent AND on a covered cell?

    ``sample_dtm`` clamps out-of-extent coordinates to the edge value, so on its own it
    would silently extrapolate a partial raster past its bounds. This returns False for
    points outside the bounding box or over a NoData (coverage<0.5) cell, so the caller
    can neutralise their prev-DTM feature rather than trust an edge value."""
    C, H, W = prior.shape
    fx = (np.asarray(xy)[:, 0] - prior.x_min) / prior.res
    fy = (np.asarray(xy)[:, 1] - prior.y_min) / prior.res
    in_box = (fx >= 0) & (fx <= W - 1) & (fy >= 0) & (fy <= H - 1)
    if C < 5:
        return in_box
    ci = np.clip(np.round(fx).astype(np.int64), 0, W - 1)
    ri = np.clip(np.round(fy).astype(np.int64), 0, H - 1)
    cov = np.asarray(prior.data[4])[ri, ci]
    return in_box & (cov >= 0.5)


def crop_downsample_multiraster(mr, x_min, y_min, x_max, y_max, target_res):
    """Crop a MultiRaster to a world box and block-mean downsample each channel
    to ~``target_res`` m (NaN-free already). Returns a small new MultiRaster."""
    if mr is None:
        return None
    C, H, W = mr.shape
    res = float(mr.res)
    c0 = max(int(np.floor((x_min - mr.x_min) / res)), 0)
    c1 = min(int(np.ceil((x_max - mr.x_min) / res)) + 1, W)
    r0 = max(int(np.floor((y_min - mr.y_min) / res)), 0)
    r1 = min(int(np.ceil((y_max - mr.y_min) / res)) + 1, H)
    if c1 <= c0 or r1 <= r0:
        return MultiRaster(np.zeros((C, 1, 1), np.float32), float(x_min), float(y_min),
                           float(target_res), mr.channels)
    sub = np.asarray(mr.data[:, r0:r1, c0:c1], dtype=np.float32)
    new_x_min = mr.x_min + c0 * res
    new_y_min = mr.y_min + r0 * res
    factor = max(int(round(float(target_res) / res)), 1)
    if factor > 1:
        c, h, w = sub.shape
        h2, w2 = (h // factor) * factor, (w // factor) * factor
        if h2 >= factor and w2 >= factor:
            blk = sub[:, :h2, :w2].reshape(c, h2 // factor, factor, w2 // factor, factor)
            sub = blk.mean(axis=(2, 4)).astype(np.float32)
            res = res * factor
    return MultiRaster(sub.astype(np.float32), float(new_x_min), float(new_y_min),
                       float(res), mr.channels)


def crop_multiraster_patch(mr, x0: float, y0: float, tile_size: float, size: int,
                           origin_z: float = 0.0) -> np.ndarray:
    """Sample a MultiRaster over [x0,x0+tile]x[y0,y0+tile] into a (C, size, size)
    patch. Height channels (dtm/dsm) have ``origin_z`` subtracted to share the
    network's local vertical frame; nDSM/gprob/coverage are left as-is. Bilinear,
    with row index increasing with y (matches the per-point sampler)."""
    C = 5 if mr is None else mr.shape[0]
    if mr is None:
        return np.zeros((C, size, size), dtype=np.float32)
    xs = np.linspace(x0, x0 + tile_size, size)
    ys = np.linspace(y0, y0 + tile_size, size)
    gx, gy = np.meshgrid(xs, ys)
    out = np.zeros((mr.shape[0], size, size), dtype=np.float32)
    # reuse the single-channel bilinear sampler per channel
    for ci in range(mr.shape[0]):
        r = Raster(mr.data[ci], mr.x_min, mr.y_min, mr.res)
        z = sample_dtm(r, np.column_stack([gx.ravel(), gy.ravel()])).reshape(size, size)
        name = mr.channels[ci] if ci < len(mr.channels) else ""
        if name in ("dtm", "dsm"):
            z = z - origin_z
        out[ci] = z.astype(np.float32)
    return out


def load_prior_raster(path: "Optional[str]") -> "Optional[MultiRaster]":
    """Load a previous-year prior raster ``.npz`` for inference, returning a
    :class:`MultiRaster`. Accepts both the 5-channel prior written by
    ``02_build_prior_raster.py`` (data shape ``(C,H,W)`` + a ``channels`` array)
    and a legacy single-channel DTM ``.npz`` (data shape ``(H,W)``), which is
    promoted to the 5-channel layout with a ground-everywhere prior so the raster
    branch still runs. Returns ``None`` if ``path`` is falsy or missing."""
    import os
    if not path or not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    data = np.asarray(d["data"], dtype=np.float32)
    if data.ndim == 2:                                   # legacy single-channel DTM
        data = np.stack([data, data, np.zeros_like(data),
                         np.ones_like(data), np.ones_like(data)], 0)
    chans = tuple(d["channels"]) if "channels" in d else PRIOR_RASTER_CHANNELS
    return MultiRaster(data, float(d["x_min"]), float(d["y_min"]), float(d["res"]), chans)
