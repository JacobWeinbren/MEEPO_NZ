"""
Shallow per-point geometric features, assembled into the model input vector.

The paper replaces RGB input with three groups of geometry-derived "shallow"
features, computed from neighbourhood domain knowledge:

  * 3.1.1 Average (weighted) elevation - equations (1)-(2)
        P_i  = (Z_i - Z_min) / (Z_max - Z_min)
        Z_ave = sum_i (Z_i * P_i) / sum_i P_i           (over a local region)
  * 3.1.2 Surface curvature - equations (3)-(5)
        fit a local plane to the k nearest neighbours, build the covariance
        matrix M, take eigenvalues l0 <= l1 <= l2, curvature = l0/(l0+l1+l2)
  * 3.1.3 Higher-order moments
        the monomials (x^2, y^2, z^2, xy, xz, yz) of the (local) coordinates,
        following Joseph-Rivlin et al. (Momen(e)t)

Together with the point's 3 xyz coordinates these give the 11-channel input
the network sees (the paper's [n, 11], mapped to [n, d] by the first linear
layer).  The single sanctioned deviation adds the previous-year ground DTM as a
separate raster branch (see models/dtm_encoder.py), not as an extra channel here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def compute_mean_elevation(xyz: np.ndarray, tree, k: int, idx=None) -> np.ndarray:
    """Weighted mean elevation of every point's k-NN region - equations (1)-(2).

    ``idx`` may be a precomputed (n, k) neighbour-index array (shared with the
    curvature feature) to avoid a redundant KD-tree query.
    """
    n = xyz.shape[0]
    k = min(k, n)
    if idx is None:
        _, idx = tree.query(xyz, k=k)
    if idx.ndim == 1:
        idx = idx[:, None]
    Z = xyz[:, 2]
    neigh_Z = Z[idx]                                   # (n, k)
    zmin = neigh_Z.min(axis=1, keepdims=True)
    zmax = neigh_Z.max(axis=1, keepdims=True)
    denom = np.maximum(zmax - zmin, 1e-6)
    P = (neigh_Z - zmin) / denom                       # (n, k) weights, eq (1)
    zave = (neigh_Z * P).sum(axis=1) / np.maximum(P.sum(axis=1), 1e-6)   # eq (2)
    return zave.astype(np.float32)


def compute_curvature(xyz: np.ndarray, tree, k: int, idx=None) -> np.ndarray:
    """Surface curvature l0/(l0+l1+l2) of every point's k-NN - equations (3)-(5).

    ``idx`` may be a precomputed (n, k) neighbour-index array (shared with the
    mean-elevation feature) to avoid a redundant KD-tree query.
    """
    n = xyz.shape[0]
    if idx is None:
        k = min(max(k, 3), n)
        _, idx = tree.query(xyz, k=k)
        if idx.ndim == 1:
            idx = idx[:, None]
    else:
        if idx.ndim == 1:
            idx = idx[:, None]
        k = idx.shape[1]                               # divisor = actual neighbour count
    neigh = xyz[idx]                                   # (n, k, 3)
    mean = neigh.mean(axis=1, keepdims=True)
    centered = neigh - mean
    # covariance matrices M (eq 4):  (n, 3, 3)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    # eigenvalues (ascending) -> l0 <= l1 <= l2
    eigvals = np.linalg.eigvalsh(cov)                  # (n, 3) ascending
    l0 = eigvals[:, 0]
    s = eigvals.sum(axis=1)
    curv = l0 / np.maximum(s, 1e-12)                   # eq (5)
    return curv.astype(np.float32)


def compute_higher_moments(xyz_local: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Per-point monomials (x^2, y^2, z^2, xy, xz, yz) of local coordinates."""
    p = xyz_local / max(scale, 1e-6)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    moments = np.stack([x * x, y * y, z * z, x * y, x * z, y * z], axis=1)
    return moments.astype(np.float32)




def compute_shallow_geom(xyz: np.ndarray, cfg):
    """Precompute the KD-tree shallow features (weighted mean elevation, curvature)
    on ``xyz`` with a single shared k-NN query. Returns ``(mean_elev, curvature)``;
    each is a ``(N,)`` float32 array, or ``None`` when its config flag is off.

    ``mean_elev`` is the RAW weighted-mean elevation (in ``xyz``'s own z units); the
    consumer re-centres it per input region (subtract the region-centre z), which is
    exact because the weighted mean is translation-equivariant. Curvature is a scale-
    and rotation-invariant eigenvalue ratio, so the precomputed value matches the
    per-region one up to the (minor) jitter augmentation. Computed once per tile at
    stage 04 so it is not rebuilt for every sphere every epoch.
    """
    if cKDTree is None:
        raise RuntimeError("scipy is required for shallow feature extraction")
    n = xyz.shape[0]
    need_elev = bool(getattr(cfg, "use_mean_elevation", False))
    need_curv = bool(getattr(cfg, "use_curvature", False))
    if n == 0 or not (need_elev or need_curv):
        return None, None
    ke = int(getattr(cfg, "feature_knn", 16))
    tree = cKDTree(xyz)
    kq = min(max(ke, 3) if need_curv else ke, n)
    _, idx = tree.query(xyz, k=kq)
    if idx.ndim == 1:
        idx = idx[:, None]
    elev = compute_mean_elevation(xyz, tree, ke, idx=idx[:, :min(ke, n)]) if need_elev else None
    curv = compute_curvature(xyz, tree, ke, idx=idx) if need_curv else None
    return elev, curv


def assemble_features(
    xyz: np.ndarray,
    cfg,
    dtm_height: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    return_number: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    return_ratio: Optional[np.ndarray] = None,
    mean_elev_precomp: Optional[np.ndarray] = None,
    curvature_precomp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the (N, F) input feature matrix according to the config flags.

    ``xyz`` must already be tile-local (centred so that the network sees small
    coordinates).  ``dtm_height`` is the per-point height above the previous-year
    DTM (the deviation channel); ``rgb`` is only used by the RGB ablation.

    ``mean_elev_precomp`` / ``curvature_precomp`` are per-point shallow features
    precomputed once at stage 04 (already re-centred for elevation by the caller).
    When given, the expensive KD-tree + eigendecomposition is skipped entirely.
    """
    if cKDTree is None:
        raise RuntimeError("scipy is required for shallow feature extraction")

    n = xyz.shape[0]
    cols = []

    # input = xyz coordinates + 8 shallow geometric features ([n, 11] in the
    # paper, Sec 3.2). Feed the 3 local coordinates as the first input channels;
    # the initial linear layer maps [n, 11] -> [n, d] and standardisation (norm
    # stats) normalises them. --no-xyz-feature drops them for the KPConv-style run.
    if getattr(cfg, "use_xyz_in_features", False):
        cols.append(xyz.astype(np.float32))                  # x, y, z (local)
    # KPConv's constant-1 channel (ablation alternative; the paper uses xyz).
    if getattr(cfg, "use_constant_feature", False):
        cols.append(np.ones((n, 1), dtype=np.float32))

    if cfg.use_rgb and rgb is not None:
        # RGB ablation: coordinates-as-feature + RGB only
        cols.append((rgb.astype(np.float32) / 255.0) if rgb.max() > 1.5 else rgb.astype(np.float32))
    else:
        ke = int(getattr(cfg, "feature_knn", 16))
        need_elev = bool(cfg.use_mean_elevation)
        need_curv = bool(cfg.use_curvature)
        have_elev = mean_elev_precomp is not None
        have_curv = curvature_precomp is not None
        # Build the KD-tree + run ONE shared k-NN query only if something still has
        # to be computed (i.e. not supplied precomputed from stage 04).
        tree = None
        idx_shared = None
        if (need_elev and not have_elev) or (need_curv and not have_curv):
            tree = cKDTree(xyz)
            kq = min(max(ke, 3) if (need_curv and not have_curv) else ke, n)
            _, idx_shared = tree.query(xyz, k=kq)
            if idx_shared.ndim == 1:
                idx_shared = idx_shared[:, None]
        if need_elev:
            if have_elev:
                cols.append(np.asarray(mean_elev_precomp, dtype=np.float32).reshape(n, 1))
            else:
                cols.append(compute_mean_elevation(xyz, tree, ke, idx=idx_shared[:, :min(ke, n)])[:, None])
        if need_curv:
            if have_curv:
                cols.append(np.asarray(curvature_precomp, dtype=np.float32).reshape(n, 1))
            else:
                cols.append(compute_curvature(xyz, tree, ke, idx=idx_shared)[:, None])
        if cfg.use_higher_moments:
            # raw monomials of the (KPConv-centred) input coordinates - paper 3.1.3
            cols.append(compute_higher_moments(xyz, scale=1.0))

    # Deviation #2 (not in the paper): laser-return cue. number_of_returns
    # separates multi-return pulses (vegetation / edges) from single-return ones
    # (bare / hard surfaces). A single per-point channel (the return count); the
    # return_number/number_of_returns ratio is no longer used.
    if getattr(cfg, "use_return_features", False):
        if num_returns is None:
            num_returns = np.ones((n,), dtype=np.float32)
        nr = num_returns.astype(np.float32).reshape(-1)
        cols.append(nr[:, None])                       # number of returns (count) - single channel

    # Deviation (not in the paper): normalised return ratio = return_number /
    # number_of_returns in (0,1]. ~1 => last/only return (the echo that reached the
    # surface -> likely ground); <1 => an earlier return (canopy / vegetation). It
    # complements the count above and targets the low-vegetation-vs-ground confusion
    # under canopy. Preferably supplied precomputed (per raw point, then averaged in
    # preprocessing = true mean of per-point ratios); else derived from the averaged
    # count/return_number here. Standardised by norm_stats. --no-return-ratio.
    if getattr(cfg, "use_return_ratio", False):
        if return_ratio is None:
            if return_number is not None and num_returns is not None:
                nr_ = np.maximum(num_returns.astype(np.float32).reshape(-1), 1.0)
                return_ratio = return_number.astype(np.float32).reshape(-1) / nr_
            else:
                return_ratio = np.ones((n,), dtype=np.float32)
        cols.append(return_ratio.astype(np.float32).reshape(-1)[:, None])

    # Intensity (not in the paper): per-point return strength. Bare / hard
    # surfaces and vegetation reflect the laser differently, so intensity is a
    # radiometric ground/vegetation cue complementing the geometric features.
    # Raw values are standardised by the per-channel norm_stats, as for the
    # geometric channels. Disable with --no-intensity.
    if getattr(cfg, "use_intensity", False):
        if intensity is None:
            intensity = np.zeros((n,), dtype=np.float32)
        inten = intensity.astype(np.float32).reshape(-1)
        # LiDAR intensity is heavy-tailed (16-bit DN: most points ~0-2000, a long tail
        # to ~60000). A plain global z-score is then dominated by the tail - std blows
        # up and the common range collapses toward 0. log1p compresses the tail so the
        # downstream per-channel standardisation (norm_stats) gives the bulk real
        # resolution. Monotonic + parameter-free, and applied HERE so norm_stats and
        # train/val/test/inference all see the identical transform.
        if getattr(cfg, "intensity_log", True):
            inten = np.log1p(np.maximum(inten, 0.0)).astype(np.float32)
        cols.append(inten[:, None])

    if cfg.use_prev_dtm:
        if dtm_height is None:
            dtm_height = np.zeros((n,), dtype=np.float32)
        cols.append(dtm_height.astype(np.float32)[:, None])

    if not cols:   # coordinates-only ablation -> constant feature
        cols.append(np.ones((n, 1), dtype=np.float32))

    feats = np.concatenate(cols, axis=1).astype(np.float32)
    return feats


def expected_feature_dim(cfg) -> int:
    """Number of input channels implied by the config (for sanity checks)."""
    d = 3 if getattr(cfg, "use_xyz_in_features", False) else 0   # xyz coordinates
    d += 1 if getattr(cfg, "use_constant_feature", False) else 0 # KPConv constant 1
    if cfg.use_rgb:
        d += 3
    else:
        d += 1 if cfg.use_mean_elevation else 0
        d += 1 if cfg.use_curvature else 0
        d += 6 if cfg.use_higher_moments else 0
    d += 1 if getattr(cfg, "use_return_features", False) else 0  # number_of_returns (count)
    d += 1 if getattr(cfg, "use_return_ratio", False) else 0     # return_number/number_of_returns ratio
    d += 1 if getattr(cfg, "use_intensity", False) else 0        # per-point intensity
    d += 1 if cfg.use_prev_dtm else 0
    return max(d, 1)
