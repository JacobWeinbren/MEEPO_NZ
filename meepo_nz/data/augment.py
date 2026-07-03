"""
KPConv-style data augmentation for training tiles.

Following KPConv (Thomas et al. 2019; cf. KP-FCNN training in the supplementary
and the published config): random rotation about the vertical axis, anisotropic
scaling in ``[scale_min, scale_max]``, an X-axis symmetry (flip), and small
Gaussian jitter.

Two subtleties specific to this codebase are handled here:

  * the eight shallow features are recomputed from the augmented points by the
    dataset (so curvature / moments / mean-elevation match the new geometry);
  * the previous-year **DTM patch** (the deviation's raster branch) is warped by
    the *same* planar transform, so the model's per-point sampling of the patch
    stays aligned with the points (otherwise a rotated point would read terrain
    from the wrong place). Patch elevations are scaled by the vertical scale.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import affine_transform
except Exception:                       # pragma: no cover
    affine_transform = None


def _planar_matrix(cfg, rng):
    """Return (A, sz): 2x2 XY linear map (rotation·scale·flip) and z-scale."""
    theta = rng.uniform(0.0, 2.0 * np.pi) if getattr(cfg, "augment_rotation_z", False) else 0.0
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)

    smin, smax = float(cfg.augment_scale_min), float(cfg.augment_scale_max)
    if getattr(cfg, "augment_anisotropic", False):
        sx, sy, sz = rng.uniform(smin, smax, size=3)
    else:
        sx = sy = sz = float(rng.uniform(smin, smax))
    S = np.diag([sx, sy]).astype(np.float64)

    F = np.eye(2, dtype=np.float64)
    if getattr(cfg, "augment_flip_x", False) and rng.random() < 0.5:
        F[0, 0] = -1.0
    if getattr(cfg, "augment_flip_y", False) and rng.random() < 0.5:
        F[1, 1] = -1.0                              # MEEPO RandomFlip flips x AND y, each p=0.5

    return R @ S @ F, float(sz)


def _warp_patch(patch: np.ndarray, A: np.ndarray, sz: float) -> np.ndarray:
    """Warp the (H,W) DTM patch by the same planar map ``A`` (about its centre).

    ``affine_transform`` maps output coords -> input coords, so we use ``A^{-1}``.
    The patch is indexed [row=y, col=x]; ``A`` is in (x, y) order, so we conjugate
    by the axis-swap P to express it in (row, col). Elevations scale by ``sz``.
    """
    if affine_transform is None or patch.ndim != 2:
        return (patch * sz).astype(np.float32)
    P = np.array([[0.0, 1.0], [1.0, 0.0]])          # swap (x,y) <-> (row,col)
    A_rc = P @ A @ P
    Ainv = np.linalg.inv(A_rc)
    h, w = patch.shape
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    offset = center - Ainv @ center
    out = affine_transform(patch.astype(np.float32), Ainv, offset=offset,
                           order=1, mode="nearest")
    return (out * sz).astype(np.float32)


_ELASTIC_WARNED = False


def _elastic_distortion(coords: np.ndarray, granularity: float, magnitude: float, rng,
                        max_grid: int = 128) -> np.ndarray:
    """Pointcept ElasticDistortion: a smooth random 3D displacement field sampled at
    ``granularity`` (m) and scaled by ``magnitude`` (m). Ported from Pointcept (two box-blur
    passes of a Gaussian noise grid, then trilinear interpolation).

    GUARD: Pointcept sizes the noise grid as (extent/granularity)^3. That was tuned for ~5 m
    indoor rooms; on ~200 m aerial blocks at MEEPO's 0.2 m granularity the grid is ~1000^3
    (tens of GB per call). To keep host RAM bounded we cap each axis at ``max_grid`` cells,
    which raises the EFFECTIVE granularity on large blocks (a coarser warp). At MEEPO's indoor
    scale the cap never binds, so this is faithful there and merely survivable here."""
    global _ELASTIC_WARNED
    import scipy.ndimage as _ndi
    from scipy.interpolate import RegularGridInterpolator as _RGI
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0 or granularity <= 0.0:
        return coords
    bx = np.ones((3, 1, 1, 1), np.float64) / 3.0
    by = np.ones((1, 3, 1, 1), np.float64) / 3.0
    bz = np.ones((1, 1, 3, 1), np.float64) / 3.0
    cmin = coords.min(0)
    ext = (coords - cmin).max(0)
    # coarsen granularity if the native grid would exceed max_grid cells on any axis
    eff_g = max(float(granularity), float(ext.max()) / float(max(max_grid - 3, 1)))
    if eff_g > granularity * 1.001 and not _ELASTIC_WARNED:
        _ELASTIC_WARNED = True
        print(f"[augment] ElasticDistortion: block extent ~{ext.max():.0f} m at granularity "
              f"{granularity} m would need a {int(ext.max()//granularity)}^3 noise grid (tens of GB); "
              f"coarsening granularity to {eff_g:.2f} m to cap the grid at {max_grid}^3. "
              f"Pointcept's 0.2 m is indoor-scale -- this aug is not faithfully reproducible at "
              f"aerial block size and still desyncs the prior; consider leaving --augment-elastic off.",
              flush=True)
    ndim = (ext // eff_g).astype(int) + 3
    noise = rng.standard_normal((int(ndim[0]), int(ndim[1]), int(ndim[2]), 3)).astype(np.float64)
    for _ in range(2):
        noise = _ndi.convolve(noise, bx, mode="constant", cval=0.0)
        noise = _ndi.convolve(noise, by, mode="constant", cval=0.0)
        noise = _ndi.convolve(noise, bz, mode="constant", cval=0.0)
    ax = [np.linspace(c0 - eff_g, c0 + eff_g * (n - 2), int(n))
          for c0, n in zip(cmin, ndim)]
    interp = _RGI(ax, noise, bounds_error=False, fill_value=0.0)
    return coords + interp(coords) * magnitude


def augment_tile(local: np.ndarray, dtm_patch: np.ndarray, cfg, rng,
                 tile_size: float, pivot: float = None):
    """Apply the augmentation to one tile or input sphere.

    ``local``      : (N,3) coordinates. For tiles, x,y in [0, tile_size]; for
                     KPConv input spheres, coordinates are already centred at the
                     sphere centre (pass ``pivot=0.0``).
    ``dtm_patch``  : (H,W) previous-year DTM patch (local vertical frame)
    ``pivot``      : in-plane rotation/scale centre (defaults to ``tile_size/2``).
    returns the augmented ``(local, dtm_patch)``.
    """
    A, sz = _planar_matrix(cfg, rng)
    center = (tile_size / 2.0) if pivot is None else float(pivot)

    out = local.astype(np.float64).copy()

    # x/y micro-tilt (MEEPO RandomRotate x/y, angle +-augment_tilt_xy rad, p=0.5 each axis),
    # about the (centred) origin. NOTE: a tilt mixes z into (x,y) by a z-dependent amount the
    # 2D georeferenced prior raster (Deviation A) cannot follow, so with use_dtm_raster on it
    # desyncs the prior by ~relief*sin(angle). Off by default (augment_tilt_xy=0).
    tmax = float(getattr(cfg, "augment_tilt_xy", 0.0))
    if tmax > 0.0:
        for axis in (0, 1):
            if rng.random() < 0.5:
                a = float(rng.uniform(-tmax, tmax)); ca, sa = np.cos(a), np.sin(a)
                R = np.eye(3, dtype=np.float64)
                if axis == 0:   R[1, 1], R[1, 2], R[2, 1], R[2, 2] = ca, -sa, sa, ca   # about x
                else:           R[0, 0], R[0, 2], R[2, 0], R[2, 2] = ca,  sa, -sa, ca   # about y
                out = out @ R.T

    xy = out[:, :2] - center
    out[:, :2] = xy @ A.T + center                  # rotate / scale / flip in-plane
    out[:, 2] = out[:, 2] * sz                       # vertical scale

    noise = float(getattr(cfg, "augment_noise", 0.0))
    if noise > 0.0:
        jit = rng.normal(0.0, noise, size=out.shape)
        clip = float(getattr(cfg, "augment_noise_clip", 0.0))
        if clip > 0.0:
            np.clip(jit, -clip, clip, out=jit)         # MEEPO RandomJitter sigma=0.005, clip=0.02
        out += jit

    # ElasticDistortion (MEEPO distortion_params=[[0.2,0.4],[0.8,1.6]]): smooth random 3D
    # displacement at each (granularity_m, magnitude_m). Warps the ground SURFACE (up to the
    # magnitude, ~1.6 m) and moves (x,y) -> desyncs the 2D prior; off by default (augment_elastic).
    if bool(getattr(cfg, "augment_elastic", False)):
        for gran, mag in (getattr(cfg, "augment_elastic_params", None) or ((0.2, 0.4), (0.8, 1.6))):
            out = _elastic_distortion(out, float(gran), float(mag), rng)

    if dtm_patch is None:
        patch = dtm_patch
    elif dtm_patch.ndim == 3:
        patch = _warp_multipatch(dtm_patch, A, sz)
    else:
        patch = _warp_patch(dtm_patch, A, sz)
    return out.astype(np.float32), patch


# Channels of the multi-channel prior raster that are HEIGHTS (vertical-scaled by
# sz). nDSM is a height difference, so it scales too; ground-prob/coverage do not.
_PRIOR_HEIGHT_CHANNELS = (0, 1, 2)   # dtm, dsm, ndsm


def _warp_multipatch(patch: np.ndarray, A: np.ndarray, sz: float) -> np.ndarray:
    """Warp a (C, H, W) prior raster patch by the planar map ``A``; vertical-scale
    only the height channels by ``sz`` (ground-prob / coverage are left as-is)."""
    if affine_transform is None:
        out = patch.astype(np.float32).copy()
        for c in _PRIOR_HEIGHT_CHANNELS:
            if c < out.shape[0]:
                out[c] *= sz
        return out
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    A_rc = P @ A @ P
    Ainv = np.linalg.inv(A_rc)
    C, h, w = patch.shape
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    offset = center - Ainv @ center
    out = np.zeros_like(patch, dtype=np.float32)
    for c in range(C):
        warped = affine_transform(patch[c].astype(np.float32), Ainv, offset=offset,
                                  order=1, mode="nearest")
        if c in _PRIOR_HEIGHT_CHANNELS:
            warped = warped * sz
        out[c] = warped.astype(np.float32)
    return out
