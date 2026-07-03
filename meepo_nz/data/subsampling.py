"""
Grid (voxel) subsampling - a NumPy port of the KPConv ``grid_subsampling`` op.

KPConv builds its multi-resolution hierarchy by voxelising the cloud at a
sequence of growing grid sizes and keeping, for every occupied voxel, the
*barycenter* of the points that fall inside it (Thomas et al., 2019).  We
reproduce exactly that behaviour:

  * points  -> barycenter (mean xyz) of the voxel,
  * features -> mean of the features of the voxel,
  * labels   -> majority vote inside the voxel.

The original op is a compiled C++ routine; this pure-NumPy version is robust on
a fresh machine (no build step) and is numerically equivalent for the sizes we
use.  An optional compiled backend can be dropped in later without touching the
call sites.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def grid_subsample(
    points: np.ndarray,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    sample_dl: float = 0.1,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Voxel-barycenter subsampling at grid size ``sample_dl`` (metres).

    Returns subsampled ``(points[, features][, labels])`` with the same set of
    optional outputs that were passed in.
    """
    if points.shape[0] == 0:
        return points, features, labels

    # voxel index of every point (shift by the min corner so indices are >= 0)
    origin = points.min(axis=0)
    inv = 1.0 / sample_dl
    vox = np.floor((points - origin) * inv).astype(np.int64)        # (N, 3)

    # a unique integer key per voxel
    keys, inverse = np.unique(vox, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_vox = keys.shape[0]

    counts = np.bincount(inverse, minlength=n_vox).astype(np.float64)

    # barycenter of points
    sub_points = np.zeros((n_vox, points.shape[1]), dtype=np.float64)
    for d in range(points.shape[1]):
        sub_points[:, d] = np.bincount(inverse, weights=points[:, d], minlength=n_vox)
    sub_points /= counts[:, None]
    sub_points = sub_points.astype(np.float32)

    sub_features = None
    if features is not None:
        sub_features = np.zeros((n_vox, features.shape[1]), dtype=np.float64)
        for d in range(features.shape[1]):
            sub_features[:, d] = np.bincount(inverse, weights=features[:, d], minlength=n_vox)
        sub_features /= counts[:, None]
        sub_features = sub_features.astype(np.float32)

    sub_labels = None
    if labels is not None:
        labels = labels.reshape(-1).astype(np.int64)
        n_class = int(labels.max()) + 1 if labels.size else 1
        # per-voxel class histogram via a flat bincount, then argmax
        flat = inverse * n_class + labels
        hist = np.bincount(flat, minlength=n_vox * n_class).reshape(n_vox, n_class)
        sub_labels = hist.argmax(axis=1).astype(np.int64)

    return sub_points, sub_features, sub_labels


def estimate_nominal_spacing(xyz: np.ndarray, max_pts: int = 120000,
                             floor: float = 0.10) -> float:
    """Robust estimate of a survey's *nominal* point spacing (metres).

    Uses the **median nearest-neighbour distance** - the actual local spacing,
    which is what ``first_subsampling_dl`` should match. This is robust to the
    things that bias a ``sqrt(area / n)`` density estimate: internal gaps (water,
    no-data, sparse swaths) inflate the bounding area without adding points, and
    coincident / duplicate points (swath overlap, multi-returns) would collapse a
    raw nearest-neighbour *minimum* toward zero - so we drop near-zero distances
    and take the median, which a few duplicates cannot move.

    For large clouds we evaluate on a **contiguous spatial window** (the points
    nearest the centroid) rather than a random subset: a random subset thins the
    cloud and inflates every nearest-neighbour distance by ``sqrt(n / n_sample)``,
    whereas a contiguous window preserves the true local density.
    """
    xyz = np.asarray(xyz)
    n = xyz.shape[0]
    if n < 2:
        return floor
    pts = xyz[:, :3]
    if n > max_pts:
        c = pts[:, :2].mean(axis=0)
        order = np.argsort(((pts[:, :2] - c) ** 2).sum(axis=1))   # contiguous disk
        pts = pts[order[:max_pts]]
    try:
        from scipy.spatial import cKDTree
        xy = pts[:, :2]                            # horizontal spacing (immune to
        d, _ = cKDTree(xy).query(xy, k=2)          # vegetation height / Z spread)
        nn = d[:, 1]
        nn = nn[nn > 1e-3]                          # drop exact-XY multi-returns/dups
        if nn.size:
            return float(max(float(np.median(nn)), floor))
    except Exception:
        pass
    # fallback (no scipy): density over the robust XY extent of the window
    xy = pts[:, :2]
    dx = float(np.percentile(xy[:, 0], 99) - np.percentile(xy[:, 0], 1))
    dy = float(np.percentile(xy[:, 1], 99) - np.percentile(xy[:, 1], 1))
    nps = (max(dx * dy, 1e-6) / max(pts.shape[0], 1)) ** 0.5
    return float(max(nps, floor))
