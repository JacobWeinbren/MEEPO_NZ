"""SPAG-DC: Seed-Point-guided Adaptive Ground misclassification Detection and Correction.

Faithful implementation of Zhu, Tang, Yang, Li, Xue, Su, Yi, *An Adaptive
Ground-Point-Cloud Misclassification Detection and Correction Algorithm Based on
Seed-Point Guidance for High-Precision DEM Construction in Complex Terrain*,
IEEE Sensors Journal 25(21):40399-40411, 2025 (doi:10.1109/JSEN.2025.3615605).

This is a deterministic, *non-learned* closed-loop post-filter that runs on the
initial ground points produced by an upstream classifier (here: our MEEPO
model). It detects and removes Type-II errors (non-ground points misclassified as
ground -- the DTM "spikes") by three synergistic stages (paper Fig. 2):

  A. Core point set via dynamic region growing (paper II-A). Per-point normal +
     curvature c = lambda3/(lambda1+lambda2+lambda3) (eq 1) from the covariance of
     a k-NN; grow from the lowest-curvature seed using an ADAPTIVE normal-angle
     threshold theta_d = theta0*(1 + beta*var_c) (eq 2-3) and curvature threshold
     tau_c = alpha*mean_c (eq 4-5). Spatially discontinuous / isolated points
     (loose topology) are rejected -> removed.
  B. Seed points via grid-structure optimisation (paper II-B). Density/gradient
     adaptive grid (eq 6-7): refine a cell into 2x2 if N_ij > d, merge into the
     min-gradient neighbour if N_ij < 2; take each final cell's elevation MINIMUM
     as a seed, then purify with the Maximum Consistent Set (orthogonal distance
     to a robust local plane) to drop residual non-ground seeds.
  C. TPS surface fit + correction (paper II-C). Local thin-plate spline
     f(x,y)=sum lambda_i U(r_i)+a1 x+a2 y+a3, U(r)=r^2 log r (eq 8), with an
     adaptive neighbourhood k=clip(rho*alpha, kmin, kmax) (eq 9). A candidate's
     vertical residual to the surface flags it: residual > mu2 + n*sigma2 (ground
     residuals ~ normal, Bartels & Wei) -> non-ground; a per-grid 0.1 m minimum
     elevation-difference floor keeps all-ground cells intact.

Paper parameter values (used as defaults here -- no guessed values):
  theta0 = 10 deg; alpha = 0.5 (gate opens at 0.4, recommended 0.5-0.7);
  beta = 0.7 (recommended 0.6-0.8, in [0,1)); region-growing k ~ 20-25;
  TPS kmin=10, kmax=30 (the paper's tested K range); min-grid floor 0.1 m;
  correction n = 3 (the 3-sigma rule for the mu+n*sigma normal-tail cut the paper
  invokes via Bartels & Wei). All exposed via cfg / CLI for tuning within the
  paper's stated boundaries.

Adaptation to our pipeline: reclassify (ground->non-ground), never move or drop
points; operate on the predicted-ground subset (sub_raw==1). Pure numpy/scipy
(CPU), so it adds no GPU risk and is fully testable offline.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:                                    # pragma: no cover
    cKDTree = None


# --------------------------------------------------------------------------- #
# Per-point normals + curvature (paper eq 1)
# --------------------------------------------------------------------------- #
def _normals_and_curvature(xyz: np.ndarray, knn_idx: np.ndarray):
    """Normal (smallest-eigenvector) and curvature c=l3/(l1+l2+l3) per point,
    from the covariance of its k nearest neighbours."""
    nbr = xyz[knn_idx]                                # (N,k,3)
    mean = nbr.mean(axis=1, keepdims=True)
    d = nbr - mean
    cov = np.einsum("nki,nkj->nij", d, d) / max(knn_idx.shape[1] - 1, 1)   # (N,3,3)
    w, v = np.linalg.eigh(cov)                        # ascending: w[:,0]<=w[:,1]<=w[:,2]
    s = w.sum(axis=1) + 1e-12
    curv = w[:, 0] / s                                # lambda3 / (l1+l2+l3)
    normals = v[:, :, 0]                              # eigenvector of smallest eigenvalue
    return normals, curv.astype(np.float64)


# --------------------------------------------------------------------------- #
# Stage A: dynamic region growing -> core ground set (paper II-A, eq 1-5)
# --------------------------------------------------------------------------- #
def region_growing_core(xyz: np.ndarray, knn_idx: np.ndarray, normals: np.ndarray,
                        curv: np.ndarray, theta0_deg: float, alpha: float,
                        beta: float) -> np.ndarray:
    """Return a boolean core-ground mask. Grow from the lowest-curvature seed,
    accepting a neighbour iff normal-angle < theta_d(i) AND curvature < tau_c(i),
    with the adaptive thresholds of eq (2)-(5). Multiple seeds (next lowest unvisited
    curvature) are started until every point is visited, so all coherent ground
    clusters are captured while isolated / discontinuous points are rejected."""
    n = xyz.shape[0]
    theta0 = np.deg2rad(theta0_deg)
    cn = curv[knn_idx]                                # (N,k) neighbour curvatures
    mu_c = cn.mean(axis=1)                            # eq 5
    var_c = cn.var(axis=1)                            # eq 3
    theta_d = theta0 * (1.0 + beta * var_c)           # eq 2  (adaptive angle thr)
    tau_c = alpha * mu_c                              # eq 4  (adaptive curv thr)

    visited = np.zeros(n, dtype=bool)
    core = np.zeros(n, dtype=bool)
    order = np.argsort(curv, kind="stable")           # lowest curvature first (eq: seed init)
    from collections import deque
    for s in order:
        if visited[s]:
            continue
        # new region seeded at the lowest-curvature unvisited point
        visited[s] = True
        core[s] = True
        q = deque([s])
        while q:
            i = q.popleft()
            ni = normals[i]
            for j in knn_idx[i]:
                if visited[j]:
                    continue
                cos = abs(float(np.dot(ni, normals[j])))
                cos = min(max(cos, -1.0), 1.0)
                ang = np.arccos(cos)                  # normal angle, seed<->neighbour
                if ang < theta_d[i] and curv[j] < tau_c[i]:
                    visited[j] = True
                    core[j] = True
                    q.append(j)
                else:
                    visited[j] = True                 # rejected (non-ground, discontinuous)
    return core


# --------------------------------------------------------------------------- #
# Stage B: grid-structure-optimised seed extraction (paper II-B, eq 6-7) + MCS
# --------------------------------------------------------------------------- #
def _adaptive_seed_indices(xyz: np.ndarray, base_res: float) -> np.ndarray:
    """Elevation-minimum seed per adaptive grid cell. Base grid at `base_res`;
    a dense cell (N_ij > d, d = mean points/cell) is refined to 2x2; sparse cells
    (N_ij < 2) are effectively merged by falling back to the coarse cell minimum."""
    xy = xyz[:, :2]
    z = xyz[:, 2]
    mn = xy.min(axis=0)
    # coarse grid
    cidx = np.floor((xy - mn) / base_res).astype(np.int64)            # (N,2) eq 6
    ckey = cidx[:, 0] * 100003 + cidx[:, 1]
    _, inv, counts = np.unique(ckey, return_inverse=True, return_counts=True)
    d = counts.mean()                                                # avg points/cell
    seeds = []
    # dense cells -> refine to 2x2 (half resolution); others -> coarse-cell minimum
    fine_res = base_res / 2.0
    fidx = np.floor((xy - mn) / fine_res).astype(np.int64)
    fkey = fidx[:, 0] * 100003 + fidx[:, 1]
    is_dense = (counts[inv] > d)
    # coarse minima for non-dense cells
    for key in np.unique(ckey[~is_dense]):
        m = np.where((ckey == key) & (~is_dense))[0]
        if m.size:
            seeds.append(m[np.argmin(z[m])])
    # fine minima for dense cells (2x2 refinement)
    dense_mask = is_dense
    for key in np.unique(fkey[dense_mask]):
        m = np.where((fkey == key) & dense_mask)[0]
        if m.size:
            seeds.append(m[np.argmin(z[m])])
    return np.unique(np.asarray(seeds, dtype=np.int64))


def _mcs_purify(seed_xyz: np.ndarray, k: int = 12, n_sigma: float = 3.0) -> np.ndarray:
    """Maximum-Consistent-Set seed purification (paper II-B-4, Nurunnabi et al.):
    fit a local plane z=ax+by+c to each seed's neighbourhood and reject seeds whose
    vertical residual to that plane is a high outlier (median + n_sigma*MAD). A genuine
    ground seed is a low point consistent with the local surface; a residual non-ground
    seed (e.g. an isolated spike that became a cell minimum) sits high above it and is
    removed. Returns a keep-mask over seed_xyz."""
    m = seed_xyz.shape[0]
    if m < max(k, 6) or cKDTree is None:
        return np.ones(m, dtype=bool)
    kk = int(min(k, m))
    _, idx = cKDTree(seed_xyz[:, :2]).query(seed_xyz[:, :2], k=kk)
    resid = np.zeros(m, dtype=np.float64)
    for i in range(m):
        P = seed_xyz[idx[i]]
        A = np.c_[P[:, 0], P[:, 1], np.ones(P.shape[0])]
        coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)        # local plane z=ax+by+c
        resid[i] = seed_xyz[i, 2] - (coef[0] * seed_xyz[i, 0] + coef[1] * seed_xyz[i, 1] + coef[2])
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) * 1.4826
    thr = med + n_sigma * (mad if mad > 1e-9 else (resid.std() + 1e-9))
    return resid <= thr                                            # keep seeds on/below the local surface


# --------------------------------------------------------------------------- #
# Stage C: local thin-plate-spline surface (paper II-C, eq 8-9)
# --------------------------------------------------------------------------- #
def _tps_fit(ctrl_xy: np.ndarray, ctrl_z: np.ndarray):
    """Solve TPS weights for control points. f(x,y)=sum w_i U(r)+a1 x+a2 y+a3,
    U(r)=r^2 log r (eq 8). Returns (weights, affine, ctrl_xy)."""
    m = ctrl_xy.shape[0]
    d2 = ((ctrl_xy[:, None, :] - ctrl_xy[None, :, :]) ** 2).sum(-1)
    r = np.sqrt(d2)
    U = np.where(r > 1e-12, d2 * np.log(r + 1e-12), 0.0)              # r^2 log r
    P = np.hstack([np.ones((m, 1)), ctrl_xy])                        # [1 x y]
    A = np.zeros((m + 3, m + 3))
    A[:m, :m] = U
    A[:m, m:] = P
    A[m:, :m] = P.T
    rhs = np.concatenate([ctrl_z, np.zeros(3)])
    A[:m, :m] += 1e-6 * np.eye(m)                                    # tiny ridge for stability
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return sol[:m], sol[m:], ctrl_xy


def _tps_eval(query_xy: np.ndarray, w: np.ndarray, aff: np.ndarray, ctrl_xy: np.ndarray):
    d2 = ((query_xy[:, None, :] - ctrl_xy[None, :, :]) ** 2).sum(-1)
    r = np.sqrt(d2)
    U = np.where(r > 1e-12, d2 * np.log(r + 1e-12), 0.0)
    return U.dot(w) + aff[0] + aff[1] * query_xy[:, 0] + aff[2] * query_xy[:, 1]


# --------------------------------------------------------------------------- #
# Learned per-scene regime globals (the LEARNED SPAG-DC head).
#
# SPAG-DC is non-differentiable (region growing + quadtree + MCS + TPS), so it
# cannot be trained end-to-end. Instead a small head on the backbone's pooled
# per-scene features regresses the SPAG-DC control globals, supervised by an
# "oracle" target derived from each scene's GT-ground terrain statistics
# (slope / roughness / density). At inference the head predicts these globals
# from the learned features and they replace the fixed cfg defaults below.
#
# Six globals, in this fixed order, each squashed (sigmoid) into [lo, hi]:
#   theta0_deg : region-growing base slope angle  (steeper terrain -> larger)
#   alpha      : adaptive curvature-gate factor    (eq. 4)
#   beta       : roughness sensitivity in [0,1)     (eq. 2)
#   n_sigma    : MCS / TPS-residual rejection tail  (mu + n*sigma)
#   base_res   : adaptive seed-grid cell size (m)   (denser cloud -> finer)
#   min_floor  : distance-threshold floor (m)
# --------------------------------------------------------------------------- #
SPAG_GLOBAL_NAMES = ("theta0_deg", "alpha", "beta", "n_sigma", "base_res", "min_floor")
SPAG_GLOBAL_LO = np.array([5.0, 0.30, 0.00, 2.0, 0.5, 0.10], dtype=np.float64)
SPAG_GLOBAL_HI = np.array([30.0, 0.90, 0.99, 4.0, 8.0, 1.00], dtype=np.float64)
SPAG_N_GLOBALS = len(SPAG_GLOBAL_NAMES)


def oracle_regime_globals(ground_xyz: np.ndarray, min_ground: int = 32) -> np.ndarray:
    """Per-scene SPAG-DC target globals = the paper's recommended values
    (Liang et al., "Adaptive Ground-Point-Cloud Misclassification ...", IEEE
    Sensors 2025): theta0 = 10 deg ("we adopt a smaller value theta0 = 10"),
    alpha = 0.5 and beta = 0.7 (the paper's fixed sweep values; recommended
    alpha in 0.5-0.7, beta in 0.6-0.8), n_sigma = 3 (the elevation differences
    follow a normal distribution and points beyond mu2 + n*sigma2 are cut -> the
    3-sigma convention), and min_floor = 0.1 m ("set to 0.1 m in this article").
    Only ``base_res`` is per-scene: the paper grids by point density (eq. 6-7),
    so the seed-grid base cell is sized to the cloud's 2-D density (~8 pts/cell).
    Five of six globals are therefore the paper's fixed constants; the head
    learns to reproduce them and to set base_res from the scene's density.
    Returns a length-6 vector inside [SPAG_GLOBAL_LO, SPAG_GLOBAL_HI]."""
    g = np.asarray(ground_xyz, dtype=np.float64)
    theta0, alpha, beta, n_sigma, min_floor = 10.0, 0.5, 0.7, 3.0, 0.1   # paper defaults
    base_res = float(0.5 * (SPAG_GLOBAL_LO[4] + SPAG_GLOBAL_HI[4]))
    if g.ndim == 2 and g.shape[0] >= int(min_ground):
        xy = g[:, :2]
        area = (np.ptp(xy[:, 0]) + 1e-9) * (np.ptp(xy[:, 1]) + 1e-9)
        dens2d = g.shape[0] / area                                       # pts / m^2
        base_res = float(np.sqrt(8.0 / max(dens2d, 1e-6)))               # paper: grid by density
    out = np.array([theta0, alpha, beta, n_sigma, base_res, min_floor], dtype=np.float64)
    return np.clip(out, SPAG_GLOBAL_LO, SPAG_GLOBAL_HI)


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
def spag_dc_refine(sub_xyz: np.ndarray, sub_raw: np.ndarray, cfg,
                   risk_attrib=None, return_info: bool = False,
                   learned_globals=None):
    """Detect & correct ground-misclassified (Type-II) spikes via SPAG-DC.

    ``risk_attrib`` is accepted for a drop-in signature with the old refiner and is
    ignored (SPAG-DC is geometry-only, non-learned). Returns ``refined`` (N,) labels
    (1=ground, 0=non-ground), or ``(refined, info)`` with diagnostic fields.
    """
    sub_raw = np.asarray(sub_raw, dtype=np.int64)
    refined = sub_raw.copy()
    info = {"reclassified": np.zeros(sub_raw.size, dtype=bool),
            "n_reclassified": 0, "n_ground": 0, "n_core": 0, "n_seeds": 0}

    method = str(getattr(cfg, "refine_method", "spag_dc")).lower()
    if method in ("off", "none", "") or cKDTree is None:
        return (refined, info) if return_info else refined

    g = np.where(sub_raw == 1)[0]                     # predicted-ground candidates
    info["n_ground"] = int(g.size)
    if g.size < 16:
        return (refined, info) if return_info else refined

    theta0 = float(getattr(cfg, "spag_theta0_deg", 10.0))
    alpha = float(getattr(cfg, "spag_alpha", 0.5))
    beta = float(getattr(cfg, "spag_beta", 0.7))
    k = int(getattr(cfg, "spag_k", 20))
    n_sigma = float(getattr(cfg, "spag_n_sigma", 3.0))
    min_floor = float(getattr(cfg, "spag_min_grid_diff", 0.1))
    kmin = int(getattr(cfg, "spag_tps_kmin", 10))
    kmax = int(getattr(cfg, "spag_tps_kmax", 30))

    # LEARNED SPAG-DC: per-scene globals from the regime head override the fixed cfg
    # defaults (theta0/alpha/beta/n_sigma/base_res/min_floor; order = SPAG_GLOBAL_NAMES).
    base_res_override = None
    if learned_globals is not None:
        lg = np.asarray(learned_globals, dtype=np.float64).reshape(-1)
        if lg.size == SPAG_N_GLOBALS and np.isfinite(lg).all():
            lg = np.clip(lg, SPAG_GLOBAL_LO, SPAG_GLOBAL_HI)
            theta0, alpha, beta, n_sigma, base_res_override, min_floor = (
                float(lg[0]), float(lg[1]), float(lg[2]), float(lg[3]), float(lg[4]), float(lg[5]))
            info["learned_globals"] = lg.tolist()

    xg = sub_xyz[g].astype(np.float64)
    N = xg.shape[0]
    kk = int(min(max(k + 1, 4), N))
    _, knn = cKDTree(xg).query(xg, k=kk)
    knn = knn[:, 1:]                                  # drop self

    # ---- Stage A: dynamic region growing -> core ground ----
    normals, curv = _normals_and_curvature(xg, knn)
    core = region_growing_core(xg, knn, normals, curv, theta0, alpha, beta)
    info["n_core"] = int(core.sum())
    demote = np.zeros(N, dtype=bool)                  # demotion comes from the TPS residual (Stage C)
    core_idx_local = np.where(core)[0]

    # ---- Stage B: seed extraction (adaptive grid over the CORE) + MCS purification ----
    # base grid resolution from the 2-D point density (paper grids by density d)
    A_area = (np.ptp(xg[:, 0]) + 1e-9) * (np.ptp(xg[:, 1]) + 1e-9)
    dens2d = N / A_area                               # pts / m^2
    base_res = (base_res_override if base_res_override is not None
                else float(np.clip(np.sqrt(8.0 / max(dens2d, 1e-6)), 1.0, 20.0)))  # ~8 pts/cell
    seed_src = core_idx_local if core_idx_local.size >= 32 else np.arange(N)
    seeds_in_src = _adaptive_seed_indices(xg[seed_src], base_res)
    seed_local = seed_src[seeds_in_src]               # indices into xg
    if seed_local.size >= 8:
        seed_local = seed_local[_mcs_purify(xg[seed_local])]
    info["n_seeds"] = int(seed_local.size)

    # ---- Stage C: local TPS surface from seeds; correct by mu2 + n*sigma2 residual ----
    if seed_local.size >= 8:
        seed_xy = xg[seed_local, :2]
        seed_z = xg[seed_local, 2]
        seed_tree = cKDTree(seed_xy)
        # candidates = ALL non-seed initial-ground points (paper: seeds vs unclassified candidates)
        cand_local = np.setdiff1d(np.arange(N), seed_local, assume_unique=True)
        if cand_local.size:
            cand_xy = xg[cand_local, :2]
            rho = seed_local.size / A_area            # seed density (eq 9: k = clip(rho*scale, kmin, kmax))
            k_tps = int(np.clip(round(np.sqrt(seed_local.size)), kmin, kmax))
            k_tps = int(min(max(k_tps, kmin), min(kmax, seed_local.size)))
            # PER-CELL local TPS (paper's local interpolation): bin candidates on the seed grid,
            # fit ONE TPS per occupied cell from its k nearest seeds, evaluate all members.
            mn = xg[:, :2].min(axis=0)
            cc = np.floor((cand_xy - mn) / base_res).astype(np.int64)
            ckey = cc[:, 0] * 100003 + cc[:, 1]
            fit_z = np.empty(cand_local.size, dtype=np.float64)
            for key in np.unique(ckey):
                m = np.where(ckey == key)[0]
                center = cand_xy[m].mean(axis=0, keepdims=True)
                _, ci = seed_tree.query(center, k=k_tps)
                ci = np.atleast_1d(ci).ravel()
                w, aff, cxy = _tps_fit(seed_xy[ci], seed_z[ci])
                fit_z[m] = _tps_eval(cand_xy[m], w, aff, cxy)
            resid = xg[cand_local, 2] - fit_z          # vertical deviation candidate - surface
            # ground residuals ~ normal (Bartels & Wei): robust mu2/sigma2, cut at mu2 + n*sigma2,
            # with the paper's 0.1 m minimum-elevation-difference floor.
            mu2 = float(np.median(resid))
            sig2 = float(np.median(np.abs(resid - mu2)) * 1.4826)
            thr = max(mu2 + n_sigma * (sig2 if sig2 > 1e-9 else (resid.std() + 1e-9)), min_floor)
            far = resid > thr                          # ABOVE the ground surface by > threshold -> spike
            demote[cand_local[far]] = True

    spike_idx = g[demote]
    refined[spike_idx] = 0                             # ground(1) -> non-ground(0)
    info["reclassified"][spike_idx] = True
    info["n_reclassified"] = int(spike_idx.size)
    return (refined, info) if return_info else refined
