"""
Evaluation metrics - equations (16)-(21) of Zhu et al. (2025).

Binary ground extraction with class 0 = non-ground, class 1 = ground.  Following
the paper's notation:

    TP1 = points correctly predicted non-ground   (pred 0, true 0)
    TP2 = points correctly predicted ground        (pred 1, true 1)
    FP1 = points incorrectly predicted non-ground  (pred 0, true 1)   [paper "red"]
    FP2 = points incorrectly predicted ground       (pred 1, true 0)   [paper "black"]

    IoU1 = TP1 / (TP1 + FP1 + FP2)                         (16)
    IoU2 = TP2 / (TP2 + FP2 + FP1)                         (17)
    OA   = (TP1 + TP2) / (TP1 + FP1 + TP2 + FP2)           (18)
    Kappa = (OA - Pe) / (1 - Pe)                           (19)
    Pe   = [(TP1+FP2)(TP1+FP1) + (TP2+FP1)(TP2+FP2)] / N^2 (20)
    MCC  = (TP1*TP2 - FP1*FP2) /
           sqrt((TP1+FP1)(TP1+FP2)(TP2+FP1)(TP2+FP2))      (21)

Values are reported in percent (x100), as in the paper's tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class ConfusionAccumulator:
    tp1: int = 0   # pred non-ground, true non-ground
    tp2: int = 0   # pred ground,     true ground
    fp1: int = 0   # pred non-ground, true ground       (red)
    fp2: int = 0   # pred ground,     true non-ground    (black)

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        pred = np.asarray(pred).reshape(-1)
        true = np.asarray(true).reshape(-1)
        self.tp1 += int(np.sum((pred == 0) & (true == 0)))
        self.tp2 += int(np.sum((pred == 1) & (true == 1)))
        self.fp1 += int(np.sum((pred == 0) & (true == 1)))
        self.fp2 += int(np.sum((pred == 1) & (true == 0)))

    def compute(self) -> Dict[str, float]:
        # cast to float64: with a full-pass epoch the counts reach ~1e9, so the
        # MCC/Kappa products (up to ~1e36) overflow int64 / become Python big-ints
        # that np.sqrt cannot handle. float64 covers this with ample precision.
        tp1, tp2 = float(self.tp1), float(self.tp2)
        fp1, fp2 = float(self.fp1), float(self.fp2)
        n = tp1 + tp2 + fp1 + fp2
        eps = 1e-12

        iou1 = tp1 / max(tp1 + fp1 + fp2, eps)
        iou2 = tp2 / max(tp2 + fp2 + fp1, eps)
        oa = (tp1 + tp2) / max(n, eps)

        pe = ((tp1 + fp2) * (tp1 + fp1) + (tp2 + fp1) * (tp2 + fp2)) / max(n * n, eps)
        kappa = (oa - pe) / max(1.0 - pe, eps)

        denom = np.sqrt(
            max((tp1 + fp1), eps) * max((tp1 + fp2), eps)
            * max((tp2 + fp1), eps) * max((tp2 + fp2), eps)
        )
        mcc = (tp1 * tp2 - fp1 * fp2) / max(float(denom), eps)

        return {
            "IoU1": 100.0 * iou1,
            "IoU2": 100.0 * iou2,
            "OA": 100.0 * oa,
            "Kappa": 100.0 * kappa,
            "MCC": 100.0 * mcc,
            "mIoU": 100.0 * 0.5 * (iou1 + iou2),
            "n_points": int(n),
        }

    def reset(self) -> None:
        self.tp1 = self.tp2 = self.fp1 = self.fp2 = 0


# ---------------------------------------------------------------------------
# DTM RMSE  -  the OpenGF / SparseGF surface metric (Qin et al., 2021; 2026).
#
#   RMSE = sqrt( mean_i (P_i - R_i)^2 )
#
# over the valid pixels of the reference DTM, where P / R are the predicted /
# reference bare-earth rasters. Unlike OA / IoU (which weight every point
# equally), RMSE is dominated by a handful of tall points wrongly kept as
# ground - a spike lifts its cell by metres and squares - so it is the metric
# that actually tracks DTM quality and the "never admit a spike" objective.
# ---------------------------------------------------------------------------
def dtm_rmse_components(xyz: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                        res: float = 1.0):
    """Return ``(sse, n_pixels)`` for one cloud, following OpenGF's DTM-RMSE
    protocol (Qin et al. 2021, Sec 4.3): rasterise the GT-ground and the
    predicted-ground points to DEMs at ``res`` by **triangulation**
    (Delaunay-linear, scipy ``griddata`` -- the same triangulation OpenGF uses
    via ArcGIS "LAS Dataset To Raster"; natural-neighbour is a smoother variant
    not available in scipy and is the only residual difference), then sum the
    squared elevation difference over the cells where BOTH DEMs are defined
    (the overlap of the two triangulations -- the region with real ground
    support on both sides). Accumulate across clouds, then
    ``RMSE = sqrt(sum sse / sum n_pixels)``.
    """
    from scipy.interpolate import griddata
    pred = np.asarray(pred).reshape(-1); gt = np.asarray(gt).reshape(-1)
    g_pred = pred == 1; g_gt = gt == 1
    if int(g_gt.sum()) < 3 or int(g_pred.sum()) < 3:     # Delaunay needs >=3 pts
        return 0.0, 0
    x_min = float(xyz[:, 0].min()); y_min = float(xyz[:, 1].min())
    x_max = float(xyz[:, 0].max()); y_max = float(xyz[:, 1].max())
    W = max(int(np.ceil((x_max - x_min) / res)), 1)
    H = max(int(np.ceil((y_max - y_min) / res)), 1)
    # cell centres of the shared 0.5 m grid
    gx = x_min + (np.arange(W) + 0.5) * res
    gy = y_min + (np.arange(H) + 0.5) * res
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    try:
        pred_dem = griddata(xyz[g_pred, :2], xyz[g_pred, 2].astype(np.float64), pts, method="linear")
        gt_dem = griddata(xyz[g_gt, :2], xyz[g_gt, 2].astype(np.float64), pts, method="linear")
    except Exception:
        return 0.0, 0
    valid = np.isfinite(pred_dem) & np.isfinite(gt_dem)   # both triangulations cover the cell
    if not valid.any():
        return 0.0, 0
    d = pred_dem[valid] - gt_dem[valid]
    return float(np.sum(d * d)), int(valid.sum())


def dtm_error_pixels(xyz: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                     res: float = 1.0) -> np.ndarray:
    """Per-pixel ABSOLUTE DEM error |pred_dem - gt_dem| over the cells covered by BOTH
    triangulations -- the same grid/griddata as ``dtm_rmse_components``, but returning the
    raw residual array so a caller can aggregate it as RMSE (sqrt(mean(d^2))), a high
    percentile (P95/P99), or the max. Empty array when undefined. Tail aggregations weight
    the worst cells (cliffs/edges) far more than mean RMSE, countering flat-area dilution."""
    from scipy.interpolate import griddata
    pred = np.asarray(pred).reshape(-1); gt = np.asarray(gt).reshape(-1)
    g_pred = pred == 1; g_gt = gt == 1
    if int(g_gt.sum()) < 3 or int(g_pred.sum()) < 3:
        return np.empty(0, dtype=np.float64)
    x_min = float(xyz[:, 0].min()); y_min = float(xyz[:, 1].min())
    x_max = float(xyz[:, 0].max()); y_max = float(xyz[:, 1].max())
    W = max(int(np.ceil((x_max - x_min) / res)), 1)
    H = max(int(np.ceil((y_max - y_min) / res)), 1)
    gx = x_min + (np.arange(W) + 0.5) * res
    gy = y_min + (np.arange(H) + 0.5) * res
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    try:
        pred_dem = griddata(xyz[g_pred, :2], xyz[g_pred, 2].astype(np.float64), pts, method="linear")
        gt_dem = griddata(xyz[g_gt, :2], xyz[g_gt, 2].astype(np.float64), pts, method="linear")
    except Exception:
        return np.empty(0, dtype=np.float64)
    valid = np.isfinite(pred_dem) & np.isfinite(gt_dem)
    if not valid.any():
        return np.empty(0, dtype=np.float64)
    return np.abs(pred_dem[valid] - gt_dem[valid])


@dataclass
class RMSEAccumulator:
    """Accumulate DTM-RMSE components across the validation clouds (OpenGF 0.5 m)."""
    sse: float = 0.0
    npix: int = 0
    res: float = 1.0

    def update(self, xyz: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> None:
        s, n = dtm_rmse_components(xyz, pred, gt, self.res)
        self.sse += s; self.npix += n

    def compute(self) -> float:
        return float(np.sqrt(self.sse / self.npix)) if self.npix > 0 else float("nan")

    def reset(self) -> None:
        self.sse = 0.0; self.npix = 0
