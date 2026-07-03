"""
Per-epoch visualisations.

1. ``render_error_image`` reproduces the paper's qualitative error figures
   (Figures 5-11) as a **shaded-relief (hillshade) map of the extracted bare
   ground**, seen from above:

      * the background is the **extracted bare ground** (points predicted as
        ground) gridded into a Digital Elevation Model, coloured by elevation
        and lit with a hillshade for relief;
      * **black** points are ground misclassified as non-ground (FP1);
      * **red** points are non-ground misclassified as ground  (FP2).

   A top-down map is used deliberately: error points are drawn at their true
   planimetric position, so they always sit *on* the ground surface they
   belong to (a 3-D surface plot in Matplotlib has no real depth buffer, so
   points leak above/below the mesh).  An "Elevation" colour-bar, a metric
   scale-bar and a legend are drawn underneath, matching the paper's layout.

2. ``update_training_charts`` writes a multi-panel PNG that is refreshed **every
   epoch**: training / validation loss, the five evaluation metrics over epochs,
   and a per-epoch time / ETA panel.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


# terrain ramp approximating the paper's figures: teal -> cream -> tan/brown
PAPER_TERRAIN = LinearSegmentedColormap.from_list(
    "paper_terrain",
    ["#1f6f78", "#4fa3a0", "#bfe0c8", "#f4eecb", "#d8a866", "#9c5a2c"],
)


def _nice_round(x: float) -> float:
    """Largest 'nice' number (1/2/5 x 10^k) not exceeding x."""
    if x <= 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(x))
    for m in (5.0, 2.0, 1.0):
        if x >= m * mag:
            return m * mag
    return mag


def _draw_scale_bar(ax, x0, x1, y0, y1) -> None:
    """Draw a real map scale-bar (lower-left) with a white halo for legibility."""
    extent = x1 - x0
    bar = _nice_round(extent * 0.28)
    xs = x0 + 0.06 * extent
    yb = y0 + 0.07 * (y1 - y0)
    halo = [pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()]
    ax.plot([xs, xs + bar], [yb, yb], color="black", lw=2.6,
            solid_capstyle="butt", path_effects=halo, zorder=6)
    for xx in (xs, xs + bar):
        ax.plot([xx, xx], [yb, yb + 0.018 * (y1 - y0)], color="black", lw=2.6,
                path_effects=halo, zorder=6)
    label = f"{bar:.0f} m" if bar >= 1 else f"{bar:g} m"
    ax.text(xs + bar / 2.0, yb + 0.03 * (y1 - y0), label, ha="center", va="bottom",
            fontsize=7.5, zorder=6,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


def _draw_error_map(ax, xyz, true_label, pred_label, point_size=1.2, grid_res=260,
                    scale_bar=True):
    """Draw the paper's Fig-6 error map onto ``ax``: a hillshaded bare-earth DEM of
    the predicted-ground points, with red = ground->non-ground and black =
    non-ground->ground. Returns (zmin, zmax) of the elevation ramp."""
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz)
    true_label = np.asarray(true_label).reshape(-1)
    pred_label = np.asarray(pred_label).reshape(-1)

    fp1 = (true_label == 1) & (pred_label == 0)         # black (ground -> non-ground)
    fp2 = (true_label == 0) & (pred_label == 1)         # red   (non-ground -> ground)
    pred_ground = (pred_label == 1)

    gp = xyz[pred_ground]
    if gp.shape[0] < 16:
        gp = xyz
    if gp.shape[0] > 40000:
        sel = np.random.default_rng(0).choice(gp.shape[0], 40000, replace=False)
        gp = gp[sel]

    zmin, zmax = (float(v) for v in np.percentile(gp[:, 2], [2.0, 98.0]))
    if zmax - zmin < 1e-6:
        zmin, zmax = float(gp[:, 2].min()), float(gp[:, 2].max())
    if zmax - zmin < 1e-6:
        zmax = zmin + 1.0
    x0, x1 = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    y0, y1 = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    if x1 - x0 < 1e-6:
        x1 = x0 + 1.0
    if y1 - y0 < 1e-6:
        y1 = y0 + 1.0

    if gp.shape[0] >= 16:
        gx = np.linspace(x0, x1, grid_res)
        gy = np.linspace(y0, y1, grid_res)
        GX, GY = np.meshgrid(gx, gy)
        lin = griddata(gp[:, :2], gp[:, 2], (GX, GY), method="linear")
        nea = griddata(gp[:, :2], gp[:, 2], (GX, GY), method="nearest")
        GZ = np.where(np.isnan(lin), nea, lin)
        cellx = (x1 - x0) / grid_res
        celly = (y1 - y0) / grid_res
        cell = max(cellx, celly, 1e-6)
        d, _ = cKDTree(gp[:, :2]).query(np.column_stack([GX.ravel(), GY.ravel()]), k=1)
        hole = (d.reshape(GX.shape) > max(0.5 * max(x1 - x0, y1 - y0), 8.0 * cell))
        ls = LightSource(azdeg=315, altdeg=45)
        ve = float(np.clip(0.10 * max(x1 - x0, y1 - y0) / max(zmax - zmin, 0.1), 1.0, 8.0))
        rgb = ls.shade(GZ, cmap=PAPER_TERRAIN, vmin=zmin, vmax=zmax,
                       blend_mode="soft", vert_exag=ve, dx=cellx, dy=celly)
        rgb[hole, 3] = 0.0
        ax.imshow(rgb, extent=[x0, x1, y0, y1], origin="lower",
                  interpolation="bilinear", zorder=1)
    else:
        ax.scatter(gp[:, 0], gp[:, 1], c=gp[:, 2], cmap=PAPER_TERRAIN,
                   vmin=zmin, vmax=zmax, s=point_size * 2.4, marker=".",
                   linewidths=0, zorder=1)

    ms = max(point_size, 1.0)
    if fp2.any():                                       # NG -> ground : RED (paper Fig 6)
        ax.scatter(xyz[fp2, 0], xyz[fp2, 1], c="#d62728", s=ms, marker="o",
                   linewidths=0, zorder=4)
    if fp1.any():                                       # ground -> NG : BLACK (paper Fig 6)
        ax.scatter(xyz[fp1, 0], xyz[fp1, 1], c="black", s=ms, marker="o",
                   linewidths=0, zorder=5)

    if scale_bar:
        _draw_scale_bar(ax, x0, x1, y0, y1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    return zmin, zmax


def render_error_image(
    xyz: np.ndarray,
    true_label: np.ndarray,
    pred_label: np.ndarray,
    out_path: str,
    title: str = "MEEPO",
    point_size: float = 1.2,
    grid_res: int = 260,
    **_ignore,
) -> None:
    """Render one error image in the paper's style (hillshaded bare-earth map)."""
    xyz = np.asarray(xyz)
    true_label = np.asarray(true_label).reshape(-1)
    pred_label = np.asarray(pred_label).reshape(-1)
    n = max(true_label.shape[0], 1)

    fp1 = (true_label == 1) & (pred_label == 0)
    fp2 = (true_label == 0) & (pred_label == 1)
    c_tp1 = int(((true_label == 0) & (pred_label == 0)).sum())
    c_tp2 = int(((true_label == 1) & (pred_label == 1)).sum())
    c_fp1, c_fp2 = int(fp1.sum()), int(fp2.sum())
    oa = 100.0 * (c_tp1 + c_tp2) / n
    iou2 = 100.0 * c_tp2 / max(c_tp2 + c_fp2 + c_fp1, 1)
    try:
        from .metrics import dtm_rmse_components
        _sse, _np = dtm_rmse_components(xyz, pred_label, true_label, res=1.0)
        rmse = float(np.sqrt(_sse / _np)) if _np > 0 else float("nan")
    except Exception:
        rmse = float("nan")

    fig = plt.figure(figsize=(6.8, 7.6))
    ax = fig.add_axes([0.06, 0.275, 0.88, 0.625])
    zmin, zmax = _draw_error_map(ax, xyz, true_label, pred_label,
                                 point_size=point_size, grid_res=grid_res)

    # title + per-image statistics line
    fig.text(0.5, 0.952, title, ha="center", fontsize=11, weight="bold")
    fig.text(0.5, 0.918,
             f"OA {oa:.1f}%     ground IoU {iou2:.1f}%     DTM RMSE {rmse:.3f} m     "
             f"FP1 (black, ground->NG) {c_fp1:,} ({100.0*c_fp1/n:.1f}%)     "
             f"FP2 (red, NG->ground) {c_fp2:,} ({100.0*c_fp2/n:.1f}%)",
             ha="center", fontsize=8.5, color="#333333")

    # elevation colour-bar
    sm = plt.cm.ScalarMappable(cmap=PAPER_TERRAIN, norm=Normalize(vmin=zmin, vmax=zmax))
    sm.set_array([])
    cax = fig.add_axes([0.20, 0.165, 0.60, 0.020])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Elevation (m)", fontsize=8, labelpad=2)

    # legend
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#7fb6a6",
               markersize=8, label="Extracted bare ground (hillshade, by elevation)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=6, label="Black: ground misclassified as non-ground (FP1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=6, label="Red: non-ground misclassified as ground (FP2)"),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=7.5,
               ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.01))

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def update_training_charts(history: List[Dict], out_path: str,
                           eta_text: Optional[str] = None) -> None:
    """Refresh the training dashboard PNG (called every epoch)."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    fig.suptitle(f"MEEPO training - through epoch {epochs[-1]}",
                 fontsize=12, weight="bold")

    def _style(ax):
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(x=0.05)

    # --- loss ---
    ax = axes[0]
    tr = [h.get("train_loss", np.nan) for h in history]
    va = [h.get("val_loss", np.nan) for h in history]
    ax.plot(epochs, tr, "-o", ms=4, color="#1f77b4", label="train loss")
    if any(not np.isnan(v) for v in va):
        ax.plot(epochs, va, "-s", ms=4, color="#d8a866", label="val loss")
    if np.isfinite(tr[-1]):
        ax.annotate(f"{tr[-1]:.3f}", (epochs[-1], tr[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color="#1f77b4")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Loss")
    _style(ax); ax.legend(fontsize=8, loc="upper right")

    # --- metrics ---
    ax = axes[1]
    palette = {"IoU1": "#1f77b4", "IoU2": "#ff7f0e", "OA": "#2ca02c",
               "Kappa": "#d62728", "MCC": "#9467bd"}
    markers = {"IoU1": "o", "IoU2": "s", "OA": "^", "Kappa": "d", "MCC": "v"}
    for key in ("IoU1", "IoU2", "OA", "Kappa", "MCC"):
        vals = [h.get(key, np.nan) for h in history]
        if any(not np.isnan(v) for v in vals):
            ax.plot(epochs, vals, marker=markers[key], ms=4, lw=1.6,
                    color=palette[key], label=key)
    ax.set_xlabel("epoch"); ax.set_ylabel("%"); ax.set_title("Validation metrics")
    ax.set_ylim(0, 100)
    _style(ax); ax.legend(fontsize=8, ncol=2, loc="lower right")

    # --- DTM RMSE (metres) - the surface metric; lower is better ---
    ax = axes[2]
    rm = [h.get("RMSE", np.nan) for h in history]
    if any(np.isfinite(v) for v in rm):
        ax.plot(epochs, rm, "-o", ms=4, color="#b5651d", label="DTM RMSE")
        best = np.nanmin(rm)
        bi = int(np.nanargmin(rm))
        ax.axhline(best, color="#4fa3a0", ls="--", lw=1, label=f"best {best:.3f} m")
        ax.annotate(f"{rm[-1]:.3f} m", (epochs[-1], rm[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color="#b5651d")
        ax.scatter([epochs[bi]], [best], s=40, color="#4fa3a0", zorder=5)
        top = np.nanpercentile(rm, 95)
        ax.set_ylim(0, max(top * 1.15, best * 1.5, 0.05))
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "RMSE pending", transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888")
    ax.set_xlabel("epoch"); ax.set_ylabel("metres"); ax.set_title("DTM RMSE (surface)")
    _style(ax)

    # --- timing / ETA (line, not a bar: reads sensibly from epoch 1) ---
    ax = axes[3]
    et = [h.get("epoch_time_s", np.nan) for h in history]
    et_min = [v / 60.0 if np.isfinite(v) else np.nan for v in et]
    ax.plot(epochs, et_min, "-o", ms=4, color="#4fa3a0")
    if any(np.isfinite(v) for v in et_min):
        mean_min = float(np.nanmean(et_min))
        ax.axhline(mean_min, color="#9c5a2c", ls="--", lw=1,
                   label=f"mean {mean_min:.1f} min")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_ylim(0, max(np.nanmax(et_min) * 1.25, 0.1))
    ax.set_xlabel("epoch"); ax.set_ylabel("minutes / epoch"); ax.set_title("Epoch time")
    _style(ax)
    if eta_text:
        ax.text(0.5, 0.93, eta_text, transform=ax.transAxes, ha="center", va="top",
                fontsize=9, bbox=dict(boxstyle="round", fc="#eef0f2", ec="#c9ccd1"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)



# ===========================================================================
# Per-epoch comparison panels — polished, paper-faithful design.
# Spatial panels are hillshade-RELIEF maps (elevation and each feature draped
# over the terrain relief, like the paper's shaded surfaces); classification
# panels show the bare-earth DEM with the non-ground / error overlay; profiles
# are filled side-on cross-sections (Fig 5 style). Cohesive palette + typography.
# ===========================================================================
from matplotlib.colors import ListedColormap as _ListedColormap

# refined topographic ramp (low teal -> green -> khaki -> tan -> brown)
PAPER_TERRAIN = LinearSegmentedColormap.from_list("paper_terrain", [
    "#1f4e5f", "#2e7d7d", "#5fa777", "#9fc06a", "#dbcd86", "#c79a5b", "#8f5a34",
])
# single elegant sequential map for all BCE feature panels (navy -> teal -> light)
FEATURE_CMAP = LinearSegmentedColormap.from_list("psct_feature", [
    "#0b1d36", "#143a5e", "#1f6f74", "#3fa67e", "#9ed487", "#f2f4c4",
])
GROUND_COLOR = "#d8a866"        # ground (warm tan)
NONGROUND_COLOR = "#2f7d3a"     # non-ground / vegetation (green)
FN_COLOR = "#101010"            # ground -> non-ground (Fig 6 black)
FP_COLOR = "#e02424"            # non-ground -> ground (Fig 6 red)
CORRECT_COLOR = "#b9c2cc"       # correctly classified (faint, profiles)
PAPER_BINARY = _ListedColormap([NONGROUND_COLOR, GROUND_COLOR])

_BG = "#ffffff"            # clean white background
_FRAME = "#c9ccd1"          # neutral panel frame
_INK = "#2b2b2b"
_SUBINK = "#5f5a52"
_RC = {
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": _FRAME, "axes.linewidth": 0.8,
    "axes.titlesize": 9.5, "axes.titlecolor": _INK,
    "axes.labelcolor": _SUBINK, "xtick.color": _SUBINK, "ytick.color": _SUBINK,
    "text.color": _INK,
}


def _extent(xyz):
    x0, x1 = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    y0, y1 = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    if x1 - x0 < 1e-6: x1 = x0 + 1.0
    if y1 - y0 < 1e-6: y1 = y0 + 1.0
    return x0, x1, y0, y1


def _zrange(z):
    zmin, zmax = (float(v) for v in np.percentile(z, [2.0, 98.0]))
    if zmax - zmin < 1e-6:
        zmin, zmax = float(z.min()), float(z.max())
    if zmax - zmin < 1e-6:
        zmax = zmin + 1.0
    return zmin, zmax


def _grid(xyz, values, x0, x1, y0, y1, res=210, hole_frac=0.5):
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    gx = np.linspace(x0, x1, res); gy = np.linspace(y0, y1, res)
    GX, GY = np.meshgrid(gx, gy)
    pts = xyz[:, :2]
    lin = griddata(pts, values, (GX, GY), method="linear")
    nea = griddata(pts, values, (GX, GY), method="nearest")
    GZ = np.where(np.isnan(lin), nea, lin)
    cellx = (x1 - x0) / res; celly = (y1 - y0) / res; cell = max(cellx, celly, 1e-6)
    d, _ = cKDTree(pts).query(np.column_stack([GX.ravel(), GY.ravel()]), k=1)
    hole = d.reshape(GX.shape) > max(hole_frac * max(x1 - x0, y1 - y0), 8.0 * cell)
    return GZ, hole, cellx, celly


def _relief_imshow(ax, value_grid, elev_grid, hole, cmap, vmin, vmax, dx, dy, ext, exag=None):
    """Drape ``value_grid`` (coloured by cmap) over the terrain relief of ``elev_grid``."""
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgb = cmap(norm(np.clip(value_grid, vmin, vmax)))[..., :3]
    ls = LightSource(azdeg=315, altdeg=45)
    zspan = max(float(np.nanmax(elev_grid) - np.nanmin(elev_grid)), 0.1)
    ve = exag if exag is not None else float(np.clip(
        0.12 * max(ext[1] - ext[0], ext[3] - ext[2]) / zspan, 1.0, 10.0))
    shaded = ls.shade_rgb(rgb, elev_grid, blend_mode="soft", vert_exag=ve, dx=dx, dy=dy)
    rgba = np.dstack([shaded, np.where(hole, 0.0, 1.0)])
    ax.imshow(rgba, extent=ext, origin="lower", interpolation="bilinear", zorder=1)


def _frame_map(ax, title):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(_FRAME); sp.set_linewidth(0.9)
    ax.set_title(title, fontsize=9.5, pad=5, color=_INK)
    ax.set_aspect("equal")


def _slim_cbar(fig, ax, mappable, label):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6, color=_FRAME)
    cb.outline.set_edgecolor(_FRAME); cb.outline.set_linewidth(0.6)
    if label:
        cb.set_label(label, fontsize=7, color=_SUBINK)
    return cb


def _scale_bar_soft(ax, x0, x1, y0, y1):
    _draw_scale_bar(ax, x0, x1, y0, y1)


def _ground_line(xr, z, nbins=80, pct=30.0):
    """Binned ground-surface line of ground points (low percentile per bin so a
    few high mislabeled points don't spike the silhouette; gaps interpolated)."""
    if xr.size < 4:
        return None, None
    edges = np.linspace(xr.min(), xr.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(xr, edges) - 1, 0, nbins - 1)
    line = np.full(nbins, np.nan)
    for b in range(nbins):
        zb = z[idx == b]
        if zb.size:
            line[b] = np.percentile(zb, pct)
    ok = ~np.isnan(line)
    if ok.sum() < 2:
        return None, None
    line = np.interp(centers, centers[ok], line[ok])
    return centers, line


def _style_profile(ax, title):
    ax.set_title(title, fontsize=9.5, pad=5, color=_INK)
    ax.set_xlabel("horizontal distance (m)", fontsize=7.5)
    ax.set_ylabel("elevation (m)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, color="#e8eaed", linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_edgecolor(_FRAME)
    ax.set_facecolor("#ffffff")
    ax.margins(x=0.01)


def _profile(ax, xyz, labels, mode, title, band_frac=0.06, point_size=6.0):
    labels = np.asarray(labels).reshape(-1)
    x0, x1, y0, y1 = _extent(xyz)
    yc = float(np.median(xyz[:, 1])); half = max(band_frac * (y1 - y0), 1.0)
    band = np.abs(xyz[:, 1] - yc) <= half
    if int(band.sum()) < 60:
        half = 0.15 * (y1 - y0); band = np.abs(xyz[:, 1] - yc) <= half
    s = xyz[band]; sl = labels[band]
    if s.shape[0] == 0:
        s, sl = xyz, labels
    xr = s[:, 0] - x0; z = s[:, 2]
    g = sl == 1
    zfloor = float(np.percentile(z, 0.5)) - 0.5
    cx, line = _ground_line(xr[g], z[g]) if g.any() else (None, None)
    if cx is not None:
        ax.fill_between(cx, zfloor, line, color=GROUND_COLOR, alpha=0.40, zorder=1,
                        linewidth=0)
        ax.plot(cx, line, color="#b07c3f", linewidth=1.0, zorder=2)
    ax.scatter(xr[~g], z[~g], c=NONGROUND_COLOR, s=point_size, marker="o",
               linewidths=0, alpha=0.85, zorder=4, label="non-ground")
    ax.scatter(xr[g], z[g], c=GROUND_COLOR, s=point_size * 0.7, marker="o",
               linewidths=0, alpha=0.9, zorder=3, label="ground")
    _style_profile(ax, title)
    ax.set_ylim(zfloor, None)
    leg = ax.legend(loc="upper right", fontsize=6.5, frameon=True, handletextpad=0.3,
                    facecolor="white", edgecolor=_FRAME)
    leg.get_frame().set_linewidth(0.6)


def _profile_error(ax, xyz, y_true, y_pred, title, band_frac=0.06, point_size=6.0):
    y_true = np.asarray(y_true).reshape(-1); y_pred = np.asarray(y_pred).reshape(-1)
    x0, x1, y0, y1 = _extent(xyz)
    yc = float(np.median(xyz[:, 1])); half = max(band_frac * (y1 - y0), 1.0)
    band = np.abs(xyz[:, 1] - yc) <= half
    if int(band.sum()) < 60:
        half = 0.15 * (y1 - y0); band = np.abs(xyz[:, 1] - yc) <= half
    s = xyz[band]; t = y_true[band]; p = y_pred[band]
    if s.shape[0] == 0:
        s, t, p = xyz, y_true, y_pred
    xr = s[:, 0] - x0; z = s[:, 2]
    zfloor = float(np.percentile(z, 0.5)) - 0.5
    gt = t == 1
    cx, line = _ground_line(xr[gt], z[gt]) if gt.any() else (None, None)
    if cx is not None:
        ax.fill_between(cx, zfloor, line, color="#d7d0c2", alpha=0.5, zorder=1, linewidth=0)
    correct = t == p; fn = (t == 1) & (p == 0); fp = (t == 0) & (p == 1)
    ax.scatter(xr[correct], z[correct], c=CORRECT_COLOR, s=point_size,
               marker="o", linewidths=0, alpha=0.85, zorder=2, label="correct")
    ax.scatter(xr[fn], z[fn], c=FN_COLOR, s=point_size, marker="o", linewidths=0,
               zorder=4, label="ground->NG (FP1)")
    ax.scatter(xr[fp], z[fp], c=FP_COLOR, s=point_size, marker="o", linewidths=0,
               zorder=5, label="NG->ground (FP2)")
    _style_profile(ax, title)
    ax.set_ylim(zfloor, None)
    leg = ax.legend(loc="upper right", fontsize=6.2, frameon=True, handletextpad=0.3,
                    facecolor="white", edgecolor=_FRAME)
    leg.get_frame().set_linewidth(0.6)


def render_epoch_panels(
    xyz: np.ndarray,
    feats: Optional[np.ndarray],
    feat_names: Optional[List[str]],
    dtm_patch: Optional[np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str = "MEEPO",
    tile_size: Optional[float] = None,
    point_size: float = 4.0,
) -> None:
    """Composite per-epoch comparison figure for one input cylinder, paper-style.

    Three stacked bands: a tightly-packed grid of input channels, then a
    prominent ground-truth / prediction / errors row, then a ground-truth /
    prediction / errors cross-section row. The column count of the input band is
    chosen so it is never padded with empty cells, and the result-bearing maps
    are drawn larger than the input thumbnails.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = max(y_true.shape[0], 1)
    x0, x1, y0, y1 = _extent(xyz)
    ext = [x0, x1, y0, y1]
    res = 210

    with plt.rc_context(_RC):
        zmin, zmax = _zrange(xyz[:, 2])
        GZ_all, hole_all, dx, dy = _grid(xyz, xyz[:, 2], x0, x1, y0, y1, res=res, hole_frac=0.5)

        def ground_grid(mask):
            gp = xyz[mask] if int(mask.sum()) >= 16 else xyz
            if gp.shape[0] > 45000:
                gp = gp[np.random.default_rng(0).choice(gp.shape[0], 45000, replace=False)]
            return _grid(gp, gp[:, 2], x0, x1, y0, y1, res=res, hole_frac=0.5)

        gz_t, gh_t, _, _ = ground_grid(y_true == 1)
        gz_p, gh_p, _, _ = ground_grid(y_pred == 1)

        # ----- panel spec -----
        inputs = [("elev", None)]
        names = list(feat_names) if (feats is not None and feat_names) else []
        if feats is not None and feats.shape[0] == n and feats.shape[1] > 0:
            for j in range(min(feats.shape[1], len(names) or feats.shape[1])):
                inputs.append(("feat", j))
        has_dtm = (dtm_patch is not None and np.asarray(dtm_patch).size > 1
                   and float(np.ptp(np.asarray(dtm_patch))) > 1e-6)
        if has_dtm:
            inputs.append(("dtm", None))

        def draw(ax, kind, payload):
            ax.set_facecolor(_BG)
            if kind == "elev":
                _relief_imshow(ax, GZ_all, GZ_all, hole_all, PAPER_TERRAIN,
                               zmin, zmax, dx, dy, ext)
                _frame_map(ax, "Input · elevation (shaded relief)")
                sm = plt.cm.ScalarMappable(cmap=PAPER_TERRAIN, norm=Normalize(zmin, zmax))
                sm.set_array([]); _slim_cbar(fig, ax, sm, "Elevation (m)")
            elif kind == "feat":
                v = feats[:, payload].astype(np.float64)
                lo, hi = np.percentile(v, [2.0, 98.0])
                if hi - lo < 1e-9:
                    lo, hi = float(v.min()), float(v.max() + 1e-6)
                GZ_f, hole_f, _, _ = _grid(xyz, v, x0, x1, y0, y1, res=res, hole_frac=0.5)
                _relief_imshow(ax, GZ_f, GZ_all, hole_f, FEATURE_CMAP, lo, hi, dx, dy, ext)
                nm = names[payload] if payload < len(names) else f"feat {payload}"
                _frame_map(ax, f"Input · {nm}")
                sm = plt.cm.ScalarMappable(cmap=FEATURE_CMAP, norm=Normalize(lo, hi))
                sm.set_array([]); _slim_cbar(fig, ax, sm, nm)
            elif kind == "dtm":
                patch = np.asarray(dtm_patch, dtype=np.float64)
                pe = [0, tile_size or patch.shape[1], 0, tile_size or patch.shape[0]]
                pz0, pz1 = _zrange(patch.ravel())
                GP, ph, pdx, pdy = (patch, np.zeros_like(patch, bool),
                                    pe[1] / patch.shape[1], pe[3] / patch.shape[0])
                _relief_imshow(ax, GP, GP, ph, PAPER_TERRAIN, pz0, pz1, pdx, pdy, pe)
                _frame_map(ax, "Input · prev-year DTM (relief)")
                sm = plt.cm.ScalarMappable(cmap=PAPER_TERRAIN, norm=Normalize(pz0, pz1))
                sm.set_array([]); _slim_cbar(fig, ax, sm, "Elev (m)")
            elif kind in ("gt", "pred"):
                gz, gh = (gz_t, gh_t) if kind == "gt" else (gz_p, gh_p)
                _relief_imshow(ax, gz, gz, gh, PAPER_TERRAIN, zmin, zmax, dx, dy, ext)
                lab = y_true if kind == "gt" else y_pred
                ng = xyz[lab == 0]
                if ng.shape[0]:
                    ax.scatter(ng[:, 0], ng[:, 1], c=NONGROUND_COLOR, s=point_size,
                               marker="o", linewidths=0, alpha=0.55, zorder=4)
                _frame_map(ax, "Ground truth" if kind == "gt" else "Prediction")
                _scale_bar_soft(ax, x0, x1, y0, y1)
                ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
                h = [Line2D([0], [0], marker="s", color="w", markerfacecolor=GROUND_COLOR,
                            markersize=8, label="ground (bare-earth)"),
                     Line2D([0], [0], marker="o", color="w", markerfacecolor=NONGROUND_COLOR,
                            markersize=7, label="non-ground")]
                leg = ax.legend(handles=h, loc="upper left", fontsize=6.4, frameon=True,
                                facecolor="white", edgecolor=_FRAME)
                leg.get_frame().set_linewidth(0.6)
            elif kind == "err":
                _relief_imshow(ax, gz_t, gz_t, gh_t, PAPER_TERRAIN, zmin, zmax, dx, dy, ext)
                fp1 = (y_true == 1) & (y_pred == 0)   # ground -> NG : black
                fp2 = (y_true == 0) & (y_pred == 1)   # NG -> ground : red
                ms = max(point_size * 0.9, 2.0)
                if fp2.any():
                    ax.scatter(xyz[fp2, 0], xyz[fp2, 1], c=FP_COLOR, s=ms, marker="o",
                               linewidths=0, zorder=4)
                if fp1.any():
                    ax.scatter(xyz[fp1, 0], xyz[fp1, 1], c=FN_COLOR, s=ms, marker="o",
                               linewidths=0, zorder=5)
                _frame_map(ax, "Errors (paper Fig. 6 style)")
                _scale_bar_soft(ax, x0, x1, y0, y1)
                ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
                h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=FN_COLOR,
                            markersize=7, label="ground->NG (FP1)"),
                     Line2D([0], [0], marker="o", color="w", markerfacecolor=FP_COLOR,
                            markersize=7, label="NG->ground (FP2)")]
                leg = ax.legend(handles=h, loc="upper left", fontsize=6.4, frameon=True,
                                facecolor="white", edgecolor=_FRAME)
                leg.get_frame().set_linewidth(0.6)
            elif kind == "pgt":
                _profile(ax, xyz, y_true, "truth", "Profile · ground truth", point_size=point_size + 1)
            elif kind == "ppred":
                _profile(ax, xyz, y_pred, "pred", "Profile · prediction", point_size=point_size + 1)
            elif kind == "perr":
                _profile_error(ax, xyz, y_true, y_pred, "Profile · errors", point_size=point_size + 1)

        def pick_cols(num, lo=3, hi=6):
            best = None
            for c in range(lo, hi + 1):
                rows = int(np.ceil(num / c)); empty = rows * c - num
                key = (empty, rows)
                if best is None or key < best[0]:
                    best = (key, c, rows)
            return best[1], best[2]

        in_cols, in_rows = pick_cols(len(inputs))
        W = 16.0
        in_pw = W / in_cols
        in_h = in_rows * (in_pw + 0.30)
        cls_h = (W / 3.0) + 0.55                      # square-ish result maps
        prof_h = 3.25                                  # landscape cross-sections
        head_h = 0.95
        total_h = head_h + in_h + cls_h + prof_h
        fig = plt.figure(figsize=(W, total_h), facecolor=_BG)
        sub = fig.subfigures(4, 1, height_ratios=[head_h, in_h, cls_h, prof_h], hspace=0.0)
        for s in sub:
            s.set_facecolor(_BG)

        gi = sub[1].add_gridspec(in_rows, in_cols, hspace=0.34, wspace=0.16,
                                 top=0.95, bottom=0.03, left=0.045, right=0.965)
        for k, (kind, payload) in enumerate(inputs):
            ax = sub[1].add_subplot(gi[k // in_cols, k % in_cols]); draw(ax, kind, payload)

        gc = sub[2].add_gridspec(1, 3, wspace=0.13, top=0.91, bottom=0.05, left=0.045, right=0.965)
        for k, kind in enumerate(["gt", "pred", "err"]):
            ax = sub[2].add_subplot(gc[0, k]); draw(ax, kind, None)

        gp = sub[3].add_gridspec(1, 3, wspace=0.13, top=0.85, bottom=0.14, left=0.045, right=0.965)
        for k, kind in enumerate(["pgt", "ppred", "perr"]):
            ax = sub[3].add_subplot(gp[0, k]); draw(ax, kind, None)

        # ----- header -----
        tp1 = int(((y_true == 0) & (y_pred == 0)).sum())
        tp2 = int(((y_true == 1) & (y_pred == 1)).sum())
        fp1 = int(((y_true == 1) & (y_pred == 0)).sum())
        fp2 = int(((y_true == 0) & (y_pred == 1)).sum())
        oa = 100.0 * (tp1 + tp2) / n
        iou2 = 100.0 * tp2 / max(tp2 + fp1 + fp2, 1)
        try:
            from .metrics import dtm_rmse_components
            _sse, _np = dtm_rmse_components(np.asarray(xyz), np.asarray(y_pred), np.asarray(y_true), res=1.0)
            _rmse = float(np.sqrt(_sse / _np)) if _np > 0 else float("nan")
        except Exception:
            _rmse = float("nan")
        fig.text(0.045, 1 - 0.34 / total_h, title, ha="left", fontsize=14, weight="bold", color=_INK)
        fig.text(0.045, 1 - 0.62 / total_h,
                 f"OA {oa:.1f}%     ground IoU {iou2:.1f}%     DTM RMSE {_rmse:.3f} m     "
                 f"FP1 (ground\u2192NG) {fp1:,}     FP2 (NG\u2192ground) {fp2:,}     {n:,} points",
                 ha="left", fontsize=9.5, color=_SUBINK)
        fig.text(0.965, 1 - 0.34 / total_h, "INPUTS  \u00b7  CLASSIFICATION  \u00b7  PROFILES",
                 ha="right", fontsize=8.5, color="#9aa0a6", weight="bold")
        fig.savefig(out_path, dpi=150, facecolor=_BG)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Combined REVIEW panel: top-down + angled-3D + 2D profile in one figure, with
# a metrics header. Used for qualitative review of a tile (truth | pred | error)
# in a single PNG. Reuses the palette/helpers above so it matches the dashboard.
# ---------------------------------------------------------------------------
def _binary_metrics(y_true, y_pred) -> Dict[str, float]:
    """Ground = class 1. Returns the standard binary segmentation metrics (%)."""
    t = np.asarray(y_true).reshape(-1).astype(np.int64)
    p = np.asarray(y_pred).reshape(-1).astype(np.int64)
    _ev = (t == 0) | (t == 1)                # drop ignore-label (unclassified) points from scoring
    t, p = t[_ev], p[_ev]
    n = max(t.size, 1)
    tp = int(((t == 1) & (p == 1)).sum())   # ground correct
    tn = int(((t == 0) & (p == 0)).sum())   # non-ground correct
    fp = int(((t == 0) & (p == 1)).sum())   # NG -> ground
    fn = int(((t == 1) & (p == 0)).sum())   # ground -> NG
    iou_g = tp / max(tp + fp + fn, 1)
    iou_ng = tn / max(tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    oa = (tp + tn) / n
    # MCC + Cohen's kappa
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = (tp * tn - fp * fn) / denom
    po = oa
    pe = (((tp + fn) * (tp + fp)) + ((tn + fp) * (tn + fn))) / (n * n)
    kappa = (po - pe) / max(1 - pe, 1e-9)
    return {"OA": 100 * oa, "mIoU": 100 * 0.5 * (iou_g + iou_ng),
            "IoU_ground": 100 * iou_g, "IoU_nonground": 100 * iou_ng,
            "precision": 100 * prec, "recall": 100 * rec, "F1": 100 * f1,
            "MCC": mcc, "kappa": kappa, "FP1": fn, "FP2": fp, "n": int(t.size)}


def _topdown_panel(ax, xyz, y_true, y_pred, mode, vmin, vmax, point_size=1.5):
    """mode: 'truth' | 'pred' | 'error'. Pure point scatter (no gridded relief) so
    the three panels are directly comparable and point density reads honestly.

    truth/pred use IDENTICAL logic - ground coloured by elevation on the SHARED
    (vmin, vmax) range, non-ground green - so only the labels differ between them.
    The error panel draws the correctly-classified points as a prominent light-grey
    base and overlays the two error classes at the SAME point size: colour (not
    size) carries the signal, so a 70 %-correct tile reads as ~70 % grey.
    """
    x0, x1, y0, y1 = _extent(xyz)
    z = xyz[:, 2]
    if mode in ("truth", "pred"):
        lab = np.asarray(y_true if mode == "truth" else y_pred).reshape(-1)
        g = lab == 1
        # flat CLASS colours (ground tan, non-ground green) - identical logic for
        # truth and pred, so the two panels are a direct ground/non-ground comparison
        # (elevation colouring made the ground's green end blend into the non-ground).
        if (~g).any():
            ax.scatter(xyz[~g, 0], xyz[~g, 1], c=NONGROUND_COLOR, s=point_size,
                       marker=".", linewidths=0, alpha=0.85, zorder=3, rasterized=True)
        if g.any():
            ax.scatter(xyz[g, 0], xyz[g, 1], c=GROUND_COLOR, s=point_size,
                       marker=".", linewidths=0, alpha=0.9, zorder=4, rasterized=True)
    else:
        yt = np.asarray(y_true).reshape(-1); yp = np.asarray(y_pred).reshape(-1)
        cor = yt == yp; fn = (yt == 1) & (yp == 0); fp = (yt == 0) & (yp == 1)
        if cor.any():
            ax.scatter(xyz[cor, 0], xyz[cor, 1], c=CORRECT_COLOR, s=point_size,
                       marker=".", linewidths=0, alpha=0.8, zorder=3, rasterized=True)
        if fp.any():
            ax.scatter(xyz[fp, 0], xyz[fp, 1], c=FP_COLOR, s=point_size,
                       marker=".", linewidths=0, alpha=0.95, zorder=4, rasterized=True)
        if fn.any():
            ax.scatter(xyz[fn, 0], xyz[fn, 1], c=FN_COLOR, s=point_size,
                       marker=".", linewidths=0, alpha=0.95, zorder=5, rasterized=True)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    _draw_scale_bar(ax, x0, x1, y0, y1)
    _frame_map(ax, "")


def _angled_panel(ax, xyz, y_true, y_pred, mode, vmin, vmax, point_size=2.2):
    """3-D oblique view. Ground coloured by elevation on the SHARED (vmin, vmax)
    range, non-ground green.

    The elevation range is passed in (computed once over ALL points) rather than
    recomputed per panel from that panel's own ground set. That was the bug making
    the prediction look like a different scene: when the model over-predicts ground,
    the predicted-ground set spans a wider/higher elevation band, so a per-panel
    range re-mapped the same heights to different colours. With a shared range the
    same elevation is the same colour in every panel, so truth and prediction are
    directly comparable and the over-prediction shows up honestly (canopy heights
    coloured as 'ground')."""
    x0, x1, y0, y1 = _extent(xyz)
    z = xyz[:, 2]
    if mode in ("truth", "pred"):
        lab = np.asarray(y_true if mode == "truth" else y_pred).reshape(-1)
        g = lab == 1
        # flat CLASS colours (ground tan, non-ground green); the 3-D z-axis still
        # carries elevation, so the terrain shape reads while ground/non-ground stay
        # distinct. Identical logic for truth and pred -> direct comparison.
        if (~g).any():
            ax.scatter(xyz[~g, 0], xyz[~g, 1], z[~g], c=NONGROUND_COLOR,
                       s=point_size, linewidths=0, alpha=0.7, depthshade=False)
        if g.any():
            ax.scatter(xyz[g, 0], xyz[g, 1], z[g], c=GROUND_COLOR,
                       s=point_size, linewidths=0, alpha=0.9, depthshade=False)
    else:
        yt = np.asarray(y_true).reshape(-1); yp = np.asarray(y_pred).reshape(-1)
        cor = yt == yp; fn = (yt == 1) & (yp == 0); fp = (yt == 0) & (yp == 1)
        # correct as a prominent grey base, errors the SAME size on top
        if cor.any():
            ax.scatter(xyz[cor, 0], xyz[cor, 1], z[cor], c=CORRECT_COLOR,
                       s=point_size, linewidths=0, alpha=0.6, depthshade=False)
        if fp.any():
            ax.scatter(xyz[fp, 0], xyz[fp, 1], z[fp], c=FP_COLOR,
                       s=point_size, linewidths=0, alpha=0.95, depthshade=False)
        if fn.any():
            ax.scatter(xyz[fn, 0], xyz[fn, 1], z[fn], c=FN_COLOR,
                       s=point_size, linewidths=0, alpha=0.95, depthshade=False)
    ax.view_init(elev=24, azim=-58)
    try:
        ax.set_box_aspect((1, 1, 0.42))
    except Exception:
        pass
    span = max(x1 - x0, y1 - y0)
    ax.set_xlim(x0, x0 + span); ax.set_ylim(y0, y0 + span)
    zlo, zhi = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    ax.set_zlim(zlo - 0.3, zhi + 0.3)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0.0)); axis.pane.set_edgecolor(_FRAME)
        axis.line.set_color(_FRAME)
    ax.grid(True, color="#eceef1", linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis="z", labelsize=6, colors=_SUBINK, pad=1)
    ax.set_zlabel("elev (m)", fontsize=6.5, color=_SUBINK, labelpad=4, rotation=90)
    try:
        ax.set_box_aspect((1, 1, 0.42))
    except Exception:
        pass
    span = max(x1 - x0, y1 - y0)
    ax.set_xlim(x0, x0 + span); ax.set_ylim(y0, y0 + span)
    zlo, zhi = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    ax.set_zlim(zlo - 0.3, zhi + 0.3)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0.0)); axis.pane.set_edgecolor(_FRAME)
        axis.line.set_color(_FRAME)
    ax.grid(True, color="#eceef1", linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis="z", labelsize=6, colors=_SUBINK, pad=1)
    ax.set_zlabel("elev (m)", fontsize=6.5, color=_SUBINK, labelpad=4, rotation=90)


def render_review_panel(xyz, y_true, y_pred, out_path: str,
                        title: str = "MEEPO ground extraction \u2014 tile review",
                        subtitle: Optional[str] = None) -> Dict[str, float]:
    """One PNG: top-down / angled-3D / 2-D profile, each as truth | pred | error,
    with a metrics header. Returns the metrics dict."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    xyz = np.asarray(xyz, dtype=np.float64)
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64)
    m = _binary_metrics(y_true, y_pred)

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(13.6, 12.4), facecolor=_BG)
        gs = fig.add_gridspec(
            4, 3, height_ratios=[0.58, 3.05, 3.05, 2.5],
            hspace=0.16, wspace=0.08,
            left=0.045, right=0.965, top=0.908, bottom=0.085)

        # ---- header: title + metric cards ----
        fig.text(0.045, 0.987, title, ha="left", va="top", fontsize=16,
                 weight="bold", color=_INK)
        if subtitle:
            fig.text(0.045, 0.962, subtitle, ha="left", va="top", fontsize=8.5,
                     color="#9aa0a6")
        cards = [("OA", f"{m['OA']:.1f}%"), ("mIoU", f"{m['mIoU']:.1f}%"),
                 ("ground IoU", f"{m['IoU_ground']:.1f}%"),
                 ("non-ground IoU", f"{m['IoU_nonground']:.1f}%"),
                 ("ground F1", f"{m['F1']:.1f}%"), ("MCC", f"{m['MCC']:.3f}"),
                 ("\u03ba", f"{m['kappa']:.3f}")]
        hax = fig.add_subplot(gs[0, :]); hax.axis("off")
        nshow = len(cards)
        for i, (k, v) in enumerate(cards):
            cx = (i + 0.5) / nshow
            hax.text(cx, 0.56, v, ha="center", va="center", fontsize=15,
                     weight="bold", color="#1f6f74", transform=hax.transAxes)
            hax.text(cx, 0.08, k, ha="center", va="center", fontsize=8.5,
                     color=_SUBINK, transform=hax.transAxes)
        hax.axhline(-0.06, color=_FRAME, linewidth=0.8)

        col_titles = ["ground truth", "prediction", "errors"]
        # SHARED elevation range over ALL points (not per-panel): keeps the same
        # height the same colour across truth / prediction / errors so the three
        # panels are directly comparable.
        zvmin, zvmax = _zrange(xyz[:, 2])
        # ---- row 1: top-down ----
        for c in range(3):
            ax = fig.add_subplot(gs[1, c])
            _topdown_panel(ax, xyz, y_true, y_pred,
                           ("truth", "pred", "error")[c], zvmin, zvmax, point_size=1.5)
            ax.set_title(f"Top-down \u00b7 {col_titles[c]}", fontsize=10, color=_INK, pad=5)
        # ---- row 2: angled 3-D ----
        for c in range(3):
            ax = fig.add_subplot(gs[2, c], projection="3d")
            _angled_panel(ax, xyz, y_true, y_pred, ("truth", "pred", "error")[c],
                          zvmin, zvmax, point_size=2.2)
            ax.set_title(f"Angled 3-D \u00b7 {col_titles[c]}", fontsize=10, color=_INK, pad=-2)
        # ---- row 3: 2-D profile ----
        for c in range(3):
            ax = fig.add_subplot(gs[3, c])
            if c == 0:
                _profile(ax, xyz, y_true, "truth", "Profile \u00b7 ground truth", point_size=5)
            elif c == 1:
                _profile(ax, xyz, y_pred, "pred", "Profile \u00b7 prediction", point_size=5)
            else:
                _profile_error(ax, xyz, y_true, y_pred, "Profile \u00b7 errors", point_size=5)

        # ---- shared legend ----
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=6,
                   markerfacecolor=GROUND_COLOR, markeredgecolor="none", label="ground"),
            Line2D([0], [0], marker="o", linestyle="", markersize=6,
                   markerfacecolor=NONGROUND_COLOR, markeredgecolor="none", label="non-ground"),
            Line2D([0], [0], marker="o", linestyle="", markersize=6,
                   markerfacecolor=CORRECT_COLOR, markeredgecolor="none", label="correct"),
            Line2D([0], [0], marker="o", linestyle="", markersize=7,
                   markerfacecolor=FN_COLOR, markeredgecolor="none", label="ground\u2192NG (FP1)"),
            Line2D([0], [0], marker="o", linestyle="", markersize=7,
                   markerfacecolor=FP_COLOR, markeredgecolor="none", label="NG\u2192ground (FP2)"),
        ]
        leg = fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5,
                         frameon=True, facecolor="white", edgecolor=_FRAME,
                         bbox_to_anchor=(0.5, 0.012), handletextpad=0.3, columnspacing=1.4)
        leg.get_frame().set_linewidth(0.6)
        fig.text(0.965, 0.012, f"{m['n']:,} points  \u00b7  vertical exaggeration applied",
                 ha="right", va="bottom", fontsize=7.5, color="#9aa0a6")

        fig.savefig(out_path, dpi=150, facecolor=_BG)
        plt.close(fig)
    return m


# ===========================================================================
# Section B: gap-free TIN DEM + combined per-scene report
# ===========================================================================
def _tin_grid(xyz_pts, x0, x1, y0, y1, res=240, reduce="max", cell=None):
    """Gap-free **TIN** surface raster.

    Reduces the points to one representative height per coarse cell (so stacked
    multi-returns don't make the surface multi-valued), Delaunay-triangulates the
    representatives, then barycentric-interpolates onto a fine grid via
    ``LinearTriInterpolator`` (gap-free within the convex hull). Any cells outside
    the hull are nearest-filled, so the returned raster is FULLY dense - no holes.

    reduce: ``"max"`` -> DSM (top surface); ``"low"`` -> robust ground (20th pct per
    cell, for DTM / predicted-DTM); ``"min"`` -> strict per-cell minimum.
    """
    import matplotlib.tri as mtri
    from scipy.spatial import cKDTree
    pts = np.asarray(xyz_pts, dtype=np.float64)
    if pts.shape[0] < 8:
        raise ValueError("too few points for a TIN")
    x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
    if cell is None:
        cell = max((x1 - x0), (y1 - y0)) / 160.0
    cell = max(float(cell), 1e-6)
    ix = np.floor((x - x0) / cell).astype(np.int64)
    iy = np.floor((y - y0) / cell).astype(np.int64)
    key = ix * 1000003 + iy
    order = np.argsort(key, kind="stable")
    key_s = key[order]; z_s = z[order]; x_s = x[order]; y_s = y[order]
    cuts = np.concatenate([[0], np.where(np.diff(key_s) != 0)[0] + 1, [len(key_s)]])
    rx = np.empty(len(cuts) - 1); ry = np.empty_like(rx); rz = np.empty_like(rx)
    for c, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        zz = z_s[a:b]
        if reduce == "max":
            j = a + int(np.argmax(zz))
            rx[c] = x_s[j]; ry[c] = y_s[j]; rz[c] = z_s[j]
        else:
            if reduce == "low":
                thr = np.percentile(zz, 20.0); sel = zz <= thr
                rz[c] = float(zz[sel].mean()) if sel.any() else float(zz.min())
            else:
                rz[c] = float(zz.min())
            rx[c] = float(x_s[a:b].mean()); ry[c] = float(y_s[a:b].mean())
    if rx.shape[0] < 4:
        raise ValueError("too few cells for a TIN")
    tri = mtri.Triangulation(rx, ry)               # raises on collinear input -> caught upstream
    interp = mtri.LinearTriInterpolator(tri, rz)
    gx = np.linspace(x0, x1, res); gy = np.linspace(y0, y1, res)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.ma.filled(np.asarray(interp(GX, GY)), np.nan)
    nan = ~np.isfinite(GZ)
    if nan.any():                                   # nearest-fill outside the hull -> fully dense
        _, idx = cKDTree(np.column_stack([rx, ry])).query(
            np.column_stack([GX[nan], GY[nan]]), k=1)
        GZ[nan] = rz[idx]
    return GZ, np.zeros_like(GZ, dtype=bool), (x1 - x0) / res, (y1 - y0) / res


def render_scene_report(xyz, y_true, y_pred, out_path, feats=None,
                        title="MEEPO", point_size=3.0):
    """ONE combined figure per scene (the single per-epoch image), five bands:

      header (metrics)
      INPUTS         - elevation relief + return count + return ratio +
                       normalised intensity + previous-year prior DTM
      TIN DEMs       - gap-free DSM, true DTM, predicted DTM
      CLASSIFICATION - ground truth, prediction, errors (top-down relief)
      PROFILES       - ground truth, prediction, errors (cross-sections)

    ``feats`` is an optional dict with any of ``return_count`` / ``return_ratio`` /
    ``intensity`` / ``prior_dtm`` (each ``(N,)`` aligned to ``xyz``); missing keys
    are skipped. Every panel is wrapped so a single failure leaves that panel blank
    rather than aborting the whole figure.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = max(y_true.shape[0], 1)
    x0, x1, y0, y1 = _extent(xyz); ext = [x0, x1, y0, y1]
    res = 240
    feats = feats or {}

    label_of = dict(return_count="return count", return_ratio="return ratio",
                    intensity="intensity (normalised)", prior_dtm="prev-year prior DTM")
    present = [k for k in ("return_count", "return_ratio", "intensity", "prior_dtm")
               if feats.get(k) is not None and np.asarray(feats[k]).reshape(-1).shape[0] == n]
    order = ["elev"] + present
    n_inputs = len(order)

    def pick_cols(num, lo=3, hi=5):
        best = None
        for c in range(lo, hi + 1):
            rows = int(np.ceil(num / c)); empty = rows * c - num; k = (empty, rows)
            if best is None or k < best[0]:
                best = (k, c, rows)
        return best[1], best[2]

    with plt.rc_context(_RC):
        zmin, zmax = _zrange(xyz[:, 2])
        GZ_all, hole_all, dx, dy = _grid(xyz, xyz[:, 2], x0, x1, y0, y1, res=res, hole_frac=0.5)

        in_cols, in_rows = pick_cols(n_inputs)
        W = 16.0
        in_pw = W / in_cols
        in_h = in_rows * (in_pw + 0.30)
        dem_h = (W / 3.0) + 0.55
        cls_h = (W / 3.0) + 0.55
        prof_h = 3.25
        head_h = 0.95
        total_h = head_h + in_h + dem_h + cls_h + prof_h
        fig = plt.figure(figsize=(W, total_h), facecolor=_BG)
        sub = fig.subfigures(5, 1, height_ratios=[head_h, in_h, dem_h, cls_h, prof_h], hspace=0.0)
        for s in sub:
            s.set_facecolor(_BG)

        # ---------- INPUTS ----------
        gi = sub[1].add_gridspec(in_rows, in_cols, hspace=0.34, wspace=0.16,
                                 top=0.95, bottom=0.03, left=0.045, right=0.965)

        def draw_input(ax, key):
            ax.set_facecolor(_BG)
            if key == "elev":
                _relief_imshow(ax, GZ_all, GZ_all, hole_all, PAPER_TERRAIN, zmin, zmax, dx, dy, ext)
                _frame_map(ax, "Input · elevation (shaded relief)")
                sm = plt.cm.ScalarMappable(cmap=PAPER_TERRAIN, norm=Normalize(zmin, zmax)); sm.set_array([])
                _slim_cbar(fig, ax, sm, "Elev (m)")
                return
            v = np.asarray(feats[key], dtype=np.float64).reshape(-1)
            lbl = label_of[key]
            if key == "intensity":
                lo, hi = np.percentile(v, [2.0, 98.0])
                v = np.clip((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0); lo, hi = 0.0, 1.0
            else:
                lo, hi = np.percentile(v, [2.0, 98.0])
                if hi - lo < 1e-9:
                    lo, hi = float(v.min()), float(v.max() + 1e-6)
            GZ_f, hole_f, _, _ = _grid(xyz, v, x0, x1, y0, y1, res=res, hole_frac=0.5)
            cmap = PAPER_TERRAIN if key == "prior_dtm" else FEATURE_CMAP
            base_relief = GZ_f if key == "prior_dtm" else GZ_all
            _relief_imshow(ax, GZ_f, base_relief, hole_f, cmap, lo, hi, dx, dy, ext)
            _frame_map(ax, f"Input · {lbl}")
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(lo, hi)); sm.set_array([])
            _slim_cbar(fig, ax, sm, lbl)

        for k, key in enumerate(order):
            ax = sub[1].add_subplot(gi[k // in_cols, k % in_cols])
            try:
                draw_input(ax, key)
            except Exception:
                _frame_map(ax, "Input")
        for k in range(len(order), in_rows * in_cols):
            ax = sub[1].add_subplot(gi[k // in_cols, k % in_cols]); ax.axis("off")

        # ---------- TIN DEMs: DSM, true DTM, predicted DTM ----------
        gd = sub[2].add_gridspec(1, 3, wspace=0.13, top=0.88, bottom=0.06, left=0.045, right=0.965)
        dem_spec = [("dsm", "TIN DSM · all returns", None, "max"),
                    ("dtm", "TIN DTM · true ground", (y_true == 1), "low"),
                    ("pred", "TIN DTM · predicted ground", (y_pred == 1), "low")]
        for k, (_, ttl, mask, red) in enumerate(dem_spec):
            ax = sub[2].add_subplot(gd[0, k]); ax.set_facecolor(_BG)
            pts = xyz if mask is None else xyz[mask]
            try:
                if pts.shape[0] < 8:
                    raise ValueError("few pts")
                GZ, hole, _, _ = _tin_grid(pts, x0, x1, y0, y1, res=res, reduce=red)
            except Exception:
                try:
                    src = pts if pts.shape[0] >= 8 else xyz
                    GZ, hole, _, _ = _grid(src, src[:, 2], x0, x1, y0, y1, res=res, hole_frac=0.5)
                except Exception:
                    _frame_map(ax, ttl); continue
            _relief_imshow(ax, GZ, GZ, hole, PAPER_TERRAIN, zmin, zmax, dx, dy, ext)
            _frame_map(ax, ttl); _scale_bar_soft(ax, x0, x1, y0, y1)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            sm = plt.cm.ScalarMappable(cmap=PAPER_TERRAIN, norm=Normalize(zmin, zmax)); sm.set_array([])
            _slim_cbar(fig, ax, sm, "Elev (m)")

        # ---------- CLASSIFICATION ----------
        def ground_grid(mask):
            gp = xyz[mask] if int(mask.sum()) >= 16 else xyz
            if gp.shape[0] > 45000:
                gp = gp[np.random.default_rng(0).choice(gp.shape[0], 45000, replace=False)]
            return _grid(gp, gp[:, 2], x0, x1, y0, y1, res=res, hole_frac=0.5)

        gz_t, gh_t, _, _ = ground_grid(y_true == 1)
        gz_p, gh_p, _, _ = ground_grid(y_pred == 1)
        gc = sub[3].add_gridspec(1, 3, wspace=0.13, top=0.91, bottom=0.06, left=0.045, right=0.965)

        def draw_cls(ax, kind):
            ax.set_facecolor(_BG)
            if kind in ("gt", "pred"):
                gz, gh = (gz_t, gh_t) if kind == "gt" else (gz_p, gh_p)
                _relief_imshow(ax, gz, gz, gh, PAPER_TERRAIN, zmin, zmax, dx, dy, ext)
                lab = y_true if kind == "gt" else y_pred
                ng = xyz[lab == 0]
                if ng.shape[0]:
                    ax.scatter(ng[:, 0], ng[:, 1], c=NONGROUND_COLOR, s=point_size,
                               marker="o", linewidths=0, alpha=0.55, zorder=4)
                _frame_map(ax, "Ground truth" if kind == "gt" else "Prediction")
            else:
                _relief_imshow(ax, gz_t, gz_t, gh_t, PAPER_TERRAIN, zmin, zmax, dx, dy, ext)
                fp1 = (y_true == 1) & (y_pred == 0); fp2 = (y_true == 0) & (y_pred == 1)
                ms = max(point_size * 0.9, 2.0)
                if fp2.any():
                    ax.scatter(xyz[fp2, 0], xyz[fp2, 1], c=FP_COLOR, s=ms, marker="o", linewidths=0, zorder=4)
                if fp1.any():
                    ax.scatter(xyz[fp1, 0], xyz[fp1, 1], c=FN_COLOR, s=ms, marker="o", linewidths=0, zorder=5)
                _frame_map(ax, "Errors  (ground\u2192NG black · NG\u2192ground red)")
            _scale_bar_soft(ax, x0, x1, y0, y1); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)

        for k, kind in enumerate(["gt", "pred", "err"]):
            ax = sub[3].add_subplot(gc[0, k])
            try:
                draw_cls(ax, kind)
            except Exception:
                _frame_map(ax, "")

        # ---------- PROFILES ----------
        gpf = sub[4].add_gridspec(1, 3, wspace=0.13, top=0.85, bottom=0.14, left=0.045, right=0.965)
        profs = [lambda ax: _profile(ax, xyz, y_true, "truth", "Profile · ground truth", point_size=point_size + 1),
                 lambda ax: _profile(ax, xyz, y_pred, "pred", "Profile · prediction", point_size=point_size + 1),
                 lambda ax: _profile_error(ax, xyz, y_true, y_pred, "Profile · errors", point_size=point_size + 1)]
        for k, fn in enumerate(profs):
            ax = sub[4].add_subplot(gpf[0, k])
            try:
                fn(ax)
            except Exception:
                _style_profile(ax, "Profile")

        # ---------- header ----------
        tp1 = int(((y_true == 0) & (y_pred == 0)).sum())
        tp2 = int(((y_true == 1) & (y_pred == 1)).sum())
        fp1 = int(((y_true == 1) & (y_pred == 0)).sum())
        fp2 = int(((y_true == 0) & (y_pred == 1)).sum())
        oa = 100.0 * (tp1 + tp2) / n
        iou2 = 100.0 * tp2 / max(tp2 + fp1 + fp2, 1)
        try:
            from .metrics import dtm_rmse_components
            _sse, _npts = dtm_rmse_components(xyz, y_pred, y_true, res=1.0)
            _rmse = float(np.sqrt(_sse / _npts)) if _npts > 0 else float("nan")
        except Exception:
            _rmse = float("nan")
        fig.text(0.045, 1 - 0.34 / total_h, title, ha="left", fontsize=14, weight="bold", color=_INK)
        fig.text(0.045, 1 - 0.62 / total_h,
                 f"OA {oa:.1f}%     ground IoU {iou2:.1f}%     DTM RMSE {_rmse:.3f} m     "
                 f"FP1 (ground\u2192NG) {fp1:,}     FP2 (NG\u2192ground) {fp2:,}     {n:,} points",
                 ha="left", fontsize=9.5, color=_SUBINK)
        fig.text(0.965, 1 - 0.34 / total_h, "INPUTS  \u00b7  TIN DEM  \u00b7  CLASSIFICATION  \u00b7  PROFILES",
                 ha="right", fontsize=8.5, color="#9aa0a6", weight="bold")
        fig.savefig(out_path, dpi=150, facecolor=_BG)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# SPAG-DC before/after (replaces the SPAG-DC panel in the per-epoch gallery)
# --------------------------------------------------------------------------- #
def render_spag_dc_panel(xyz, y_raw, y_refined, out_path: str, y_true=None,
                           title: str = "SPAG-DC refine", res: float = 1.0) -> dict:
    """Visualise the SPAG-DC spike reclassifier's effect: the DTM surface
    BEFORE vs AFTER (built mean-per-cell, exactly the RMSE metric's DTM, so the
    reclassified spikes show as bright anomalies in 'before' and vanish in
    'after'), plus the reclassified spike points. Header reports DTM-RMSE and
    ground-IoU before->after when ground truth is available, so the impact is
    quantitative even when few points flip. Reclassified = ground(1)->non-ground(0)."""
    from ..data.dtm import build_dtm_from_ground
    from .metrics import dtm_rmse_components
    xyz = np.asarray(xyz, dtype=np.float64)
    yr = np.asarray(y_raw).reshape(-1).astype(np.int64)
    yf = np.asarray(y_refined).reshape(-1).astype(np.int64)
    reclassified = (yr == 1) & (yf == 0)              # ground -> non-ground spikes
    n_recl = int(reclassified.sum())
    info = {"n_reclassified": n_recl}

    x0, x1 = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    y0, y1 = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    bounds = (x0, y0, x1, y1)
    gb, ga = (yr == 1), (yf == 1)
    dtm_b = build_dtm_from_ground(xyz[gb], res=res, bounds=bounds).data if int(gb.sum()) >= 4 else None
    dtm_a = build_dtm_from_ground(xyz[ga], res=res, bounds=bounds).data if int(ga.sum()) >= 4 else None

    sub = ""
    if y_true is not None:
        yt = np.asarray(y_true).reshape(-1).astype(np.int64)
        sse_b, nb = dtm_rmse_components(xyz, yr, yt, res)
        sse_a, na = dtm_rmse_components(xyz, yf, yt, res)
        rb = float(np.sqrt(sse_b / max(nb, 1))); ra = float(np.sqrt(sse_a / max(na, 1)))
        mr = _binary_metrics(yt, yr); mc = _binary_metrics(yt, yf)
        info.update(RMSE_raw=rb, RMSE_refined=ra,
                    IoU_ground_raw=mr["IoU_ground"], IoU_ground_refined=mc["IoU_ground"])
        sub = (f"DTM-RMSE {rb:.3f} \u2192 {ra:.3f} m  (\u0394{ra - rb:+.3f})      "
               f"ground-IoU {mr['IoU_ground']:.1f}% \u2192 {mc['IoU_ground']:.1f}%")

    # common robust color range across the two surfaces
    pool = [d[np.isfinite(d)].ravel() for d in (dtm_b, dtm_a) if d is not None]
    if pool:
        allz = np.concatenate(pool)
        vmin, vmax = (float(np.percentile(allz, 2)), float(np.percentile(allz, 98))) if allz.size else (0.0, 1.0)
    else:
        vmin, vmax = 0.0, 1.0
    ext = (x0, x1, y0, y1)

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(13.6, 5.6), facecolor=_BG)
        gs = fig.add_gridspec(2, 3, height_ratios=[0.42, 4.0], hspace=0.10, wspace=0.08,
                              left=0.04, right=0.97, top=0.90, bottom=0.14)
        fig.text(0.04, 0.985, title, ha="left", va="top", fontsize=15, weight="bold", color=_INK)
        hax = fig.add_subplot(gs[0, :]); hax.axis("off")
        msg = (f"SPAG-DC reclassified {n_recl:,} ground spikes \u2192 non-ground"
               if n_recl else "SPAG-DC reclassified 0 points on this scene")
        hax.text(0.0, 0.5, msg, ha="left", va="center", fontsize=11.5, weight="bold",
                 color="#1f6f74", transform=hax.transAxes)
        if sub:
            hax.text(1.0, 0.5, sub, ha="right", va="center", fontsize=10, color=_SUBINK,
                     transform=hax.transAxes)
        hax.axhline(-0.10, color=_FRAME, linewidth=0.8)

        ax0 = fig.add_subplot(gs[1, 0])
        if dtm_b is not None:
            ax0.imshow(dtm_b, origin="lower", extent=ext, cmap="terrain", vmin=vmin, vmax=vmax, aspect="equal")
        _frame_map(ax0, ""); ax0.set_title("before \u00b7 DTM from raw ground", fontsize=10, color=_INK, pad=5)
        _draw_scale_bar(ax0, x0, x1, y0, y1)

        ax1 = fig.add_subplot(gs[1, 1])
        if dtm_a is not None:
            ax1.imshow(dtm_a, origin="lower", extent=ext, cmap="terrain", vmin=vmin, vmax=vmax, aspect="equal")
        _frame_map(ax1, ""); ax1.set_title("after \u00b7 DTM (SPAG-DC)", fontsize=10, color=_INK, pad=5)
        _draw_scale_bar(ax1, x0, x1, y0, y1)

        ax2 = fig.add_subplot(gs[1, 2])
        keep = ~reclassified
        if keep.any():
            ax2.scatter(xyz[keep, 0], xyz[keep, 1], c=CORRECT_COLOR, s=1.0, marker=".",
                        linewidths=0, alpha=0.40, zorder=2, rasterized=True)
        if reclassified.any():
            ax2.scatter(xyz[reclassified, 0], xyz[reclassified, 1], c="#d61f8c", s=9, marker=".",
                        linewidths=0, alpha=0.95, zorder=5, rasterized=True)
        ax2.set_xlim(x0, x1); ax2.set_ylim(y0, y1); ax2.set_aspect("equal", adjustable="box")
        _draw_scale_bar(ax2, x0, x1, y0, y1); _frame_map(ax2, "")
        ax2.set_title("reclassified spikes", fontsize=10, color=_INK, pad=5)

        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=CORRECT_COLOR,
                          markeredgecolor="none", label="kept"),
                   Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor="#d61f8c",
                          markeredgecolor="none", label="reclassified spike (ground\u2192NG)")]
        leg = fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=True,
                         facecolor="white", edgecolor=_FRAME, bbox_to_anchor=(0.5, 0.012))
        leg.get_frame().set_linewidth(0.6)
        fig.savefig(out_path, dpi=150, facecolor=_BG)
        plt.close(fig)
    return info


def update_refine_charts(history: List[Dict], out_dir: str) -> None:
    """Write ``refine_impact.csv`` + ``refine_impact.png``: SPAG-DC's before/after
    on the deployed validation metrics over epochs (raw vs refined ground-IoU and
    DTM-RMSE, plus how many spikes were reclassified each epoch)."""
    import csv
    rows = [r for r in history if r.get("IoU1_raw") is not None and np.isfinite(r.get("IoU1_raw", float("nan")))]
    if not rows:
        return
    csv_path = os.path.join(out_dir, "refine_impact.csv")
    cols = ["epoch", "IoU1_raw", "IoU1_ref", "IoU2_raw", "IoU2_ref", "OA_raw", "OA_ref",
            "RMSE_raw", "RMSE_ref", "n_reclassified", "reclassified_frac"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            w.writerow([r.get("epoch"), r.get("IoU1_raw"), r.get("IoU1"), r.get("IoU2_raw"),
                        r.get("IoU2"), r.get("OA_raw"), r.get("OA"), r.get("RMSE_raw"),
                        r.get("RMSE"), r.get("n_reclassified"), r.get("reclassified_frac")])

    ep = np.array([r["epoch"] for r in rows], dtype=float)
    def col(name):
        return np.array([r.get(name, np.nan) if r.get(name) is not None else np.nan for r in rows], dtype=float)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3), facecolor=_BG)
        fig.suptitle("SPAG-DC refinement impact (validation, raw vs refined)",
                     x=0.02, ha="left", fontsize=14, weight="bold", color=_INK)
        ax = axes[0]
        ax.plot(ep, col("IoU2_raw"), "--", color="#9aa0a6", label="ground IoU raw")
        ax.plot(ep, col("IoU2"), "-", color="#1f6f74", label="ground IoU refined")
        ax.set_title("ground IoU (IoU2)", fontsize=10, color=_INK); ax.set_xlabel("epoch")
        ax.legend(fontsize=8, frameon=False); ax.grid(True, alpha=0.25)
        ax = axes[1]
        ax.plot(ep, col("RMSE_raw"), "--", color="#cf8a3b", label="DTM-RMSE raw")
        ax.plot(ep, col("RMSE"), "-", color="#a8331f", label="DTM-RMSE refined")
        ax.set_title("DTM-RMSE (m, lower is better)", fontsize=10, color=_INK); ax.set_xlabel("epoch")
        ax.legend(fontsize=8, frameon=False); ax.grid(True, alpha=0.25)
        ax = axes[2]
        ax.plot(ep, col("n_reclassified"), "-", color="#d61f8c", label="spikes reclassified")
        ax.set_title("ground spikes reclassified / epoch", fontsize=10, color=_INK); ax.set_xlabel("epoch")
        ax.legend(fontsize=8, frameon=False); ax.grid(True, alpha=0.25)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(os.path.join(out_dir, "refine_impact.png"), dpi=140, facecolor=_BG)
        plt.close(fig)
