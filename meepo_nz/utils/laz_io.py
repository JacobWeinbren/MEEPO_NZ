"""
LAZ / LAS input-output via laspy (with the lazrs or laszip backend).

We only need a thin wrapper:

  * ``read_points``  -> xyz, classification, (optional) rgb, plus the LAS header
                        so we can write results back in the same CRS / offset /
                        scale;
  * ``write_classified`` -> copy an input cloud and overwrite its classification
                        with the ground / non-ground prediction, then save .laz.

ASPRS class codes that matter for this project (binary ground extraction):
    2  = ground          -> label 1 (ground)
    9  = water           -> label 1 (ground)  ("water is ground in this case")
    everything else      -> label 0 (non-ground / unclassified)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import laspy
except Exception:  # pragma: no cover
    laspy = None


GROUND_CLASSES = (2, 9)          # ground + water (both LINZ-mandatory)
WATER_CLASS = 9                  # LINZ-mandatory
BUILDING_CLASS = 6               # OPTIONAL in LINZ data (buyer add-on) - may be absent
HIGH_VEG_CLASS = 5               # OPTIONAL in LINZ data (buyer add-on) - may be absent
LOW_VEG_CLASS = 3                # optional
MED_VEG_CLASS = 4                # optional
ROAD_CLASS = 11                  # optional
BRIDGE_CLASS = 17                # LINZ-mandatory; non-ground (removed from bare earth)
NOISE_CLASSES = (7, 18)          # low / high noise (LINZ-mandatory) - dropped before training
UNCLASSIFIED_CLASSES = (0, 1)    # 0 = never classified, 1 = unassigned. PNOA marks unclassified as BOTH.
IGNORE_LABEL = 2                 # label for unclassified points: excluded from the loss (CrossEntropyLoss ignore_index) AND from metrics. A compact 3rd class index so grid_subsample's majority-vote histogram stays small (n_class=3) while CrossEntropyLoss(ignore_index=2) still skips it on 2-class logits.
VEG_CLASSES = (3, 4, 5)
OPTIONAL_CLASSES = (3, 4, 5, 6, 11)   # not guaranteed present in LINZ point clouds


@dataclass
class LasMeta:
    scales: np.ndarray
    offsets: np.ndarray
    point_format: int
    version: str
    crs: Optional[object] = None        # source CRS (pyproj CRS) if the file carried one


def _require_laspy():
    if laspy is None:
        raise RuntimeError("laspy is not installed (pip install laspy[lazrs])")


def _laspy_read(path):
    """Read a LAS/LAZ decoding **single-threaded**.

    laspy's default lazrs backend is *parallel*: it builds a rayon thread pool
    per process (sized to all cores) and re-initialises the global pool per file.
    Under multiprocessing - many worker processes each decoding LAZ - that
    oversubscribes and exhausts OS threads (rayon panics with
    ``GlobalPoolAlreadyInitialized`` / ``WouldBlock``). We decode single-threaded
    and get parallelism from the worker *processes* instead. Falls back to the
    default backend if the single-threaded one is unavailable.
    """
    try:
        return laspy.read(path, laz_backend=laspy.LazBackend.Lazrs)      # single-threaded
    except Exception:
        try:
            return laspy.read(path, laz_backend=laspy.LazBackend.Laszip)
        except Exception:
            return laspy.read(path)                                       # last resort


def read_points(path: str, want_rgb: bool = False, drop_noise: bool = True
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], LasMeta]:
    """Read a LAS/LAZ file.

    Returns ``(xyz[N,3] float64, classification[N] uint8, num_returns[N] uint8,
    return_number[N] uint8, intensity[N] float32, rgb[N,3] or None, meta)``.

    ``num_returns`` (returns per pulse) and ``return_number`` (which return this
    is) distinguish vegetation (multi-return; early returns) from hard surfaces /
    bare earth (single or *last* returns). ``intensity`` is the per-point return
    strength (a radiometric ground/vegetation cue). They drive tile categorisation
    and, when ``use_return_features`` / ``use_intensity`` are set, per-point input
    channels. Withheld points and noise (classes 7 / 18) are dropped by default.
    """
    _require_laspy()
    las = _laspy_read(path)
    xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
    classification = np.asarray(las.classification).astype(np.uint8)

    try:
        num_returns = np.asarray(las.number_of_returns).astype(np.uint8)
    except Exception:
        num_returns = np.ones((xyz.shape[0],), dtype=np.uint8)
    try:
        return_number = np.asarray(las.return_number).astype(np.uint8)
    except Exception:
        return_number = np.ones((xyz.shape[0],), dtype=np.uint8)
    try:
        intensity = np.asarray(las.intensity).astype(np.float32)
    except Exception:
        intensity = np.zeros((xyz.shape[0],), dtype=np.float32)

    rgb = None
    if want_rgb and {"red", "green", "blue"}.issubset(set(las.point_format.dimension_names)):
        rgb = np.stack([las.red, las.green, las.blue], axis=1).astype(np.float32)

    if drop_noise:
        keep = ~np.isin(classification, NOISE_CLASSES)
        try:                                  # withheld flag (LAS 1.4 PDRF 6-8)
            keep &= ~np.asarray(las.withheld).astype(bool)
        except Exception:
            pass
        if not keep.all():
            xyz = xyz[keep]
            classification = classification[keep]
            num_returns = num_returns[keep]
            return_number = return_number[keep]
            intensity = intensity[keep]
            if rgb is not None:
                rgb = rgb[keep]

    meta = LasMeta(
        scales=np.asarray(las.header.scales, dtype=np.float64),
        offsets=np.asarray(las.header.offsets, dtype=np.float64),
        point_format=int(las.header.point_format.id),
        version=str(las.header.version),
        crs=_safe_parse_crs(las),
    )
    return xyz, classification, num_returns, return_number, intensity, rgb, meta


def _safe_parse_crs(las):
    """Return the file's CRS (pyproj CRS) if present and parseable, else None."""
    try:
        return las.header.parse_crs()        # reads GeoTIFF VLRs (<=1.2) or WKT (1.4)
    except Exception:
        return None


def label_from_classification(classification: np.ndarray,
                              ground_classes=None,
                              unclassified_classes=None) -> np.ndarray:
    """ASPRS class codes -> training label.

    1 = ground/water (ground_classes, default GROUND_CLASSES = {2, 9}),
    0 = non-ground, IGNORE_LABEL (2) = unclassified (unclassified_classes,
    default UNCLASSIFIED_CLASSES = {0, 1}). Unclassified points are kept in
    the cloud as geometric context but carry the ignore label, so they are excluded
    from BOTH the loss (CrossEntropyLoss ignore_index) and the metrics (the
    confusion accumulator only counts true in {0, 1}).

    DATASET-CONVENTION WARNING: the defaults suit LINZ/PNOA-style data where real
    non-ground codes (3-6, ...) exist and 0/1 mean 'never classified'. Many national
    datasets (e.g. some British EA products) classify ONLY ground and leave ALL
    non-ground as class 1 -- under the defaults that IGNORES every non-ground point,
    the loss then supervises ground only, and the model degenerates to predicting
    ground everywhere. For such data pass unclassified_classes=(0,) (stage 04:
    --unclassified-classes 0) so class 1 counts as non-ground.
    """
    classification = np.asarray(classification)
    gc = tuple(ground_classes) if ground_classes is not None else GROUND_CLASSES
    uc = tuple(unclassified_classes) if unclassified_classes is not None else UNCLASSIFIED_CLASSES
    out = np.zeros(classification.shape, dtype=np.int64)              # 0 = non-ground (default)
    out[np.isin(classification, gc)] = 1                              # ground
    out[np.isin(classification, uc)] = IGNORE_LABEL                   # ignore
    return out


def write_classified(
    out_path: str,
    xyz: np.ndarray,
    pred_label: np.ndarray,
    meta: Optional[LasMeta] = None,
    true_label: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    return_number: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    return_ratio: Optional[np.ndarray] = None,
    scan_angle: Optional[np.ndarray] = None,
    epsg: int = 2193,
    extra_dims: bool = False,
) -> None:
    """Write a classified LAZ.

    The PREDICTED ground/non-ground is written to the standard ``classification``
    field (1 -> ASPRS 2 = ground, 0 -> 1 = unclassified/non-ground). Only STANDARD
    LAS point fields are written by default - classification, intensity, return
    number and number of returns - so the file opens cleanly in every LAS reader
    (QGIS of any version, CloudCompare, PDAL, ...). The non-standard ExtraBytes
    (``true_class`` reference labels, ``return_ratio``, ``scan_angle_deg``) are
    written ONLY when ``extra_dims=True`` is passed explicitly; older QGIS builds
    choke on ExtraBytes VLRs, so they are off by default. The reference labels are
    instead surfaced in the per-scene visualisation PNGs, not baked into the LAZ.

    CRS: the source CRS (carried on ``meta``) is stamped if present; otherwise the
    cloud is stamped with ``epsg`` (default 2193, NZGD2000 / NZTM2000 - New Zealand).
    """
    _require_laspy()
    # LAS 1.4 / point format 6: 8-bit classification, int16 scan angle, and clean
    # WKT CRS (so QGIS / CloudCompare pick up EPSG:2193 directly).
    header = laspy.LasHeader(point_format=6, version="1.4")
    if meta is not None:
        header.scales = meta.scales
        header.offsets = meta.offsets
    else:
        header.offsets = np.asarray(xyz).min(axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])

    # ---- CRS: prefer the source CRS, else stamp EPSG:epsg (NZ = 2193) ----
    src_crs = getattr(meta, "crs", None) if meta is not None else None
    try:
        if src_crs is not None:
            header.add_crs(src_crs)
        else:
            from pyproj import CRS as _CRS
            header.add_crs(_CRS.from_epsg(int(epsg)))
    except Exception:
        pass   # best-effort: still write the cloud if pyproj/CRS stamping is unavailable

    # ---- OPTIONAL non-standard ExtraBytes (off by default for max compatibility) ----
    extra = []
    if extra_dims and true_label is not None:
        extra.append(laspy.ExtraBytesParams(
            name="true_class", type=np.uint8,
            description="ref: 2=ground 1=non-ground"))
    if extra_dims and return_ratio is not None:
        extra.append(laspy.ExtraBytesParams(
            name="return_ratio", type=np.float32,
            description="return_num/num_returns"))
    if extra_dims and scan_angle is not None:
        extra.append(laspy.ExtraBytesParams(
            name="scan_angle_deg", type=np.float32, description="scan angle (degrees)"))
    if extra:
        header.add_extra_dims(extra)

    las = laspy.LasData(header)
    las.x = np.asarray(xyz)[:, 0]
    las.y = np.asarray(xyz)[:, 1]
    las.z = np.asarray(xyz)[:, 2]
    las.classification = np.where(np.asarray(pred_label) == 1, 2, 1).astype(np.uint8)

    if extra_dims and true_label is not None:
        tl = np.asarray(true_label)
        # 1 -> 2 (ground); IGNORE_LABEL -> 0 (unclassified, kept distinct); else -> 1 (non-ground)
        las.true_class = np.select(
            [tl == 1, tl == IGNORE_LABEL], [2, 0], default=1).astype(np.uint8)
    if intensity is not None:
        iv = np.asarray(intensity, dtype=np.float64)
        if iv.size and float(np.nanmax(iv)) > 65535.0:      # rescale floats into uint16
            iv = iv * (65535.0 / float(np.nanmax(iv)))
        las.intensity = np.clip(np.nan_to_num(iv), 0, 65535).astype(np.uint16)
    if return_number is not None:
        las.return_number = np.clip(np.asarray(return_number), 1, 15).astype(np.uint8)
    if num_returns is not None:
        las.number_of_returns = np.clip(np.asarray(num_returns), 1, 15).astype(np.uint8)
    if extra_dims and scan_angle is not None:
        las.scan_angle_deg = np.asarray(scan_angle, dtype=np.float32)
    if extra_dims and return_ratio is not None:
        las.return_ratio = np.asarray(return_ratio, dtype=np.float32)

    las.write(out_path)
