"""
Shared helpers for large-scene inference on STAGE-04 PREPROCESSED TILES.

These were factored out of ``scripts/09_infer_large_scenes.py`` so the tile-scan,
grid-block grouping, DTM mosaic and resume logic live in the package (importable
and unit-testable) rather than being duplicated in the script. The actual voting
uses :func:`meepo_nz.inference.voting.predict_cloud_spheres` (the same routine
the per-epoch gallery calls), so a large scene is fed to the network exactly as a
training tile is.

Coordinate frame
----------------
Each tile stores points in its own ``file_origin``-relative (tile-local) frame and
a previous-year DTM raster georeferenced in WORLD coordinates. To merge adjacent
tiles into one scene we put every point in a common frame re-localised in X/Y to
``min(file_origin)`` (so coordinates stay small, the regime the network/gallery
expect) but we KEEP Z IN WORLD UNITS. That last point matters: ``crop_dtm_patch``
forms the terrain channel as ``dtm_z - centre_z`` and the network was trained with
``dtm_z`` and ``centre_z`` both in world Z (``__getitem__`` uses ``cw = centre +
origin``). Re-localising Z would offset the whole patch by the scene's min-Z
(hundreds of metres) and push the terrain channel far out of distribution. We
therefore zero the Z component of the common origin and re-reference only X/Y.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from ..data.dtm import Raster
from ..data.tile_io import load_tile
from ..utils.laz_io import IGNORE_LABEL


# --------------------------------------------------------------------------- scan
def parse_en_region(stem: str):
    """(E, N) grid coords = the last two integer runs in the filename; region = the
    prefix before them. Keying by region means tiles from different captures (which
    reuse the same local E/N indices) never merge across each other."""
    ms = list(re.finditer(r"\d+", stem))
    if len(ms) >= 2:
        e, n = int(ms[-2].group()), int(ms[-1].group())
        region = stem[:ms[-2].start()].rstrip("_-. ") or "scene"
        return (e, n), region
    return None, stem


def scan_tiles(tiles_dir: str, split_filter: str | None = None):
    """Light scan of a stage-04 tile dir: per tile returns point count, world
    ``file_origin``, parsed (E,N), region and split (memmap, no big-array load)."""
    out = []
    for p in sorted(glob.glob(os.path.join(tiles_dir, "*.npz"))):
        try:
            t = load_tile(p, mmap=True)
        except Exception:
            continue
        sv = t.get("split")
        sp = str(np.asarray(sv).reshape(-1)[0]) if sv is not None and np.asarray(sv).size else "train"
        if split_filter and sp != split_filter:
            continue
        try:
            n = int(np.asarray(t["local"]).shape[0])
            fo = np.asarray(t["file_origin"], dtype=np.float64).reshape(-1)
        except Exception:
            continue
        en, region = parse_en_region(os.path.splitext(os.path.basename(p))[0])
        out.append({"path": p, "n": n, "file_origin": fo, "en": en, "region": region, "split": sp})
    return out


# ---------------------------------------------------------------- grid grouping
def components(en_set, step):
    """4-connected components over a set of (E,N) grid coordinates."""
    seen, comps = set(), []
    for s in en_set:
        if s in seen:
            continue
        st, comp = [s], []
        while st:
            e, n = st.pop()
            if (e, n) in seen or (e, n) not in en_set:
                continue
            seen.add((e, n)); comp.append((e, n))
            st += [(e + step, n), (e - step, n), (e, n + step), (e, n - step)]
        comps.append(comp)
    return comps


def compact(comp, max_tiles):
    """Keep the ``max_tiles`` cells of a component closest to its centroid."""
    if len(comp) <= max_tiles:
        return comp
    cx = sum(e for e, _ in comp) / len(comp)
    cy = sum(n for _, n in comp) / len(comp)
    return sorted(comp, key=lambda en: (en[0] - cx) ** 2 + (en[1] - cy) ** 2)[:max_tiles]


def build_grid_blocks(tiles, max_tiles, num_scenes):
    """Group tiles into contiguous grid blocks (per region, <= ``max_tiles`` each)
    and return the ``num_scenes`` largest by total point count."""
    by_region = {}
    for t in tiles:
        if t["en"] is None:
            continue
        by_region.setdefault(t["region"], {})[t["en"]] = t
    blocks = []
    for region, en_map in by_region.items():
        ens = list(en_map.keys())
        Es = sorted({e for e, _ in ens}); Ns = sorted({n for _, n in ens})
        gaps = [b - a for a, b in zip(Es, Es[1:]) if b > a] + \
               [b - a for a, b in zip(Ns, Ns[1:]) if b > a]
        step = min(gaps) if gaps else 1
        for comp in components(set(ens), step):
            sub = compact(comp, max_tiles)
            ts = [en_map[en] for en in sub]
            e0 = min(e for e, _ in sub); n0 = min(n for _, n in sub)
            blocks.append({"tiles": ts, "n": sum(t["n"] for t in ts), "n_tiles": len(sub),
                           "name": f"{region}_grid{len(sub)}t_{e0}_{n0}"})
    blocks.sort(key=lambda b: b["n"], reverse=True)
    return blocks[: max(num_scenes, 1)]


# ------------------------------------------------------------------ DTM + load
def mosaic_relocal_dtm(rasters, common_origin):
    """Mosaic per-tile stored DTM rasters (world-framed, equal res) into one raster
    and re-reference it to ``common_origin`` IN X/Y ONLY. The Z (elevation) values
    are left in world units so the terrain channel matches training (see module
    docstring). Returns None if no real DTM is available (1x1 placeholders ignored)."""
    rasters = [r for r in rasters if r is not None and np.asarray(r.data).size > 1]
    if not rasters:
        return None
    res = float(rasters[0].res)
    x0 = min(r.x_min for r in rasters); y0 = min(r.y_min for r in rasters)
    x1 = max(r.x_min + r.data.shape[1] * res for r in rasters)
    y1 = max(r.y_min + r.data.shape[0] * res for r in rasters)
    W = max(int(round((x1 - x0) / res)), 1); H = max(int(round((y1 - y0) / res)), 1)
    big = np.full((H, W), np.nan, dtype=np.float32)
    for r in rasters:
        c0 = int(round((r.x_min - x0) / res)); r0 = int(round((r.y_min - y0) / res))
        h = min(r.data.shape[0], H - r0); w = min(r.data.shape[1], W - c0)
        if h <= 0 or w <= 0:
            continue
        blk = big[r0:r0 + h, c0:c0 + w]
        src = np.asarray(r.data[:h, :w], dtype=np.float32)
        big[r0:r0 + h, c0:c0 + w] = np.where(np.isnan(blk), src, blk)
    return Raster(big, float(x0 - common_origin[0]), float(y0 - common_origin[1]), res)


def load_block(tiles):
    """Materialise + merge a block of preprocessed tiles into one scene.

    Returns ``(relocal_xyz, num_returns, return_number, intensity, ret_ratio,
    labels, dtm_raster, common_origin)``. ``relocal_xyz`` is re-localised in X/Y to
    ``common_origin`` (Z kept in world units); add ``common_origin`` back to recover
    world coordinates for writing/rendering."""
    common_origin = np.min(np.stack([t["file_origin"] for t in tiles]), axis=0).astype(np.float64)
    common_origin[2] = 0.0          # keep Z in world units (terrain-channel correctness)
    PTS, RET, INT, RR, LAB, rasters = [], [], [], [], [], []
    for t in tiles:
        pk = load_tile(t["path"], mmap=False)
        local = np.asarray(pk["local"], dtype=np.float64)
        fo = np.asarray(pk["file_origin"], dtype=np.float64).reshape(-1)
        PTS.append(((local + fo) - common_origin).astype(np.float32))
        ret = np.asarray(pk["returns"], dtype=np.float32)
        RET.append(ret)
        n = local.shape[0]
        INT.append(np.asarray(pk["intensity"], dtype=np.float32).reshape(-1) if pk.get("intensity") is not None
                   else np.zeros((n,), np.float32))
        if pk.get("ret_ratio") is not None:
            RR.append(np.asarray(pk["ret_ratio"], dtype=np.float32).reshape(-1))
        else:
            RR.append((ret[:, 1] / np.maximum(ret[:, 0], 1.0)).astype(np.float32))
        LAB.append(np.asarray(pk["labels"], dtype=np.int64).reshape(-1))
        dd = pk.get("dtm_data"); geo = pk.get("dtm_geo")
        if dd is not None and np.asarray(dd).size > 1 and geo is not None:
            g = np.asarray(geo).reshape(-1)
            rasters.append(Raster(np.asarray(dd, dtype=np.float32), float(g[0]), float(g[1]), float(g[2])))
    xyz = np.concatenate(PTS, 0)
    ret = np.concatenate(RET, 0)
    inten = np.concatenate(INT, 0)
    rr = np.concatenate(RR, 0)
    lab = np.concatenate(LAB, 0)
    dtm = mosaic_relocal_dtm(rasters, common_origin)
    return xyz, ret[:, 0], ret[:, 1], inten, rr, lab, dtm, common_origin


# ------------------------------------------------------------------- resume
def resume_from_laz(laz_path: str):
    """Read an existing classified LAZ back into ``(xyz_world, pred, true_label)``
    so visuals can be re-rendered without re-voting."""
    import laspy
    las = laspy.read(laz_path)
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float64)
    pred = (np.asarray(las.classification) == 2).astype(np.int64)
    true_label = None
    if "true_class" in set(las.point_format.dimension_names):
        tc = np.asarray(las.true_class)
        true_label = np.select([tc == 2, tc == 0], [1, int(IGNORE_LABEL)], default=0).astype(np.int64)
    return xyz, pred, true_label
