#!/usr/bin/env python3
"""
02 - Build the previous-year ground DTM for every downloaded year-pair.

This realises the single sanctioned deviation: each current-year point cloud is
given a previous-year terrain context (consumed by the DTM CNN branch). For each
pair in ``manifest.json`` we produce one small previous-year DTM raster PER CURRENT
CLOUD, built from that cloud's spatially-matched previous-year twin (the same
fixed-grid cell in the prior survey; ``clouds[i]`` <-> ``prev_clouds[i]`` are
index-aligned). Each is a compact ``.npz`` (data, x_min, y_min, res) that stage 04
samples. Per-cloud rasters are ~tile-sized (a few MB) instead of region-sized (GBs),
so the build and the stage-04 crop stay within memory and run at full parallelism.

Source (auto-selected per pair):
  * prev-year point clouds (``prev_clouds``): the twin's ground returns are
    rasterised with ``build_dtm_from_ground`` -> one raster per current cloud;
  * else ``dtms`` (LINZ ``dem_1m`` GeoTIFFs): mosaicked once and shared by the
    pair's clouds.

Output: ``<out>/dtm/<region>_<prev_year>/<cloud>.npz`` and an updated
``manifest.json`` gaining a per-cloud ``dtm_rasters`` list (parallel to ``clouds``).

    python scripts/02_build_dtm.py --root data/nz
"""
from __future__ import annotations

import os
# Keep each worker process single-threaded in native libs (rayon/BLAS/OMP); the
# parallelism comes from the worker *processes*. Without this, N workers x N
# library threads oversubscribes and the lazrs rayon pool exhausts OS threads.
for _v in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meepo_nz.utils.config import Config
from meepo_nz.data.dtm import Raster, build_dtm_from_ground
from meepo_nz.utils.laz_io import read_points, GROUND_CLASSES


def _save_raster(path: str, r: Raster):
    np.savez_compressed(path, data=r.data.astype(np.float32),
                        x_min=np.float64(r.x_min), y_min=np.float64(r.y_min),
                        res=np.float64(r.res))


def _mosaic_geotiffs(paths, res):
    """Mosaic a list of GeoTIFFs into one Raster (requires rasterio)."""
    import rasterio
    from rasterio.merge import merge
    srcs = [rasterio.open(p) for p in paths]
    try:
        mosaic, transform = merge(srcs, res=(res, res))
    finally:
        for s in srcs:
            s.close()
    band = mosaic[0].astype(np.float32)
    nodata = srcs[0].nodata if srcs else None
    if nodata is not None:
        band = np.where(band == nodata, np.nan, band)
    # rasterio rows go top->bottom (north up); our Raster rows go y_min->y_max,
    # so flip vertically and set the origin to the lower-left corner.
    band = np.flipud(band)
    x_min = transform.c
    y_max = transform.f
    y_min = y_max - band.shape[0] * res
    return Raster(data=band, x_min=float(x_min), y_min=float(y_min), res=float(res))


def _dtm_from_prev_clouds(paths, res):
    chunks = []
    for p in paths:
        xyz, cls, _nr, _rn, _it, _, _ = read_points(p, want_rgb=False)
        g = np.isin(cls, GROUND_CLASSES)
        if g.any():
            chunks.append(xyz[g])
    if not chunks:
        return None
    ground = np.concatenate(chunks, axis=0)
    return build_dtm_from_ground(ground, res=res)


def _build_one_pair(task):
    """Worker: build ONE pair's previous-year DTM, save it, return its path."""
    pair, i, res, dtm_dir = task
    region = pair.get("region", f"pair{i}")
    prev_year = pair.get("prev_year", "prev")
    out_path = os.path.join(dtm_dir, f"{region}_{prev_year}_{i:03d}.npz")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return (i, region, prev_year, out_path, "exists; skipped")     # resume
    raster = None
    msg = ""
    if pair.get("dtms"):
        try:
            raster = _mosaic_geotiffs(pair["dtms"], res)
            msg = f"mosaicked {len(pair['dtms'])} DTM tiles -> {raster.shape}"
        except Exception as e:
            msg = f"GeoTIFF mosaic failed ({e}); trying point-cloud fallback"
    if raster is None and pair.get("prev_clouds"):
        raster = _dtm_from_prev_clouds(pair["prev_clouds"], res)
        if raster is not None:
            msg = f"built DTM from prev ground -> {raster.shape}"
    if raster is None:
        return (i, region, prev_year, None, "no DTM source; tiles will use a zero channel")
    _save_raster(out_path, raster)
    return (i, region, prev_year, out_path, msg)


def _build_one_cloud_dtm(task):
    """Build ONE small previous-year DTM raster for a SINGLE current cloud, from its
    spatially-matched previous-year twin (same fixed-grid cell, prior survey). The
    raster is ~tile-sized (a few MB), so the build and the stage-04 crop never OOM and
    can run at full parallelism. Resumable: an existing output is skipped."""
    cur_path, prev_path, out_path, res = task
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return (cur_path, out_path, "exists")
    try:
        raster = _dtm_from_prev_clouds([prev_path], res)   # single twin
    except Exception as e:
        return (cur_path, None, f"FAILED ({e})")
    if raster is None:
        return (cur_path, None, "no ground in twin")
    _save_raster(out_path, raster)
    return (cur_path, out_path, str(raster.shape))


def main():
    ap = argparse.ArgumentParser(description="Build previous-year DTM rasters.")
    ap.add_argument("--root", required=True, help="Download root (holds manifest.json).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--res", type=float, default=None, help="DTM resolution (m).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel worker processes (default: all CPU cores).")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    res = args.res or cfg.dtm_resolution

    man_path = os.path.join(args.root, "manifest.json")
    with open(man_path) as fh:
        manifest = json.load(fh)

    dtm_dir = os.path.join(args.root, "dtm")
    os.makedirs(dtm_dir, exist_ok=True)

    pairs = manifest.get("pairs", [])
    import concurrent.futures as _cf
    from concurrent.futures.process import BrokenProcessPool
    import multiprocessing as _mp

    # ---- flat PER-CLOUD task list ----
    # Each current cloud gets its OWN small DTM raster, built from its spatially-matched
    # previous-year twin (clouds[i] <-> prev_clouds[i] are index-aligned twins from the
    # download). This replaces the old one-giant-raster-per-pair scheme: per-cloud
    # rasters are ~tile-sized (a few MB) rather than region-sized (GBs), so neither this
    # build nor the stage-04 crop OOMs, and both can use full parallelism. The raster
    # path for clouds[i] is stored at pair["dtm_rasters"][i] (parallel to "clouds").
    cloud_tasks = []                  # (cur_path, prev_twin, out_path, res)
    dem_pairs = []                    # (pair, pid) for GeoTIFF-DEM pairs (one shared mosaic)
    for pid, e in enumerate(pairs):
        clouds = e.get("clouds", [])
        prevs = e.get("prev_clouds", [])
        e.setdefault("dtm_rasters", [None] * len(clouds))
        if prevs and len(prevs) == len(clouds):
            sub = os.path.join(dtm_dir, f"{e.get('region', 'pair' + str(pid))}_{e.get('prev_year', 'prev')}")
            os.makedirs(sub, exist_ok=True)
            for c, pv in zip(clouds, prevs):
                op = os.path.join(sub, os.path.splitext(os.path.basename(c))[0] + ".npz")
                cloud_tasks.append((c, pv, op, res))
        elif e.get("dtms"):
            dem_pairs.append((e, pid))

    n_cpu = os.cpu_count() or 4
    default_w = min(n_cpu, 32)        # per-cloud rasters are small -> high parallelism is fine
    n_workers = max(min(int(args.workers) if args.workers else default_w, len(cloud_tasks) or 1), 1)

    if cloud_tasks:
        print(f"[02] building {len(cloud_tasks)} per-cloud previous-year DTMs "
              f"(twin -> ~tile-sized raster) with {n_workers} workers ...")

        def _run(nw):
            with _cf.ProcessPoolExecutor(max_workers=nw, mp_context=_mp.get_context("spawn")) as ex:
                return list(ex.map(_build_one_cloud_dtm, cloud_tasks))

        try:
            results = _run(n_workers)
        except BrokenProcessPool:
            print("[02] pool died (OOM?); retrying SERIALLY (finished DTMs are skipped) ...", flush=True)
            results = [_build_one_cloud_dtm(t) for t in cloud_tasks]
        built = {c: op for c, op, _ in results}
        ok = sum(1 for v in built.values() if v)
        fails = [(c, m) for c, op, m in results if not op]
        print(f"[02]   {ok}/{len(results)} per-cloud DTMs ready"
              + (f"; {len(fails)} tiles have no usable twin -> zero DTM channel" if fails else ""))
        for c, m in fails[:8]:
            print(f"[02]     no DTM: {os.path.basename(c)} ({m})")
        for e in pairs:                                   # write per-cloud paths back
            cl = e.get("clouds", [])
            if e.get("prev_clouds") and len(e["prev_clouds"]) == len(cl):
                e["dtm_rasters"] = [built.get(c) for c in cl]

    for e, pid in dem_pairs:           # GeoTIFF source: keep one shared mosaic per pair
        _, region, prev_year, out_path, msg = _build_one_pair((e, pid, res, dtm_dir))
        print(f"[02] {region} {prev_year} (DEM mosaic): {msg}", flush=True)
        e["dtm_raster"] = out_path
        e["dtm_rasters"] = [out_path] * len(e.get("clouds", []))

    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    n_with = sum(1 for e in pairs for r in (e.get("dtm_rasters") or []) if r)
    print(f"[02] updated {man_path} with per-cloud dtm_rasters ({n_with} clouds have a DTM)")


if __name__ == "__main__":
    main()
