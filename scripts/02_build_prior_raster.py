#!/usr/bin/env python3
"""
02 - Build the previous-year CLASSIFICATION raster for every downloaded year-pair
     (Deviation A).

For each current-year cloud we rasterise its spatially-matched previous-year twin
(``clouds[i]`` <-> ``prev_clouds[i]`` are index-aligned) into a compact 5-channel
prior raster (data/dtm.py:build_prior_raster_from_prev):

    [ DTM, DSM, nDSM, ground_prob, coverage ]   (see models/prior_raster_encoder.py)

Unlike the old single-channel ground-DTM build, this reads ALL classes of the prior
survey so the raster carries where ground/vegetation/structures were and how
confidently each cell was ground - the spatial prior the MEEPO raster branch
consumes. Each is a small ``.npz`` (data (C,H,W), x_min, y_min, res, channels) that
stage 04 crops/downsamples per tile. Per-cloud rasters are ~tile-sized, so build and
crop stay in memory and run at full parallelism.

Output: ``<out>/prior/<region>_<prev_year>/<cloud>.npz`` and an updated
``manifest.json`` gaining a per-cloud ``prior_rasters`` list (parallel to ``clouds``).
A pair with only LINZ ``dem_1m`` GeoTIFFs (no prev cloud) falls back to a height-only
prior (DSM=DTM, nDSM=0, ground_prob=1, coverage=1).

    python scripts/02_build_prior_raster.py --root data/nz
"""
from __future__ import annotations

import os
for _v in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meepo_nz.utils.config import Config
from meepo_nz.data.dtm import (MultiRaster, build_prior_raster_from_prev,
                                    prior_from_raster_file, PRIOR_RASTER_CHANNELS)
from meepo_nz.utils.laz_io import read_points, GROUND_CLASSES
from meepo_nz.data.folder_manifest import build_folder_manifest, match_rasters


def _save_multiraster(path: str, mr: MultiRaster):
    np.savez_compressed(path, data=mr.data.astype(np.float32),
                        x_min=np.float64(mr.x_min), y_min=np.float64(mr.y_min),
                        res=np.float64(mr.res),
                        channels=np.array(mr.channels, dtype=object))


def _prior_from_prev_clouds(paths, res):
    xyz_chunks, cls_chunks = [], []
    for p in paths:
        xyz, cls, _nr, _rn, _it, _, _ = read_points(p, want_rgb=False)
        xyz_chunks.append(xyz); cls_chunks.append(cls)
    if not xyz_chunks:
        return None
    xyz = np.concatenate(xyz_chunks, 0)
    cls = np.concatenate(cls_chunks, 0)
    return build_prior_raster_from_prev(xyz, cls, GROUND_CLASSES, res=res)


def _prior_from_dem(paths, res):
    """Height-only fallback prior from a LINZ DEM GeoTIFF mosaic (requires rasterio)."""
    import rasterio
    from rasterio.merge import merge
    srcs = [rasterio.open(p) for p in paths]
    try:
        mosaic, transform = merge(srcs, res=(res, res))
        nodata = srcs[0].nodata
    finally:
        for s in srcs:
            s.close()
    band = np.flipud(mosaic[0].astype(np.float32))
    if nodata is not None:
        band = np.where(band == nodata, np.nan, band)
    from meepo_nz.data.dtm import _nearest_fill
    dtm = _nearest_fill(band)
    data = np.stack([dtm, dtm, np.zeros_like(dtm),
                     np.ones_like(dtm), np.ones_like(dtm)], 0).astype(np.float32)
    x_min = transform.c
    y_min = transform.f - band.shape[0] * res
    return MultiRaster(data, float(x_min), float(y_min), float(res), PRIOR_RASTER_CHANNELS)


def _build_one_cloud_prior(task):
    cur_path, prev_path, out_path, res = task
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return (cur_path, out_path, "exists")
    try:
        mr = _prior_from_prev_clouds([prev_path], res)
    except Exception as e:
        return (cur_path, None, f"FAILED ({e})")
    if mr is None:
        return (cur_path, None, "empty twin")
    _save_multiraster(out_path, mr)
    return (cur_path, out_path, str(mr.shape))


def _build_one_cloud_prior_from_raster(task):
    """Build the 5-channel prior from a pre-made (hand-crafted) raster matched to a cloud.
    Partial coverage is preserved: cells the raster leaves as NoData become coverage=0."""
    cur_path, raster_path, out_path, res = task
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return (cur_path, out_path, "exists")
    try:
        mr = prior_from_raster_file(raster_path, res)
    except Exception as e:
        return (cur_path, None, f"FAILED ({e})")
    if mr is None:
        return (cur_path, None, "unreadable raster")
    _save_multiraster(out_path, mr)
    cov = float(np.asarray(mr.data[4]).mean()) if mr.data.shape[0] >= 5 else 1.0
    return (cur_path, out_path, f"{mr.shape} cover={cov:.0%}")


def main():
    ap = argparse.ArgumentParser(description="Build previous-year classification rasters.")
    ap.add_argument("--root", required=True, help="Workspace dir (holds/receives manifest.json + prior/).")
    ap.add_argument("--input-dir", default=None,
                    help="Folder of current-year .las/.laz to process directly (bypasses stage 01). "
                         "A manifest.json is synthesized into --root.")
    ap.add_argument("--prev-dir", default=None,
                    help="Optional folder of previous-year .las/.laz. Twins matched by identical file "
                         "stem become the prior source; unmatched current clouds get no prior.")
    ap.add_argument("--raster-dir", default=None,
                    help="Folder of pre-made previous-year rasters (GeoTIFF/ASC/.npz), matched to clouds "
                         "by file stem. Takes precedence over --prev-dir. NoData / uncovered cells are kept "
                         "as coverage=0 so a raster that only partly fills a cloud injects no phantom signal.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--res", type=float, default=None, help="Raster resolution (m).")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    res = args.res or cfg.dtm_resolution

    os.makedirs(args.root, exist_ok=True)
    man_path = os.path.join(args.root, "manifest.json")
    if args.input_dir:
        manifest = build_folder_manifest(args.input_dir, args.prev_dir)
        n_cl = sum(len(p.get("clouds", [])) for p in manifest["pairs"])
        n_tw = sum(len(p.get("prev_clouds", [])) for p in manifest["pairs"])
        with open(man_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[02] synthesized manifest from {args.input_dir}: {n_cl} clouds "
              f"({n_tw} with a previous-year twin) -> {man_path}")
    else:
        with open(man_path) as fh:
            manifest = json.load(fh)

    prior_dir = os.path.join(args.root, "prior")
    os.makedirs(prior_dir, exist_ok=True)
    pairs = manifest.get("pairs", [])

    import concurrent.futures as _cf
    from concurrent.futures.process import BrokenProcessPool
    import multiprocessing as _mp

    n_cpu = os.cpu_count() or 4

    # ---- pre-made raster folder (hand-crafted previous-year priors) ---------------
    # Authoritative for the official run: match a raster to each cloud by file stem and
    # build the 5-channel prior from it. Partial coverage is preserved (NoData -> cover 0).
    if args.raster_dir:
        rtasks = []
        sub = os.path.join(prior_dir, "manual")
        os.makedirs(sub, exist_ok=True)
        for pid, e in enumerate(pairs):
            clouds = e.get("clouds", [])
            e.setdefault("prior_rasters", [None] * len(clouds))
            rmatch = match_rasters(clouds, args.raster_dir)
            for i, (c, rp) in enumerate(zip(clouds, rmatch)):
                if rp is None:
                    continue
                op = os.path.join(sub, os.path.splitext(os.path.basename(c))[0] + ".npz")
                rtasks.append((c, rp, op, res))
        n_match = len(rtasks)
        n_total = sum(len(e.get("clouds", [])) for e in pairs)
        print(f"[02] pre-made raster mode: {n_match}/{n_total} clouds matched a raster in "
              f"{args.raster_dir} (unmatched clouds get no prior -> prev-DTM zero-filled)")
        nw = max(min(int(args.workers) if args.workers else min(n_cpu, 32), n_match or 1), 1)
        if rtasks:
            def _runr(k):
                with _cf.ProcessPoolExecutor(max_workers=k, mp_context=_mp.get_context("spawn")) as ex:
                    return list(ex.map(_build_one_cloud_prior_from_raster, rtasks))
            try:
                results = _runr(nw)
            except BrokenProcessPool:
                print("[02] pool died; retrying serially ...", flush=True)
                results = [_build_one_cloud_prior_from_raster(t) for t in rtasks]
            built = {c: op for c, op, _ in results}
            ok = sum(1 for v in built.values() if v)
            fails = [(c, m) for c, op, m in results if not op]
            print(f"[02]   {ok}/{len(results)} priors built from rasters"
                  + (f"; {len(fails)} failed: " + ", ".join(os.path.basename(c) for c, _ in fails[:5]) if fails else ""))
            for e in pairs:
                e["prior_rasters"] = [built.get(c) for c in e.get("clouds", [])]
        with open(man_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        n_with = sum(1 for e in pairs for r in (e.get("prior_rasters") or []) if r)
        print(f"[02] updated {man_path} with per-cloud prior_rasters ({n_with} clouds have a prior)")
        return

    cloud_tasks, dem_pairs = [], []
    for pid, e in enumerate(pairs):
        clouds = e.get("clouds", [])
        prevs = e.get("prev_clouds", [])
        e.setdefault("prior_rasters", [None] * len(clouds))
        if prevs and len(prevs) == len(clouds):
            sub = os.path.join(prior_dir, f"{e.get('region', 'pair' + str(pid))}_{e.get('prev_year', 'prev')}")
            os.makedirs(sub, exist_ok=True)
            for c, pv in zip(clouds, prevs):
                op = os.path.join(sub, os.path.splitext(os.path.basename(c))[0] + ".npz")
                cloud_tasks.append((c, pv, op, res))
        elif e.get("dtms"):
            dem_pairs.append((e, pid))

    n_workers = max(min(int(args.workers) if args.workers else min(n_cpu, 32),
                        len(cloud_tasks) or 1), 1)

    if cloud_tasks:
        print(f"[02] building {len(cloud_tasks)} per-cloud prior-classification rasters "
              f"(5-ch) with {n_workers} workers ...")

        def _run(nw):
            with _cf.ProcessPoolExecutor(max_workers=nw, mp_context=_mp.get_context("spawn")) as ex:
                return list(ex.map(_build_one_cloud_prior, cloud_tasks))

        try:
            results = _run(n_workers)
        except BrokenProcessPool:
            print("[02] pool died; retrying serially ...", flush=True)
            results = [_build_one_cloud_prior(t) for t in cloud_tasks]
        built = {c: op for c, op, _ in results}
        ok = sum(1 for v in built.values() if v)
        fails = [(c, m) for c, op, m in results if not op]
        print(f"[02]   {ok}/{len(results)} prior rasters ready"
              + (f"; {len(fails)} tiles have no usable twin -> synthesized prior" if fails else ""))
        for e in pairs:
            cl = e.get("clouds", [])
            if e.get("prev_clouds") and len(e["prev_clouds"]) == len(cl):
                e["prior_rasters"] = [built.get(c) for c in cl]

    for e, pid in dem_pairs:
        region = e.get("region", f"pair{pid}"); prev_year = e.get("prev_year", "prev")
        out_path = os.path.join(prior_dir, f"{region}_{prev_year}_{pid:03d}.npz")
        try:
            mr = _prior_from_dem(e["dtms"], res)
            _save_multiraster(out_path, mr)
            print(f"[02] {region} {prev_year} (DEM mosaic, height-only prior): {mr.shape}", flush=True)
            e["prior_rasters"] = [out_path] * len(e.get("clouds", []))
        except Exception as ex:
            print(f"[02] {region} {prev_year} DEM mosaic failed ({ex})")

    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    n_with = sum(1 for e in pairs for r in (e.get("prior_rasters") or []) if r)
    print(f"[02] updated {man_path} with per-cloud prior_rasters ({n_with} clouds have a prior)")


if __name__ == "__main__":
    main()
