#!/usr/bin/env python3
"""
04 - Preprocess current-year clouds into network-ready tiles.

For every current-year cloud in ``manifest.json`` we tile it, compute the eight
shallow features (Section 3.1) plus the ninth deviation channel (height above
the previous-year DTM produced by step 02), derive the binary ground label,
classify each tile into the seven special categories, and write a ``.npz``.

Splits are assigned **per cloud** (not per tile) so overlapping tiles never leak
across train/val/test.  A final pass computes per-channel normalisation stats.

    python scripts/04_preprocess.py --root data/nz --out data/nz/tiles
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
from meepo_nz.data.dtm import Raster, MultiRaster, PRIOR_RASTER_CHANNELS
from meepo_nz.data.preprocess import preprocess_file, compute_norm_stats
from meepo_nz.data.folder_manifest import build_folder_manifest


def _load_raster(path):
    if not path or not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return Raster(data=d["data"].astype(np.float32),
                  x_min=float(d["x_min"]), y_min=float(d["y_min"]), res=float(d["res"]))


def _load_multiraster(path):
    """Load a 5-channel previous-year classification raster (stage 02 prior)."""
    if not path or not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    data = np.asarray(d["data"], dtype=np.float32)
    if data.ndim == 2:                                   # tolerate a single-channel file
        data = np.stack([data, data, np.zeros_like(data),
                         np.ones_like(data), np.ones_like(data)], 0)
    chans = tuple(d["channels"]) if "channels" in d else PRIOR_RASTER_CHANNELS
    return MultiRaster(data, float(d["x_min"]), float(d["y_min"]), float(d["res"]), chans)


# Per-WORKER raster caches. Tasks are ordered by pair, so a worker sees a run of
# clouds sharing one prior/DTM back-to-back and reloads only when it changes. This
# keeps per-cloud parallelism (every core busy) without re-reading the raster for
# every cloud; with ample RAM the rare reload on a pair boundary is cheap.
from collections import OrderedDict as _OD
_RASTER_CACHE = _OD()
_PRIOR_CACHE = _OD()
_RASTER_CACHE_MAX = 1


def _get_cached(cache, path, loader):
    if path is None:
        return None
    hit = cache.get(path, "MISS")
    if not isinstance(hit, str):                  # cached (None is a valid miss-result; "MISS" is the sentinel)
        cache.move_to_end(path)
        return hit
    r = loader(path)
    cache[path] = r
    while len(cache) > _RASTER_CACHE_MAX:
        cache.popitem(last=False)
    return r


def _get_raster(path):
    return _get_cached(_RASTER_CACHE, path, _load_raster)


def _get_prior(path):
    return _get_cached(_PRIOR_CACHE, path, _load_multiraster)


def _preprocess_one(task):
    """Worker: process ONE cloud (per-cloud parallelism). The previous-year prior
    raster (5-ch) is cached per worker, so a run of clouds sharing it reloads only on
    change. Falls back to a legacy single-channel DTM when no prior is available."""
    cloud, prior_path, dtm_path, split, out_dir, cfg, seed = task
    from meepo_nz.data.preprocess import preprocess_file
    if not os.path.exists(cloud):
        return (os.path.basename(cloud), split, 0, (prior_path or dtm_path) is not None, True)
    prior = _get_prior(prior_path)
    raster = None if prior is not None else _get_raster(dtm_path)
    rng = np.random.default_rng(seed)
    n = preprocess_file(cloud, cfg, out_dir, prev_dtm=raster, prev_prior=prior, split=split, rng=rng)
    return (os.path.basename(cloud), split, n, (prior is not None or raster is not None), False)


def _preprocess_pair(task):
    """Worker: process ALL clouds of one pair, loading that pair's prior/DTM once."""
    clouds, prior_path, dtm_path, splits, out_dir, cfg, base_seed = task
    from meepo_nz.data.preprocess import preprocess_file
    prior = _load_multiraster(prior_path)         # loaded ONCE per pair
    raster = None if prior is not None else _load_raster(dtm_path)
    out = []
    for k, (cloud, split) in enumerate(zip(clouds, splits)):
        if not os.path.exists(cloud):
            out.append((os.path.basename(cloud), split, 0, (prior is not None or raster is not None), True))
            continue
        rng = np.random.default_rng(base_seed + k)
        n = preprocess_file(cloud, cfg, out_dir, prev_dtm=raster, prev_prior=prior, split=split, rng=rng)
        out.append((os.path.basename(cloud), split, n, (prior is not None or raster is not None), False))
    return out


def _assign_splits(n_clouds, cfg, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_clouds)
    n_test = int(round(cfg.test_fraction * n_clouds))
    n_val = int(round(cfg.val_fraction * n_clouds))
    split = np.array(["train"] * n_clouds, dtype=object)
    split[order[:n_test]] = "test"
    split[order[n_test:n_test + n_val]] = "val"
    return split


def main():
    ap = argparse.ArgumentParser(description="Preprocess clouds -> tiles.")
    ap.add_argument("--root", default=None,
                    help="Workspace dir holding manifest.json (from stage 01/02). "
                         "Optional if --input-dir is given.")
    ap.add_argument("--input-dir", default=None,
                    help="Folder of .las/.laz to tile directly (bypasses stages 01/02). Used only if no "
                         "<root>/manifest.json is present. No prior -> prev-DTM feature is zero-filled. "
                         "For the previous-year prior, run stage 02 with --input-dir/--prev-dir first, then "
                         "pass this stage the same --root.")
    ap.add_argument("--out", default=None, help="Tile output dir (default <root>/tiles).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="Process at most N clouds (debug).")
    ap.add_argument("--auto-dl", action="store_true",
                    help="Set first_subsampling_dl from the data's nominal point spacing.")
    ap.add_argument("--first-subsampling-dl", "--dl", dest="dl", type=float, default=None,
                    help="Force the subsampling grid size in metres (overrides --auto-dl), e.g. --dl 0.1.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel worker processes (default: all CPU cores).")
    ap.add_argument("--in-radius", type=float, default=None,
                    help="Sphere/cylinder radius in m (sphere mode). Bakes the candidate cylinders, e.g. --in-radius 8.")
    ap.add_argument("--min-dtm-coverage", type=float, default=None,
                    help="Drop tiles whose sphere centres are <this fraction covered by the "
                         "previous-year DTM (e.g. 0.9). Guarantees prev-DTM overlap. Default: keep all.")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.min_dtm_coverage is not None:
        cfg.min_dtm_coverage = float(args.min_dtm_coverage)
    if args.in_radius is not None:
        cfg.in_radius = float(args.in_radius); cfg.auto_in_radius = False
    if not args.root and not args.input_dir:
        ap.error("provide --root (with a manifest.json from stage 01/02) or --input-dir <folder>")
    out_dir = args.out or (os.path.join(args.root, "tiles") if args.root else None)
    if not out_dir:
        ap.error("--out is required when using --input-dir without --root")
    os.makedirs(out_dir, exist_ok=True)

    # Manifest precedence: a real manifest.json (stage 01/02 output, incl. built priors)
    # wins; otherwise synthesize one from --input-dir (no prior -> prev-DTM zero-filled).
    man_path = os.path.join(args.root, "manifest.json") if args.root else None
    if man_path and os.path.exists(man_path):
        with open(man_path) as fh:
            manifest = json.load(fh)
    elif args.input_dir:
        manifest = build_folder_manifest(args.input_dir)
        n_cl = sum(len(p.get("clouds", [])) for p in manifest["pairs"])
        print(f"[04] synthesized manifest from {args.input_dir}: {n_cl} clouds (no prior; "
              f"prev-DTM feature zero-filled). Run stage 02 first if you have previous-year data.")
    else:
        ap.error(f"no manifest.json under {args.root!r} and no --input-dir given")

    # flatten (cloud, prior-raster, dtm-raster, pair_id) jobs. Prefer the per-cloud
    # 5-channel prior (pair["prior_rasters"][i], stage 02_build_prior_raster); fall
    # back to a legacy single-channel DTM (pair["dtm_rasters"][i] / pair["dtm_raster"]).
    jobs = []
    for pid, pair in enumerate(manifest.get("pairs", [])):
        priors = pair.get("prior_rasters")
        rasters = pair.get("dtm_rasters")
        fb_prior = pair.get("prior_raster")
        fb_dtm = pair.get("dtm_raster")
        for i, c in enumerate(pair.get("clouds", [])):
            pp = (priors[i] if priors and i < len(priors) else None) or fb_prior
            rp = (rasters[i] if rasters and i < len(rasters) else None) or fb_dtm
            jobs.append((c, pp, rp, pid))
    if args.limit:
        jobs = jobs[:args.limit]

    # --- choose first_subsampling_dl from the survey's NOMINAL point spacing ---
    # (robust to coincident points; never the literal nearest-neighbour minimum)
    first_cloud = next((c for c, _, _, _ in jobs if os.path.exists(c)), None)
    if first_cloud is not None:
        from meepo_nz.utils.laz_io import read_points
        from meepo_nz.data.subsampling import estimate_nominal_spacing
        sxyz, _, _, _, _, _, _ = read_points(first_cloud, want_rgb=False)
        nps = estimate_nominal_spacing(sxyz, floor=cfg.min_subsampling_dl)
        if args.dl is not None:
            cfg.first_subsampling_dl = float(args.dl)
            cfg.auto_subsampling_dl = False
            print(f"[04] first_subsampling_dl = {cfg.first_subsampling_dl} m "
                  f"(explicit --dl; nominal spacing ~{nps:.3f} m)")
        elif args.auto_dl or cfg.auto_subsampling_dl or cfg.first_subsampling_dl <= 0:
            cfg.first_subsampling_dl = round(max(nps, cfg.min_subsampling_dl), 3)
            print(f"[04] auto first_subsampling_dl = {cfg.first_subsampling_dl} m "
                  f"(nominal spacing ~{nps:.3f} m, floor {cfg.min_subsampling_dl} m)")
        else:
            # No warning when first_subsampling_dl is below the *median* nominal
            # spacing. That median is a whole-tile figure, but dense vegetation /
            # bush is locally far denser, and a fine grid there is precisely what
            # lets the network separate ground from non-ground under canopy. A
            # sub-median dl is an intentional resolution choice here, not a mistake.
            print(f"[04] nominal point spacing ~{nps:.3f} m; "
                  f"first_subsampling_dl = {cfg.first_subsampling_dl} m")
        # persist the resolved config so stages 05/06 use the SAME grid as the tiles
        # just written (always - not only on the auto path - so a re-run can't leave a
        # stale snapshot pointing at a different dl).
        try:
            cfg.save(os.path.join(out_dir, "config.used.yaml"))
            print(f"[04] wrote {os.path.join(out_dir, 'config.used.yaml')} "
                  f"(stage 05 auto-loads it)")
        except Exception:
            pass

    # size the input cylinder to the conv reach (auto_in_radius) now that dl is final
    cfg.resolve_geometry()
    if getattr(cfg, "auto_in_radius", False):
        print(f"[04] auto in_radius = {cfg.in_radius:.2f} m "
              f"(conv_radius {cfg.conv_radius} x dl {cfg.first_subsampling_dl} x 2^4; "
              f"sphere_center_spacing = {cfg.sphere_center_spacing:.2f} m)")

    # split is assigned PER CLOUD (so overlapping tiles never leak across splits)
    splits = _assign_splits(len(jobs), cfg, cfg.seed)

    # one task per CLOUD, kept in pair order so a worker sees same-raster clouds
    # back-to-back (its raster cache then reloads only on a pair boundary). This
    # uses every core, instead of one worker grinding a whole pair serially.
    tasks = [(cloud, prior_path, dtm_path, str(splits[i]), out_dir, cfg, int(cfg.seed) + i)
             for i, (cloud, prior_path, dtm_path, pid) in enumerate(jobs)]

    total = 0
    per_split = {"train": 0, "val": 0, "test": 0}
    n_workers = max(int(args.workers) if args.workers else (os.cpu_count() or 4), 1)
    n_workers = min(n_workers, len(tasks)) if tasks else 1
    # Cap for memory: each worker holds the cloud's point arrays + a KD-tree + the
    # cylinder-index buffers (~1-3 GB). The per-cloud DTM raster is now tiny (a few MB,
    # built from the spatial twin in stage 02), so it is NO LONGER the limiter - the old
    # giant per-pair raster that forced this cap is gone, so raise --workers freely.
    MEM_SAFE_WORKERS = 32
    if n_workers > MEM_SAFE_WORKERS:
        print(f"[04] capping workers {n_workers} -> {MEM_SAFE_WORKERS} (per-cloud memory: cloud "
              f"arrays + DTM raster + cylinder indices). Pass --workers <= {MEM_SAFE_WORKERS}; "
              f"lower it further if you still hit an OOM / BrokenProcessPool.")
        n_workers = MEM_SAFE_WORKERS
    n_clouds = len(jobs)
    # contiguous chunks so a worker sees same-raster (same-pair) clouds back-to-back
    # -> the size-1 raster cache reloads only at pair boundaries, not every cloud.
    chunk = max(1, len(tasks) // (n_workers * 8)) if tasks else 1
    print(f"[04] preprocessing {n_clouds} clouds (one task each) "
          f"with {n_workers} workers (chunksize={chunk}) ...")
    import concurrent.futures as _cf
    import multiprocessing as _mp
    done = 0
    with _cf.ProcessPoolExecutor(max_workers=n_workers,
                                 mp_context=_mp.get_context("spawn")) as ex:
        for name, split, n, had_dtm, missing in ex.map(_preprocess_one, tasks, chunksize=chunk):
            done += 1
            if missing:
                print(f"[04] missing (run 01 first): {name}", flush=True)
                continue
            total += n
            per_split[split] += n
            print(f"[04] [{done}/{n_clouds}] {name} "
                  f"split={split} prior={'yes' if had_dtm else 'no'} -> {n} tiles", flush=True)

    print(f"[04] wrote {total} tiles  (train={per_split['train']} "
          f"val={per_split['val']} test={per_split['test']}) to {out_dir}")

    stats = compute_norm_stats(out_dir, cfg)
    print(f"[04] norm_stats.json: {stats['n_features']} channels over "
          f"{stats['n_points']:,} points")


if __name__ == "__main__":
    main()
