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
    ap.add_argument("--ground-classes", default=None,
                    help="Comma list of ASPRS codes counted as GROUND (default '2,9').")
    ap.add_argument("--unclassified-classes", default=None,
                    help="Comma list of ASPRS codes mapped to IGNORE (default '0,1'). If your data marks "
                         "all non-ground as class 1 (British EA convention), pass '0' so class 1 counts "
                         "as NON-GROUND instead of being excluded from the loss.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="Process at most N clouds (debug).")
    ap.add_argument("--auto-dl", action="store_true",
                    help="Set first_subsampling_dl from the data's nominal point spacing.")
    ap.add_argument("--first-subsampling-dl", "--dl", dest="dl", type=float, default=None,
                    help="Force the subsampling grid size in metres (overrides --auto-dl), e.g. --dl 0.1.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel worker processes (default: all CPU cores).")
    ap.add_argument("--min-dtm-coverage", type=float, default=None,
                    help="Drop tiles whose sphere centres are <this fraction covered by the "
                         "previous-year DTM (e.g. 0.9). Guarantees prev-DTM overlap. Default: keep all.")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.min_dtm_coverage is not None:
        cfg.min_dtm_coverage = float(args.min_dtm_coverage)
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

    # ---- label class-mapping overrides (baked into tiles; see config comments) ----
    if getattr(args, "ground_classes", None):
        cfg.ground_classes = tuple(int(x) for x in args.ground_classes.split(","))
    if getattr(args, "unclassified_classes", None) is not None:
        cfg.unclassified_classes = tuple(int(x) for x in args.unclassified_classes.split(",")
                                         ) if args.unclassified_classes.strip() else tuple()

    # ---- raw ASPRS classification histogram (sampled) + mapped-label preview -------
    # The check that catches dataset label conventions BEFORE a wasted training run:
    # if most points map to IGNORE, the loss never sees them and the model degenerates
    # toward predicting the supervised majority (usually: ground everywhere).
    try:
        import collections
        import laspy as _laspy
        _sample = [c for e in manifest.get("pairs", []) for c in e.get("clouds", [])][:6]
        if _sample:
            hist = collections.Counter()
            for _c in _sample:
                with _laspy.open(_c) as _f:
                    hist.update(np.asarray(_f.read().classification).tolist())
            tot = max(sum(hist.values()), 1)
            top = ", ".join(f"{k}:{100.0*v/tot:.1f}%"
                            for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:8])
            gc = tuple(getattr(cfg, "ground_classes", (2, 9)))
            uc = tuple(getattr(cfg, "unclassified_classes", (0, 1)))
            fg = sum(v for k, v in hist.items() if k in gc) / tot
            fi = sum(v for k, v in hist.items() if k in uc) / tot
            fn = 1.0 - fg - fi
            print(f"[04] raw class histogram ({len(_sample)} sampled clouds): {top}")
            print(f"[04] mapped with ground={gc} ignore={uc}: ground {100*fg:.1f}%  "
                  f"non-ground {100*fn:.1f}%  IGNORE {100*fi:.1f}%")
            if fi > 0.30:
                print(f"[04] *** WARNING: {100*fi:.0f}% of points map to IGNORE -- excluded from the "
                      f"loss AND metrics. If class 1 means 'non-ground' in this dataset (British EA "
                      f"convention: only ground is classified), re-run stage 04 with "
                      f"--unclassified-classes 0 so class 1 supervises as NON-GROUND. ***")
            if fn < 0.05:
                print(f"[04] *** WARNING: only {100*fn:.1f}% of points supervise as NON-GROUND -- a "
                      f"model trained on this degenerates to predicting ground everywhere. ***")
    except Exception as _e:
        print(f"[04] (class-histogram preview skipped: {type(_e).__name__}: {_e})")

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

    # ------------------------------------------------------------------ #
    #  Fail-fast manifest diagnostics, split assignment, dispatch, stats.
    #  (Reconstructed 2026-07-10: the shipped file was truncated here --
    #  it built `jobs` and then fell off the end, so stage 04 exited
    #  silently having written nothing. This tail restores the pipeline
    #  and makes every previously-silent branch loud.)
    # ------------------------------------------------------------------ #
    n_pairs = len(manifest.get("pairs", []))
    print(f"[04] manifest: {n_pairs} pairs, {len(jobs)} clouds -> out {out_dir}")
    if not jobs:
        sys.exit("[04] ERROR: the manifest lists NO clouds (manifest['pairs'][*]['clouds'] "
                 "is empty). Re-run stage 02 (or 01) with the correct --project-dir; "
                 f"manifest inspected: {man_path or args.input_dir}")
    missing = [c for c, _, _, _ in jobs if not os.path.exists(c)]
    if missing:
        print(f"[04] WARNING: {len(missing)}/{len(jobs)} manifest cloud paths do not exist "
              f"(first: {missing[0]!r}). The manifest stores SOURCE paths as found by stage "
              f"02 (e.g. under your --project-dir). If the folder moved/renamed, re-run stage 02.")
    if len(missing) == len(jobs):
        sys.exit("[04] ERROR: none of the manifest cloud paths exist on this machine. Aborting.")

    splits = _assign_splits(len(jobs), cfg, seed=0)
    tasks = [(c, pp, rp, str(splits[k]), out_dir, cfg, 1000 + k)
             for k, (c, pp, rp, _pid) in enumerate(jobs)]

    workers = max(1, int(args.workers)) if args.workers else max(1, (os.cpu_count() or 2) - 1)
    print(f"[04] preprocessing {len(tasks)} clouds with {workers} workers "
          f"(ground={tuple(cfg.ground_classes)} ignore={tuple(cfg.unclassified_classes)} "
          f"dl={cfg.first_subsampling_dl}) ...")

    results = []
    if workers <= 1:
        for k, t in enumerate(tasks):
            results.append(_preprocess_one(t))
            name, split, n, has_prior, failed = results[-1]
            print(f"[04] ({k + 1}/{len(tasks)}) {name}: "
                  f"{'FAILED/MISSING' if failed else f'n={n}'} split={split} "
                  f"prior={'yes' if has_prior else 'NO'}")
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")            # Windows-safe; matches historical behaviour
        with ctx.Pool(processes=workers) as pool:
            # tasks stay pair-ordered so each worker's raster cache sees runs of
            # clouds sharing one prior (see the per-worker cache note above).
            for k, res in enumerate(pool.imap(_preprocess_one, tasks, chunksize=1)):
                results.append(res)
                name, split, n, has_prior, failed = res
                print(f"[04] ({k + 1}/{len(tasks)}) {name}: "
                      f"{'FAILED/MISSING' if failed else f'n={n}'} split={split} "
                      f"prior={'yes' if has_prior else 'NO'}", flush=True)

    ok = [r for r in results if not r[4]]
    bad = [r for r in results if r[4]]
    from collections import Counter
    by_split = Counter(r[1] for r in ok)
    print(f"[04] done: {len(ok)}/{len(results)} clouds tiled "
          f"(train={by_split.get('train', 0)} val={by_split.get('val', 0)} "
          f"test={by_split.get('test', 0)}); {len(bad)} failed/missing"
          + (f" (first: {bad[0][0]})" if bad else ""))
    if not ok:
        sys.exit("[04] ERROR: zero tiles written. Nothing to train on.")

    try:
        stats = compute_norm_stats(out_dir, cfg)
        print(f"[04] wrote {os.path.join(out_dir, 'norm_stats.json')} "
              f"({len(stats.get('mean', []))} feature channels)")
    except Exception as e:
        print(f"[04] norm-stats skipped ({type(e).__name__}: {e}); stage 05 recomputes it in place.")

    print(f"[04] next: python scripts/12_label_audit.py --tiles {out_dir}")


if __name__ == "__main__":
    main()
