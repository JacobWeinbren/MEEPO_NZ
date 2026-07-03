#!/usr/bin/env python3
"""
01 - Download New Zealand LiDAR year-pairs.

NZ LiDAR **point clouds** come from OpenTopography's anonymous bulk S3 mirror
(no API key). The ``nz-elevation`` AWS bucket holds **DEM/DSM rasters only** and
is offered solely as an alternative previous-year DTM source. Each pair is a
survey plus the most recent earlier survey of the same area; the previous-year
cloud becomes the DTM (the one deviation) in step 02.

Examples
--------
    # see what WOULD be downloaded (recommended first), no files fetched
    python scripts/01_download_data.py --out data/nz --list-only

    # download ~40 GB of point clouds from OpenTopography (no key needed)
    python scripts/01_download_data.py --out data/nz --budget-gb 40

    # DEM-only (previous-year DTM rasters from nz-elevation; NOT trainable alone)
    python scripts/01_download_data.py --out data/nz --source nz_elevation_dem
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meepo_nz.utils.config import Config
from meepo_nz.data.nz_data import download_opentopography, download_nz_elevation
from meepo_nz.data.es_data import download_pnoa_es


def main():
    ap = argparse.ArgumentParser(description="Download NZ LiDAR year-pairs.")
    ap.add_argument("--config", default=None, help="YAML config (optional).")
    ap.add_argument("--out", default=None, help="Output root (default: <data_root>/nz).")
    ap.add_argument("--budget-gb", type=float, default=None, help="Download budget in GB.")
    ap.add_argument("--source", default=None,
                    choices=["pnoa_es", "opentopography", "nz_elevation_dem"],
                    help="Point clouds: pnoa_es (Spanish PNOA, default) or "
                         "opentopography (NZ). DEM-only DTM: nz_elevation_dem.")
    ap.add_argument("--regions", nargs="*", default=None, help="Override region list.")
    ap.add_argument("--min-year", type=int, default=None,
                    help="opentopography: only consider pairs whose current capture is >= this year (default 2020).")
    ap.add_argument("--min-density", type=float, default=None,
                    help="opentopography: minimum point density to keep (pts/m^2, default 2.0). Band 2-9 spans a wide regional range.")
    ap.add_argument("--max-density", type=float, default=None,
                    help="opentopography: maximum point density to keep (pts/m^2, default 9.0).")
    ap.add_argument("--list-only", action="store_true",
                    help="Discover + scan density + plan only; do not download files.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel download streams (default: download_workers=16).")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.budget_gb is not None:
        cfg.download_budget_gb = args.budget_gb
    if args.source is not None:
        cfg.download_source = args.source
    if args.regions:
        cfg.regions = args.regions
    if args.min_year is not None:
        cfg.ot_min_year = args.min_year
    if args.min_density is not None:
        cfg.ot_min_density = args.min_density
    if args.max_density is not None:
        cfg.ot_max_density = args.max_density

    out_root = args.out or os.path.join(cfg.data_root, "es" if cfg.download_source == "pnoa_es" else "nz")
    os.makedirs(out_root, exist_ok=True)

    print(f"[01] source={cfg.download_source} budget={cfg.download_budget_gb} GB "
          f"-> {out_root}")

    if cfg.download_source == "pnoa_es":
        manifest = download_pnoa_es(cfg, out_root, list_only=args.list_only,
                                    workers=args.workers)
    elif cfg.download_source == "nz_elevation_dem":
        manifest = download_nz_elevation(cfg, out_root, list_only=args.list_only)
    else:
        manifest = download_opentopography(cfg, out_root, list_only=args.list_only,
                                           workers=args.workers)

    n_pairs = len(manifest.get("pairs", []))
    n_clouds = sum(len(p.get("clouds", [])) for p in manifest.get("pairs", []))
    n_prev = sum(len(p.get("prev_clouds", [])) for p in manifest.get("pairs", []))
    n_dtms = sum(len(p.get("dtms", [])) for p in manifest.get("pairs", []))
    print(f"[01] manifest: {n_pairs} pairs, {n_clouds} clouds, "
          f"{n_prev} prev clouds, {n_dtms} DEM tiles")
    if n_clouds == 0 and cfg.download_source != "nz_elevation_dem":
        print("[01] WARNING: no point clouds were found - the later stages need "
              "point clouds. Re-run with --list-only to inspect discovery.")


if __name__ == "__main__":
    main()
