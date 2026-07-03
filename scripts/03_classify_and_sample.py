#!/usr/bin/env python3
"""
03 - Regional distribution report for the preprocessed corpus.

MEEPO here samples spheres UNIFORMLY at random (regional diversity comes
from stage 01's round-robin interleave over areas), so there is no scene-type
taxonomy to report. This stage instead summarises how the candidate spheres are
spread across SOURCE CLOUDS / REGIONS and train/val/test splits, which is the
distribution the trainer actually sees.

    python scripts/03_classify_and_sample.py --tile-dir data/nz/tiles
    python scripts/03_classify_and_sample.py --root data/nz          # uses <root>/tiles
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meepo_nz.utils.config import Config


def _region_of(fname: str) -> str:
    """Coarse region/survey key from a tile filename (strip trailing tile id and
    year suffixes) so spheres can be grouped by the area they came from."""
    base = os.path.splitext(os.path.basename(fname))[0]
    base = re.split(r"[_-]?\d{3,}", base)[0]          # drop long numeric tile ids
    base = re.sub(r"(_|-)?(19|20)\d{2}.*$", "", base)  # drop a year suffix if present
    return base.strip("_-") or base


def _bar(n, total, width=40):
    return "#" * int(width * n / max(total, 1))


def main():
    ap = argparse.ArgumentParser(description="Regional distribution of preprocessed tiles.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None, help="Data root (uses <root>/tiles).")
    ap.add_argument("--tile-dir", default=None, help="Preprocessed tile dir (step 04).")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    tile_dir = args.tile_dir or os.path.join(args.root or cfg.data_root, "tiles")
    files = sorted(glob.glob(os.path.join(tile_dir, "*.npz")))
    if not files:
        print(f"[03] no .npz tiles in {tile_dir} - run stage 04 first.")
        return

    by_region_tiles = defaultdict(int)
    by_region_spheres = defaultdict(int)
    by_split_tiles = defaultdict(int)
    by_split_spheres = defaultdict(int)
    n_spheres = 0
    for f in files:
        try:
            with np.load(f, allow_pickle=True) as d:
                split = str(d["split"]) if "split" in d else "train"
                ncc = int(d["centers"].shape[0]) if "centers" in d else 0
        except Exception:
            continue
        reg = _region_of(f)
        by_region_tiles[reg] += 1
        by_region_spheres[reg] += ncc
        by_split_tiles[split] += 1
        by_split_spheres[split] += ncc
        n_spheres += ncc

    n_tiles = len(files)
    print("\n========== corpus regional distribution ==========")
    print(f"tiles            : {n_tiles}")
    print(f"candidate spheres: {n_spheres}")
    print(f"regions          : {len(by_region_tiles)}")
    print("--------------------------------------------------")
    print(f"{'region':<24}{'tiles':>7}{'spheres':>9}{'  share':>8}")
    for reg in sorted(by_region_spheres, key=lambda r: -by_region_spheres[r]):
        sp = by_region_spheres[reg]
        print(f"  {reg:<22}{by_region_tiles[reg]:>7}{sp:>9}  {100.0*sp/max(n_spheres,1):5.1f}%  {_bar(sp, n_spheres, 24)}")
    print("--------------------------------------------------")
    for sp_name in ("train", "val", "test"):
        if sp_name in by_split_tiles:
            print(f"  {sp_name:<6} tiles={by_split_tiles[sp_name]:<5} spheres={by_split_spheres[sp_name]}")
    print("==================================================")
    print("[03] sampling = UNIFORM over all spheres (regional diversity from stage 01).")
    print("     Set use_region_balanced_sampler=True to weight every region equally.")


if __name__ == "__main__":
    main()
