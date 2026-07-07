#!/usr/bin/env python3
"""Audit an EXISTING tiles directory (no training, no LAS re-read): per-split tile
counts, mapped-label balance (ground / non-ground / IGNORE), and feature-channel
means from norm_stats.json.

Reads the ``split`` field of each ``*.npz`` and the ``*.labels.npy`` sidecars written
by stage 04. Labels here are the MAPPED training labels (1=ground, 0=non-ground,
2=IGNORE) -- i.e. exactly what the loss and metrics see.

Why this exists: two silent failure modes look like "the model didn't learn" --
 (1) a dataset label convention where non-ground carries ASPRS class 1, which the
     default mapping sends to IGNORE (the loss then supervises ~only ground and the
     model degenerates to predicting ground everywhere), and
 (2) a previous-DTM prior in a different vertical datum than the LAS heights, which
     turns the strongest feature into an active misleader.
Both are visible from the tiles in seconds. Run this BEFORE burning GPU-days.

    python scripts/12_label_audit.py --tiles data/official/tiles
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", required=True, help="tiles directory from stage 04")
    ap.add_argument("--max-tiles", type=int, default=0,
                    help="cap tiles scanned per split (0 = all; labels are strided-sampled "
                         "either way, so even 'all' is fast)")
    args = ap.parse_args()

    npzs = sorted(glob.glob(os.path.join(args.tiles, "*.npz")))
    if not npzs:
        raise SystemExit(f"no *.npz tiles under {args.tiles!r}")

    per = {}
    for p in npzs:
        try:
            with np.load(p, allow_pickle=True) as d:
                split = str(d["split"]) if "split" in d.files else "train"
        except Exception as e:
            print(f"[audit] unreadable {os.path.basename(p)}: {type(e).__name__}: {e}")
            continue
        st = per.setdefault(split, {"tiles": 0, "cnt": np.zeros(3, dtype=np.int64), "pts": 0})
        st["tiles"] += 1
        if args.max_tiles and st["tiles"] > args.max_tiles:
            continue
        lp = p[:-4] + ".labels.npy"
        if not os.path.exists(lp):
            continue
        lab = np.load(lp, mmap_mode="r")
        st["pts"] += int(lab.shape[0])
        step = max(1, lab.shape[0] // 200000)
        st["cnt"] += np.bincount(np.asarray(lab[::step]).ravel().astype(np.int64),
                                 minlength=3)[:3]

    print(f"[audit] {len(npzs)} tiles under {args.tiles}")
    warn_ignore = warn_ng = False
    for split in ("train", "val", "test"):
        if split not in per:
            continue
        st = per[split]
        tot = max(int(st["cnt"].sum()), 1)
        g, ng, ig = st["cnt"][1] / tot, st["cnt"][0] / tot, st["cnt"][2] / tot
        print(f"[audit] {split:5s}: {st['tiles']:4d} tiles  {st['pts']:>12,d} pts  |  "
              f"ground {100*g:5.1f}%   non-ground {100*ng:5.1f}%   IGNORE {100*ig:5.1f}%")
        if split == "train":
            warn_ignore = ig > 0.30
            warn_ng = ng < 0.05
    if warn_ignore:
        print("[audit] *** WARNING: >30% of train points are IGNORE -- excluded from the loss AND "
              "the metrics. If this dataset marks non-ground as ASPRS class 1 (British EA "
              "convention: only ground classified), the model is supervised ~only on ground and "
              "degenerates to all-ground. Fix: re-run stage 04 with --unclassified-classes 0 "
              "(labels are baked into tiles; re-preprocess required). ***")
    elif warn_ng:
        print("[audit] *** WARNING: <5% of train points supervise as NON-GROUND -- expect collapse "
              "toward all-ground predictions. ***")
    else:
        print("[audit] label balance looks supervisable (both classes present in the loss).")

    ns = os.path.join(args.tiles, "norm_stats.json")
    if os.path.exists(ns):
        with open(ns) as fh:
            stt = json.load(fh)
        mean = stt.get("mean")
        if mean:
            print("[audit] feature channel means: [" +
                  ", ".join(f"{float(m):+.2f}" for m in mean) + "]")
            big = [i for i, m in enumerate(mean) if abs(float(m)) > 10.0]
            if big:
                print(f"[audit] *** WARNING: channel(s) {big} have |mean| > 10 -- if one is the "
                      f"z-minus-previous-DTM channel, the prior DTM and LAS heights disagree "
                      f"(vertical datum / units) and the prior misleads. Check the rasters. ***")


if __name__ == "__main__":
    main()
