"""Build an in-memory manifest from a plain folder of LAS/LAZ files.

Stages 02 (prior raster) and 04 (preprocess) are normally driven by the
``manifest.json`` that stage 01 writes after an OpenTopography download. This
helper lets those stages run on user-supplied clouds instead: point them at a
folder and it discovers the ``.las`` / ``.laz`` files and emits the same manifest
structure (a list of ``pairs`` each holding ``clouds``), so nothing downstream
changes.

Previous-year prior (optional): if a ``--prev-dir`` is given, each current cloud
is matched to a previous-year twin by identical file stem. Only twinned clouds
get an index-aligned ``prev_clouds`` list (the contract stage 02 relies on to
build the 5-channel prior); untwinned clouds are processed with no prior (their
prev-DTM feature channel is zero-filled at load, so the feature dimension is
unchanged).
"""
from __future__ import annotations

import glob
import os


def discover_clouds(input_dir):
    """Recursively find .las/.laz (case-insensitive), sorted and de-duplicated."""
    if not input_dir or not os.path.isdir(input_dir):
        raise FileNotFoundError(f"--input-dir is not a directory: {input_dir!r}")
    seen, out = set(), []
    for pat in ("*.las", "*.laz", "*.LAS", "*.LAZ"):
        for p in glob.glob(os.path.join(input_dir, "**", pat), recursive=True):
            rp = os.path.realpath(p)
            if rp not in seen and os.path.isfile(p):
                seen.add(rp)
                out.append(p)
    return sorted(out)


def _match_prev(clouds, prev_dir):
    """Return a list aligned with ``clouds`` of previous-year twin paths (or None),
    matched by identical file stem. None if no --prev-dir or nothing matched."""
    if not prev_dir:
        return None
    by_stem = {}
    for p in discover_clouds(prev_dir):
        by_stem.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    twins = [by_stem.get(os.path.splitext(os.path.basename(c))[0]) for c in clouds]
    return twins if any(t is not None for t in twins) else None


def match_rasters(clouds, raster_dir):
    """Return a list aligned with ``clouds`` of pre-made raster paths (or None), matched
    by identical file stem. Searches common raster/grid extensions recursively."""
    if not raster_dir or not os.path.isdir(raster_dir):
        raise FileNotFoundError(f"--raster-dir is not a directory: {raster_dir!r}")
    by_stem, seen = {}, set()
    for pat in ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.asc", "*.ASC",
                "*.img", "*.IMG", "*.vrt", "*.VRT", "*.npz", "*.npy"):
        for p in glob.glob(os.path.join(raster_dir, "**", pat), recursive=True):
            rp = os.path.realpath(p)
            if rp in seen or not os.path.isfile(p):
                continue
            seen.add(rp)
            by_stem.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return [by_stem.get(os.path.splitext(os.path.basename(c))[0]) for c in clouds]


def build_folder_manifest(input_dir, prev_dir=None, region=None):
    """Synthesize a manifest dict from a folder. Twinned clouds (when --prev-dir is
    given) go in a pair with an aligned ``prev_clouds``; untwinned clouds go in a
    separate no-prior pair."""
    clouds = discover_clouds(input_dir)
    if not clouds:
        raise FileNotFoundError(f"no .las/.laz files found under {input_dir!r}")
    region = region or os.path.basename(os.path.normpath(input_dir)) or "folder"

    twins = _match_prev(clouds, prev_dir)
    if twins is None:
        return {"pairs": [{"clouds": clouds, "region": region, "prev_year": "none"}]}

    twinned = [(c, t) for c, t in zip(clouds, twins) if t is not None]
    untwinned = [c for c, t in zip(clouds, twins) if t is None]
    pairs = []
    if twinned:
        cc, pp = zip(*twinned)
        pairs.append({"clouds": list(cc), "prev_clouds": list(pp),
                      "region": region, "prev_year": "prev"})
    if untwinned:
        pairs.append({"clouds": untwinned, "region": region, "prev_year": "none"})
    return {"pairs": pairs}
