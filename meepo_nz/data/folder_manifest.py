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


def discover_rasters(raster_dir):
    """All rasterio-readable rasters under a folder (tif/asc/img/vrt), sorted."""
    out, seen = [], set()
    for pat in ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.asc", "*.ASC",
                "*.img", "*.IMG", "*.vrt", "*.VRT"):
        for p in glob.glob(os.path.join(raster_dir, "**", pat), recursive=True):
            rp = os.path.realpath(p)
            if rp not in seen and os.path.isfile(p):
                seen.add(rp)
                out.append(p)
    return sorted(out)


def cloud_bounds(las_path):
    """(xmin, ymin, xmax, ymax) from the LAS/LAZ header only (no point read)."""
    import laspy
    with laspy.open(las_path) as f:
        h = f.header
        return (float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1]))


def match_rasters_spatial(clouds, raster_dir, epsg=None):
    """For each cloud, the LIST of rasters whose extent intersects the cloud's extent
    (matched spatially, NOT by name -- project-wide DTMs rarely share tile names).
    Returns (matches, crs_warnings). Assumes clouds and rasters share one CRS; if
    ``epsg`` is given, rasters reporting a different EPSG are flagged (not dropped)."""
    import rasterio
    rasters = discover_rasters(raster_dir)
    rb, warns = [], []
    for r in rasters:
        with rasterio.open(r) as src:
            bl, bb, br, bt = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
            code = src.crs.to_epsg() if src.crs else None
        if epsg is not None and code is not None and int(code) != int(epsg):
            warns.append(f"{os.path.basename(r)} reports EPSG:{code}, expected {epsg}")
        rb.append((r, (bl, bb, br, bt)))
    matches = []
    for c in clouds:
        x0, y0, x1, y1 = cloud_bounds(c)
        hit = [r for r, (bl, bb, br, bt) in rb
               if (x0 <= br and x1 >= bl and y0 <= bt and y1 >= bb)]
        matches.append(hit)
    return matches, warns


def _find_subdir(project, names, exts):
    """First child dir whose name contains any of ``names`` (case-insensitive) and
    holds files with ``exts``; else the project itself if it holds such files directly."""
    kids = [d for d in sorted(os.listdir(project)) if os.path.isdir(os.path.join(project, d))]
    def has(d):
        return any(glob.glob(os.path.join(d, "**", f"*{e}"), recursive=True) or
                   glob.glob(os.path.join(d, "**", f"*{e.upper()}"), recursive=True) for e in exts)
    for want in names:
        for k in kids:
            d = os.path.join(project, k)
            if want in k.lower() and has(d):
                return d
    for k in kids:                                                # any child with the files
        d = os.path.join(project, k)
        if has(d):
            return d
    return project if has(project) else None


def build_project_manifest(project_dir):
    """Walk a project tree: each subfolder of ``project_dir`` is one survey project
    holding a LAS folder (clouds) and optionally a previous-DTM folder (rasters), e.g.

        TRAINING DATA/
          P_13761_13762/{LAS/, Previous DTM/}
          P_13820/{LAS/, Previous DTM/}

    Returns a manifest with one pair per project; each pair carries ``raster_dir`` for
    stage 02 to spatially match its rasters to its clouds."""
    if not os.path.isdir(project_dir):
        raise FileNotFoundError(f"--project-dir is not a directory: {project_dir!r}")
    pairs = []
    for name in sorted(os.listdir(project_dir)):
        proj = os.path.join(project_dir, name)
        if not os.path.isdir(proj):
            continue
        las_dir = _find_subdir(proj, ("las", "laz", "cloud", "point"), (".las", ".laz"))
        if las_dir is None:
            continue
        clouds = discover_clouds(las_dir)
        if not clouds:
            continue
        rast_dir = _find_subdir(proj, ("dtm", "prev", "prior", "raster"),
                                (".tif", ".tiff", ".asc", ".img", ".vrt"))
        pairs.append({"clouds": clouds, "region": name, "prev_year": "raster",
                      **({"raster_dir": rast_dir} if rast_dir else {})})
    if not pairs:
        raise FileNotFoundError(f"no project subfolders with LAS files under {project_dir!r}")
    return {"pairs": pairs}


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
