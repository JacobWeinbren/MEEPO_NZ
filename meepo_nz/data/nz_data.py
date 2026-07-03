"""
New Zealand LiDAR discovery + download.

IMPORTANT - where NZ point clouds actually live
------------------------------------------------
The public ``s3://nz-elevation`` bucket (LINZ open data on AWS) contains **only
DEM / DSM rasters** - its products are ``dem`` and ``dsm`` (see LINZ
``docs/naming.md``); it has **no point clouds**.  NZ LiDAR *point clouds* are
distributed by **OpenTopography** (and the LINZ Data Service via Koordinates,
which needs an account).  OpenTopography exposes every hosted dataset through an
anonymous, S3-compatible **bulk mirror** - no API key required:

    endpoint : https://opentopography.s3.sdsc.edu
    bucket   : pc-bulk
    example  : s3://pc-bulk/NZ24_North/   (Northland, New Zealand 2024)

So this module's **primary** source is OpenTopography bulk S3 for the point
clouds, fetched in **year pairs** (a survey + the most recent earlier survey of
the same area).  The previous-year cloud is turned into the DTM raster (the
single sanctioned deviation) by ``scripts/02_build_dtm.py`` from its ground
returns.  The ``nz-elevation`` bucket is offered only as an *alternative DTM
source* (its 1 m bare-earth DEM), never as a point-cloud source.

Nothing here is exercised in the build sandbox (no AWS / OpenTopography egress);
it runs on the target server.  Use ``scripts/01_download_data.py --list-only``
to verify discovery before downloading.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Tuple
import random
import struct
import zlib

# ---- OpenTopography bulk S3 (anonymous) ----
OT_ENDPOINT = "https://opentopography.s3.sdsc.edu"
OT_BUCKET = "pc-bulk"

# ---- nz-elevation (DEM/DSM rasters only; optional DTM source) ----
NZ_ELEV_BUCKET = "nz-elevation"
NZ_ELEV_REGION = "ap-southeast-2"


# ======================================================================== #
#  OpenTopography bulk S3 - the point-cloud source (no API key)
# ======================================================================== #
def _ot_s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
    return boto3.client(
        "s3", endpoint_url=OT_ENDPOINT, region_name="us-east-1",
        config=BotoConfig(signature_version=UNSIGNED,
                          s3={"addressing_style": "path"},
                          connect_timeout=30, read_timeout=120,
                          retries={"max_attempts": 10, "mode": "adaptive"},
                          max_pool_connections=32),
    )

    # NOTE: a transient S3 read-timeout on a bulk LISTING or a dropped download
    # stream must never abort a multi-hour download. The longer timeouts + adaptive
    # retries above (botocore-internal) plus the _s3_retry outer guard below make
    # listing/header/download calls resilient; the download is resumable, so a
    # re-run just skips files already on disk.


def _s3_retry(fn, *a, _tries=6, _base=2.0, **k):
    """Call an S3 op with bounded exponential-backoff retries on transient network
    errors (read/connect timeouts, dropped connections, throttling, 5xx). This is an
    OUTER guard on top of botocore's own retries so a slow bulk-listing or a stalled
    stream cannot abort the whole download."""
    import time
    from botocore.exceptions import (ClientError, EndpointConnectionError,
                                      ConnectionClosedError, ReadTimeoutError,
                                      ConnectTimeoutError)
    transient = (EndpointConnectionError, ConnectionClosedError,
                 ReadTimeoutError, ConnectTimeoutError)
    for i in range(int(_tries)):
        try:
            return fn(*a, **k)
        except transient as e:
            if i == _tries - 1:
                raise
            wait = _base * (2 ** i)
            print(f"[ot]   transient S3 error ({type(e).__name__}); "
                  f"retry {i+1}/{_tries-1} in {wait:.0f}s", flush=True)
            time.sleep(wait)
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in ("500", "503", "SlowDown", "RequestTimeout", "InternalError") and i < _tries - 1:
                wait = _base * (2 ** i)
                print(f"[ot]   S3 {code}; retry {i+1}/{_tries-1} in {wait:.0f}s", flush=True)
                time.sleep(wait)
            else:
                raise


def _list_prefixes(s3, bucket: str, prefix: str = "") -> List[str]:
    out, token = [], None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, Delimiter="/")
        if token:
            kw["ContinuationToken"] = token
        resp = _s3_retry(s3.list_objects_v2, **kw)
        out += [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return out


def _list_laz(s3, bucket: str, prefix: str, suffixes=(".laz", ".las")) -> List[Dict]:
    out, token = [], None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kw["ContinuationToken"] = token
        resp = _s3_retry(s3.list_objects_v2, **kw)
        for obj in resp.get("Contents", []):
            if obj["Key"].lower().endswith(suffixes):
                out.append({"Key": obj["Key"], "Size": int(obj["Size"])})
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return out


_RANGE_RE = re.compile(r"(19|20)\d{2}-(19|20)\d{2}")
_YEAR_RE = re.compile(r"(19|20)\d{2}")
_NZYY_RE = re.compile(r"NZ(\d{2})", re.IGNORECASE)


def _parse_year_and_area(prefix_name: str) -> Tuple[Optional[int], str]:
    """From a pc-bulk dataset folder name, parse (start_year, area-key).

    The area key is the name with the year token removed and normalised, so that
    different-year captures of the same area share a key (and thus pair up).
    e.g. 'NZ24_North' -> (2024, 'north'); 'canterbury_2016-2018' -> (2016,
    'canterbury'); a later 'canterbury_2021' -> (2021, 'canterbury') -> pairs.
    """
    name = prefix_name.strip("/")
    year, token = None, None
    m = _RANGE_RE.search(name)                       # e.g. 2016-2018 -> start 2016
    if m:
        year, token = int(m.group(0)[:4]), m.group(0)
    else:
        m = _YEAR_RE.search(name)
        if m:
            year, token = int(m.group(0)), m.group(0)
        else:
            m2 = _NZYY_RE.search(name)
            if m2:
                year, token = 2000 + int(m2.group(1)), m2.group(0)
    stem = name.replace(token, " ") if token else name
    stem = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    stem = re.sub(r"_+", "_", stem)
    return year, (stem or name.lower())


def discover_opentopography(s3=None) -> Dict[str, List[Tuple[int, str]]]:
    """Enumerate NZ datasets on the OpenTopography bulk mirror, grouped by area.

    Returns ``{area_key: [(year, dataset_prefix), ...]}`` sorted by year.
    Only datasets whose folder name looks like New Zealand (contains 'nz') are
    kept; tune via the prefix filter if needed.
    """
    s3 = s3 or _ot_s3()
    groups: Dict[str, List[Tuple[int, str]]] = {}
    for pref in _list_prefixes(s3, OT_BUCKET, ""):
        name = pref.strip("/")
        if "nz" not in name.lower():
            continue
        year, area = _parse_year_and_area(name)
        if year is None:
            continue
        groups.setdefault(area, []).append((year, pref))
    for area in groups:
        groups[area].sort(key=lambda t: t[0])
    return groups


def pair_opentopography(groups: Dict[str, List[Tuple[int, str]]]) -> List[Dict]:
    """Form year-pairs (current + most-recent-earlier) within each area."""
    pairs = []
    for area, items in groups.items():
        for i in range(1, len(items)):
            prev_year, prev_pref = items[i - 1]
            cur_year, cur_pref = items[i]
            if cur_year == prev_year:
                continue
            pairs.append({"area": area, "cur_year": cur_year, "prev_year": prev_year,
                          "cur_prefix": cur_pref, "prev_prefix": prev_pref})
    return pairs


def _las_header_density(s3, bucket: str, key: str) -> Optional[float]:
    """Point density (pts/m^2) read from a LAS/LAZ **public header only**.

    A single 512-byte range request gives the point count and the X/Y bounding
    box; density = count / bbox-area. The LAS public header block is uncompressed
    even inside a LAZ file, so no points are decompressed and the tile is never
    fully downloaded. Field offsets per the ASPRS LAS 1.1-1.4 spec: Max/Min X,Y
    are 4 doubles at byte 179; the point count is a uint64 at byte 247 (LAS 1.4)
    or the legacy uint32 at byte 107 (<=1.3).
    """
    try:
        obj = _s3_retry(s3.get_object, Bucket=bucket, Key=key, Range="bytes=0-511")
        buf = obj["Body"].read(512)
        if len(buf) < 256 or buf[0:4] != b"LASF":
            return None
        ver_minor = buf[25]
        maxx, minx, maxy, miny = struct.unpack_from("<4d", buf, 179)
        if ver_minor >= 4:
            n = struct.unpack_from("<Q", buf, 247)[0]
            if n == 0:                                   # some writers leave 1.4 count only in legacy
                n = struct.unpack_from("<I", buf, 107)[0]
        else:
            n = struct.unpack_from("<I", buf, 107)[0]
        area = (maxx - minx) * (maxy - miny)
        return (n / area) if (area > 0 and n > 0) else None
    except Exception:
        return None


def _estimate_dataset_density(s3, bucket: str, prefix: str,
                              n_sample: int = 6) -> Tuple[Optional[float], int]:
    """Median per-tile density for a dataset prefix (samples up to ``n_sample``
    tiles via header reads). Returns (median_density_or_None, n_tiles).

    The sample is chosen with an RNG seeded from the prefix (a stable CRC32, not
    Python's salted hash), so a given dataset's estimate is identical across runs
    and independent of how many other datasets were scanned first.
    """
    keys = _list_laz(s3, bucket, prefix)
    if not keys:
        return None, 0
    keys = sorted(keys, key=lambda k: k["Key"])
    rng = random.Random(zlib.crc32(prefix.encode("utf-8")))
    sample = keys if len(keys) <= n_sample else rng.sample(keys, n_sample)
    ds = [d for d in (_las_header_density(s3, bucket, k["Key"]) for k in sample) if d is not None]
    if not ds:
        return None, len(keys)
    return float(median(ds)), len(keys)


def download_opentopography(cfg, out_root: str, list_only: bool = False, workers: Optional[int] = None) -> Dict:
    """Download regionally-diverse year-pairs of point clouds from OpenTopography.

    Writes a manifest whose pairs carry the **current**-year clouds (training
    input) and the **previous**-year clouds (turned into the DTM by step 02).
    """
    s3 = _ot_s3()
    budget_bytes = int(cfg.download_budget_gb * 1e9)   # decimal GB, matches the display
    used = 0
    raw_dir = os.path.join(out_root, "raw")
    prev_dir = os.path.join(out_root, "raw_prev")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    groups = discover_opentopography(s3)
    pairs = pair_opentopography(groups)
    print(f"[ot] {sum(len(v) for v in groups.values())} NZ datasets in "
          f"{len(groups)} areas -> {len(pairs)} year-pairs")

    # ---- PHASE 0: keep post-<min_year> pairs whose CURRENT (training) capture
    # sits in the wanted point-density band. Only year-pairs are considered, so
    # every training tile has a previous-survey DTM twin. Density is read from
    # each candidate dataset's LAS headers (a 512-byte range request per sampled
    # tile -> point count / bbox area); no points are decompressed and no tile is
    # fully downloaded, so the scan is cheap even over the whole catalogue.
    min_year = int(getattr(cfg, "ot_min_year", 2020))
    dmin = float(getattr(cfg, "ot_min_density", 2.0))
    dmax = float(getattr(cfg, "ot_max_density", 5.0))
    n_sample = int(getattr(cfg, "ot_density_sample_tiles", 6))
    regions = [r.lower() for r in (getattr(cfg, "regions", None) or []) if r]

    cand = [p for p in pairs if int(p["cur_year"]) >= min_year]
    if regions:
        cand = [p for p in cand if any(r in p["area"] for r in regions)]
    print(f"[ot] scanning {len(cand)} pairs with current year >= {min_year}"
          + (f" and region in {regions}" if regions else "")
          + f" for density {dmin}-{dmax} pts/m^2 (LAS-header sampling, {n_sample} tiles/dataset) ...")

    dens_cache: Dict[str, Optional[float]] = {}
    kept = []
    for p in sorted(cand, key=lambda q: q["cur_prefix"]):
        pref = p["cur_prefix"]
        if pref not in dens_cache:
            dens_cache[pref] = _estimate_dataset_density(s3, OT_BUCKET, pref, n_sample)[0]
        d = dens_cache[pref]
        ok = (d is not None) and (dmin <= d <= dmax)
        print(f"[ot]   [{'KEEP' if ok else 'skip'}] {p['area']} "
              f"{p['prev_year']}->{p['cur_year']}  "
              f"density={'n/a' if d is None else f'{d:.2f}'} pts/m^2")
        if ok:
            kept.append(p)
    pairs = kept
    print(f"[ot] selected {len(pairs)} pairs in {dmin}-{dmax} pts/m^2 "
          f"(current capture post-{min_year}, each with a previous-survey DTM twin)")
    if not pairs:
        print("[ot] WARNING: no datasets matched. Many NZ captures are denser than "
              f"{dmax} pts/m^2 - widen --max-density (or lower --min-year) and re-scan.")

    # ---- PHASE 1: build a budget-bounded, geographically-diverse plan ----
    # Round-robin over pairs (interleaved across areas), grabbing the NEXT
    # `tiles_per_visit` tiles from each pair every pass, until the BUDGET is
    # filled or every prefix is exhausted. Planning is cheap (listing + sizes,
    # no transfers); the transfers themselves run in parallel in PHASE 2.
    tiles_per_visit = max(int(getattr(cfg, "tiles_per_pair", 6)), 1)
    manifest = {"source": "opentopography", "pairs": []}

    key_cache: Dict[str, List[Dict]] = {}
    def _keys(prefix):
        if prefix not in key_cache:
            key_cache[prefix] = _list_laz(s3, OT_BUCKET, prefix)
        return key_cache[prefix]

    by_area: Dict[str, List[Dict]] = {}
    for p in pairs:
        by_area.setdefault(p["area"], []).append(p)
    from itertools import zip_longest
    interleaved = [p for grp in zip_longest(*by_area.values()) for p in grp if p is not None]

    plan: List[Tuple[str, str]] = []          # (s3_key, dest_path) to fetch
    offset = {id(p): 0 for p in interleaved}
    entry_of: Dict[int, Dict] = {}

    # Spatially MATCH current<->previous tiles so the previous-year DTM actually covers
    # the current tiles. LINZ tiles share a fixed grid across years, so the filename with
    # the survey year removed is a stable spatial key (CL2_BD33_2023_1000_1017 ->
    # CL2_BD33_1000_1017). We only ever fetch a current tile that has a previous-year twin
    # (and we fetch that twin -> DTM), guaranteeing per-tile overlap. The old code sliced
    # both lists by the SAME index, pairing spatially-unrelated tiles -> ~16% overlap.
    def _spatial_key(key, year):
        return os.path.basename(key).replace(f"_{int(year)}_", "_", 1)
    matched: Dict[int, List[Tuple[Dict, Dict]]] = {}
    for p in interleaved:
        prev_by_sk: Dict[str, Dict] = {}
        for k in _keys(p["prev_prefix"]):
            prev_by_sk.setdefault(_spatial_key(k["Key"], p["prev_year"]), k)
        mp = []
        for k in _keys(p["cur_prefix"]):
            pk = prev_by_sk.get(_spatial_key(k["Key"], p["cur_year"]))
            if pk is not None:
                mp.append((k, pk))
        matched[id(p)] = mp
    n_match = sum(len(v) for v in matched.values())
    n_overlap_pairs = sum(1 for v in matched.values() if v)
    print(f"[ot] spatially-matched current<->prev tiles: {n_match} across "
          f"{n_overlap_pairs}/{len(interleaved)} pairs (every fetched current tile has a prev-year DTM twin)")

    progressing = True
    while progressing and used < budget_bytes:
        progressing = False
        for p in interleaved:
            if used >= budget_bytes:
                break
            off = offset[id(p)]
            chunk = matched[id(p)][off:off + tiles_per_visit]
            offset[id(p)] = off + tiles_per_visit
            took_cur, took_prev = [], []
            for ck, pk in chunk:                # current tile + its previous-year twin
                if used + ck["Size"] + pk["Size"] > budget_bytes:
                    break
                cdest = os.path.join(raw_dir, p["area"], str(p["cur_year"]), os.path.basename(ck["Key"]))
                pdest = os.path.join(prev_dir, p["area"], str(p["prev_year"]), os.path.basename(pk["Key"]))
                plan.append((ck["Key"], cdest)); took_cur.append(cdest)
                plan.append((pk["Key"], pdest)); took_prev.append(pdest)
                used += ck["Size"] + pk["Size"]
            if took_cur or took_prev:
                progressing = True
                e = entry_of.get(id(p))
                if e is None:
                    e = {"region": p["area"], "cur_year": p["cur_year"],
                         "prev_year": p["prev_year"], "clouds": [], "prev_clouds": [], "dtms": []}
                    entry_of[id(p)] = e
                    manifest["pairs"].append(e)
                e["clouds"].extend(took_cur)
                e["prev_clouds"].extend(took_prev)

    n_cur = sum(len(e["clouds"]) for e in manifest["pairs"])
    n_prev = sum(len(e["prev_clouds"]) for e in manifest["pairs"])
    print(f"[ot] plan: {len(plan)} files ({n_cur} current + {n_prev} previous) "
          f"= {used/1e9:.2f}/{cfg.download_budget_gb} GB across {len(manifest['pairs'])} pairs")
    if not manifest["pairs"]:
        print("[ot] WARNING: no point clouds selected. Check connectivity and that "
              "OpenTopography NZ datasets are reachable at the bulk endpoint.")

    # ---- PHASE 2: download the plan in parallel streams ----
    if not list_only and plan:
        for _, dest in plan:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        n_workers = max(int(workers if workers else getattr(cfg, "download_workers", 16)), 1)
        import concurrent.futures as _cf
        import threading as _th
        _tls = _th.local()
        def _client():
            c = getattr(_tls, "c", None)
            if c is None:
                c = _ot_s3()                    # one anonymous client per thread (thread-safe)
                _tls.c = c
            return c
        done = {"n": 0, "bytes": 0}
        lock = _th.Lock()
        n_total = len(plan)
        def _dl(item):
            key, dest = item
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                got = os.path.getsize(dest)     # resume: skip a file already on disk
            else:
                tmp = dest + ".part"
                _s3_retry(_client().download_file, OT_BUCKET, key, tmp)
                os.replace(tmp, dest)           # atomic: no truncated file on crash
                got = os.path.getsize(dest) if os.path.exists(dest) else 0
            with lock:
                done["n"] += 1
                done["bytes"] += got
                if done["n"] % 20 == 0 or done["n"] == n_total:
                    print(f"[ot]   downloaded {done['n']}/{n_total} files "
                          f"({done['bytes']/1e9:.2f} GB)", flush=True)
        print(f"[ot] downloading with {n_workers} parallel streams ...")
        with _cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_dl, plan))

    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[ot] total {used/1e9:.2f} GB across {len(manifest['pairs'])} pairs "
          f"-> {os.path.join(out_root, 'manifest.json')}")
    return manifest


# ======================================================================== #
#  nz-elevation - DEM/DSM rasters ONLY (optional DTM source, no point clouds)
# ======================================================================== #
def _nz_elev_s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
    return boto3.client("s3", region_name=NZ_ELEV_REGION,
                        config=BotoConfig(signature_version=UNSIGNED))


@dataclass
class DemSurvey:
    region: str
    name: str
    prefix: str            # <region>/<survey>/
    year: int
    dem_prefix: Optional[str] = None     # <region>/<survey>/dem_1m/<crs>/


def _nz_list_prefixes(s3, prefix: str) -> List[str]:
    out, token = [], None
    while True:
        kw = dict(Bucket=NZ_ELEV_BUCKET, Prefix=prefix, Delimiter="/")
        if token:
            kw["ContinuationToken"] = token
        resp = _s3_retry(s3.list_objects_v2, **kw)
        out += [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return out


def _nz_year_from_collection(s3, dem_prefix: str) -> Optional[int]:
    """Read dem_1m/<crs>/collection.json temporal extent -> start year."""
    try:
        obj = s3.get_object(Bucket=NZ_ELEV_BUCKET, Key=dem_prefix + "collection.json")
        coll = json.loads(obj["Body"].read())
        ext = coll["extent"]["temporal"]["interval"][0][0]
        return int(str(ext)[:4])
    except Exception:
        return None


def discover_dem_surveys(s3, region: str) -> List[DemSurvey]:
    """Enumerate nz-elevation DEM surveys for a region (path: region/survey/dem_1m/crs/)."""
    surveys: List[DemSurvey] = []
    region_prefix = f"{region}/"
    for survey_prefix in _nz_list_prefixes(s3, region_prefix):
        name = survey_prefix[len(region_prefix):].strip("/")
        dem_prefix = None
        for prod in _nz_list_prefixes(s3, survey_prefix):
            if "dem" in prod.rsplit("/", 2)[-2]:           # <survey>/dem_1m/
                crs = _nz_list_prefixes(s3, prod)
                dem_prefix = crs[0] if crs else prod
                break
        # year: prefer the collection.json, else a 4-digit year in the survey name
        year = _nz_year_from_collection(s3, dem_prefix) if dem_prefix else None
        if year is None:
            m = _YEAR_RE.search(name)
            year = int(m.group(0)) if m else 0
        surveys.append(DemSurvey(region=region, name=name, prefix=survey_prefix,
                                 year=year, dem_prefix=dem_prefix))
    return surveys


def download_nz_elevation(cfg, out_root: str, list_only: bool = False) -> Dict:
    """nz-elevation has DEM/DSM rasters only - NOT a point-cloud source.

    This fetches previous/current **DEM** year-pairs (1 m bare-earth GeoTIFFs)
    that step 02 can mosaic into the previous-year DTM, but it provides **no
    point clouds**, so it cannot be used as the training source on its own.
    """
    print("[nz-elevation] NOTE: this bucket holds DEM/DSM rasters only - it has "
          "NO point clouds. Use --source opentopography for training point clouds.")
    print("[nz-elevation] Fetching DEM year-pairs to serve as the previous-year DTM ...")
    s3 = _nz_elev_s3()
    budget_bytes = int(cfg.download_budget_gb * 1e9)   # decimal GB, matches the display
    used = 0
    dtm_dir = os.path.join(out_root, "dtm_src")
    os.makedirs(dtm_dir, exist_ok=True)
    manifest = {"source": "nz_elevation_dem", "pairs": []}

    for region in cfg.regions:
        try:
            surveys = sorted(discover_dem_surveys(s3, region), key=lambda s: s.year)
        except Exception as e:
            print(f"[nz-elevation] {region}: discovery failed ({e})")
            continue
        surveys = [s for s in surveys if s.dem_prefix]
        print(f"[nz-elevation] {region}: {len(surveys)} DEM surveys")
        for i in range(1, len(surveys)):
            prev, cur = surveys[i - 1], surveys[i]
            if cur.year == prev.year or used >= budget_bytes:
                continue
            keys = _list_laz(s3, NZ_ELEV_BUCKET, prev.dem_prefix,
                             suffixes=(".tiff", ".tif"))[:8]
            dtms = []
            for k in keys:
                if used + k["Size"] > budget_bytes:
                    break
                dest = os.path.join(dtm_dir, region, str(prev.year), os.path.basename(k["Key"]))
                dtms.append(dest)
                if not list_only:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    s3.download_file(NZ_ELEV_BUCKET, k["Key"], dest)
                used += k["Size"]
            if dtms:
                manifest["pairs"].append({"region": region, "cur_year": cur.year,
                                          "prev_year": prev.year, "clouds": [],
                                          "prev_clouds": [], "dtms": dtms})
            print(f"[nz-elevation] {region} {prev.year}->{cur.year}: {len(dtms)} DEM tiles, "
                  f"used={used/1e9:.2f}/{cfg.download_budget_gb} GB")

    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[nz-elevation] wrote {os.path.join(out_root, 'manifest.json')} "
          f"(DEM-only; no point clouds -> not trainable alone)")
    return manifest
