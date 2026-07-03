"""
Spanish PNOA-LiDAR (CNIG) discovery + download - the training source that
REPLACES New Zealand.

Point clouds come from the public **flai-ai ``open-lidar-data``** S3 bucket
(region ``eu-central-1``, anonymous - no credentials), as COPC tiles in ETRS89 /
UTM 29N (EPSG:25829) - i.e. Galicia / NW Spain: coast, sea cliffs, hills, forest
and towns. CC BY 4.0.

    bucket : open-lidar-data                                  (region eu-central-1, UNSIGNED)
    current: data/ES/CNIG/Lidar_2015-2021_epsg25829/copc/     2.8 pts/m^2  (training clouds)
    prior  : data/ES/CNIG/Lidar_2008-2015_epsg25829/copc/     0.9 pts/m^2  (previous-survey DTM)
    tile   : PNOA_<year>_<region>_<E>-<N>_ORT-...copc.laz      (E,N = UTM29N km, 2 km grid)
    URL    : https://open-lidar-data.s3.eu-central-1.amazonaws.com/<key>

Pairing (cfg.es_pairing):
  * "cross" (default) - the 1st national coverage (2008-2015) is the previous
    surface. Each selected 2015-2021 training tile is matched to the 2008-2015
    tile at the SAME UTM grid key (E, N); step 02 then builds the prior-year DTM
    from that older tile's ground. PNOA uses one national 2 km grid across both
    coverages, so (E, N) matching is exact; tiles with no 2008-2015 counterpart
    fall back to self-pairing (and are reported), so none are lost.
  * "self" - each tile is paired with itself (one download per tile); the DTM is
    built from the 2015-2021 tile's own ground.

The download budget (cfg.download_budget_gb) caps the CURRENT (training) tiles;
matched 2008-2015 tiles are fetched additionally into ``raw_prev/`` (they are
sparser, so smaller). COPC tiles are valid LAZ 1.4 read directly by
``utils.laz_io.read_points`` (laspy + lazrs) and carry their CRS, so the LAZ
output is stamped EPSG:25829 automatically. Nothing here runs in the build
sandbox (no S3 egress); it runs on the target server. Use
``scripts/01_download_data.py --source pnoa_es --list-only`` to inspect the plan
(including the cross-coverage match rate) before downloading.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

# ---- flai-ai open-lidar-data (anonymous public bucket) ----
ES_BUCKET = "open-lidar-data"
ES_REGION = "eu-central-1"
ES_PREFIX = "data/ES/CNIG/Lidar_2015-2021_epsg25829/copc/"        # 2nd coverage (training clouds)
ES_PREV_PREFIX = "data/ES/CNIG/Lidar_2008-2015_epsg25829/copc/"   # 1st coverage (previous-survey DTM)
ES_EPSG = 25829


def _es_s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
    return boto3.client("s3", region_name=ES_REGION,
                        config=BotoConfig(signature_version=UNSIGNED))


def _list_laz(s3, bucket: str, prefix: str,
              suffixes=(".copc.laz", ".laz", ".las")) -> List[Dict]:
    """List every point-cloud object under ``prefix`` with its size (bytes)."""
    out, token = [], None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            if obj["Key"].lower().endswith(suffixes):
                out.append({"Key": obj["Key"], "Size": int(obj["Size"])})
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return out


# PNOA_2015_GAL-W_474-4750_ORT-CLA-COL.copc.laz -> (2015, 'gal_w', 474, 4750)
_TILE_RE = re.compile(r"PNOA_(\d{4})_([A-Za-z][A-Za-z\-]*)_(\d+)-(\d+)_", re.IGNORECASE)


def _parse_tile(key: str) -> Tuple[Optional[int], str, Optional[int], Optional[int]]:
    base = os.path.basename(key)
    m = _TILE_RE.search(base)
    if m:
        return (int(m.group(1)), m.group(2).lower().replace("-", "_"),
                int(m.group(3)), int(m.group(4)))
    return None, "es", None, None


def _index_by_grid(tiles: List[Dict]) -> Dict[Tuple[int, int], List[Dict]]:
    """Index prior-coverage tiles by their UTM 2 km grid key (E, N)."""
    idx: Dict[Tuple[int, int], List[Dict]] = {}
    for t in tiles:
        _, _, E, N = _parse_tile(t["Key"])
        if E is not None and N is not None:
            idx.setdefault((E, N), []).append(t)
    return idx


def download_pnoa_es(cfg, out_root: str, list_only: bool = False,
                     workers: Optional[int] = None) -> Dict:
    """Download a budget-bounded, spatially-spread PNOA training set plus, in
    cross-coverage mode, the matching 2008-2015 tiles used as the prior DTM.
    """
    s3 = _es_s3()
    cur_prefix = getattr(cfg, "es_prefix", ES_PREFIX)
    pairing = (getattr(cfg, "es_pairing", "cross") or "cross").lower()
    prev_prefix = (getattr(cfg, "es_prev_prefix", ES_PREV_PREFIX) or "") if pairing == "cross" else ""
    region_filter = (getattr(cfg, "es_region_filter", "") or "").lower()
    budget_bytes = int(cfg.download_budget_gb * 1e9)        # decimal GB (matches the repo's listed sizes)
    raw_dir = os.path.join(out_root, "raw")
    raw_prev_dir = os.path.join(out_root, "raw_prev")
    os.makedirs(raw_dir, exist_ok=True)

    # ---- 1. current (2015-2021) training tiles ----
    cur_tiles = _list_laz(s3, ES_BUCKET, cur_prefix)
    if region_filter:
        cur_tiles = [t for t in cur_tiles if region_filter in os.path.basename(t["Key"]).lower()]
    print(f"[es] {len(cur_tiles)} current tiles under s3://{ES_BUCKET}/{cur_prefix}"
          + (f"  (filter '{region_filter}')" if region_filter else ""))
    if not cur_tiles:
        print("[es] WARNING: no current tiles found - check es_prefix / connectivity.")

    # ---- 2. prior (2008-2015) coverage, indexed by UTM grid key ----
    prev_index: Dict[Tuple[int, int], List[Dict]] = {}
    if prev_prefix:
        prev_tiles = _list_laz(s3, ES_BUCKET, prev_prefix)
        prev_index = _index_by_grid(prev_tiles)
        print(f"[es] {len(prev_tiles)} prior tiles under s3://{ES_BUCKET}/{prev_prefix} "
              f"-> {len(prev_index)} grid cells")
        if not prev_index:
            print("[es] WARNING: prior coverage empty / unparsable - every tile will "
                  "fall back to self-pairing. Check es_prev_prefix.")

    # Spread the budget across the region: deterministic shuffle, then accumulate
    # CURRENT tiles to the byte budget. Seeded -> same set every run, so partial
    # downloads resume cleanly (existing files are skipped at download time).
    rng = random.Random(int(getattr(cfg, "es_sample_seed", 1234)))
    cur_tiles = sorted(cur_tiles, key=lambda t: t["Key"])
    rng.shuffle(cur_tiles)

    manifest = {"source": "pnoa_es", "epsg": int(getattr(cfg, "es_epsg", ES_EPSG)),
                "pairing": pairing, "pairs": []}
    plan_cur: List[Tuple[str, str]] = []      # (s3_key, dest) for training tiles
    plan_prev: Dict[str, str] = {}            # s3_key -> dest for prior tiles (dedup)
    used = 0
    n_skipped_unpaired = 0
    cross = (pairing == "cross")

    for t in cur_tiles:                       # shuffled
        year, region, E, N = _parse_tile(t["Key"])
        year = year or 2015

        cands: List[Dict] = []
        if cross:
            # cross-coverage ONLY: keep a training tile iff a 2008-2015 tile exists at
            # the same UTM grid cell. No self-pairing; unpaired tiles are NOT downloaded.
            cands = prev_index.get((E, N), []) if E is not None else []
            if not cands:
                n_skipped_unpaired += 1
                continue                      # no prior tile here -> skip entirely

        if used + t["Size"] > budget_bytes:
            continue                          # over the remaining budget; keep scanning for smaller paired tiles

        dest = os.path.join(raw_dir, region, str(year), os.path.basename(t["Key"]))
        plan_cur.append((t["Key"], dest))
        used += t["Size"]

        if cross:
            prev_clouds, prev_year = [], year
            for pt in cands:                  # ALL prior tiles at the cell feed the DTM (better extent coverage)
                py, preg, _, _ = _parse_tile(pt["Key"])
                py = py or 2009
                pdest = os.path.join(raw_prev_dir, preg, str(py), os.path.basename(pt["Key"]))
                plan_prev[pt["Key"]] = pdest
                prev_clouds.append(pdest)
                prev_year = py
        else:
            prev_clouds, prev_year = [dest], year   # self-pairing: tile is its own prior-DTM source

        manifest["pairs"].append({
            "region": region, "cur_year": year, "prev_year": prev_year,
            "clouds": [dest], "prev_clouds": prev_clouds, "dtms": [],
            "tile_e": E, "tile_n": N,
        })

    if cross:
        pair_note = (f"cross-coverage ONLY: {len(plan_cur)} paired tiles kept, "
                     f"{n_skipped_unpaired} unpaired skipped (not downloaded); "
                     f"{len(plan_prev)} prior tiles feed the DTMs")
    else:
        pair_note = "self-paired (each tile is its own prior-DTM source)"
    print(f"[es] plan: {len(plan_cur)} training tiles = {used/1e9:.2f}/{cfg.download_budget_gb} GB. {pair_note}")

    # ---- 3. download (current -> raw/, prior -> raw_prev/), parallel, resumable ----
    if not list_only and (plan_cur or plan_prev):
        all_jobs = list(plan_cur) + list(plan_prev.items())
        for _, dest in all_jobs:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        n_workers = max(int(workers if workers else getattr(cfg, "download_workers", 16)), 1)
        import concurrent.futures as _cf
        import threading as _th
        _tls = _th.local()

        def _client():
            c = getattr(_tls, "c", None)
            if c is None:
                c = _es_s3()                  # one UNSIGNED client per thread (thread-safe)
                _tls.c = c
            return c

        done = {"n": 0, "bytes": 0}
        lock = _th.Lock()
        n_total = len(all_jobs)

        def _dl(item):
            key, dest = item
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                got = os.path.getsize(dest)   # resume: skip a file already on disk
            else:
                _client().download_file(ES_BUCKET, key, dest)
                got = os.path.getsize(dest) if os.path.exists(dest) else 0
            with lock:
                done["n"] += 1
                done["bytes"] += got
                if done["n"] % 50 == 0 or done["n"] == n_total:
                    print(f"[es]   downloaded {done['n']}/{n_total} files "
                          f"({done['bytes']/1e9:.2f} GB)", flush=True)

        print(f"[es] downloading {n_total} files with {n_workers} parallel streams ...")
        with _cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_dl, all_jobs))

    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[es] training set {used/1e9:.2f} GB across {len(manifest['pairs'])} paired tiles "
          f"({len(plan_prev)} prior tiles) -> {os.path.join(out_root, 'manifest.json')}")
    return manifest
