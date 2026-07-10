"""
Dataset, region-balanced sampler, and collate for MEEPO (KPConv-style sphere mode).

* ``SphereDataset`` indexes **candidate sphere centres** (one sampling unit per
  centre, with its in_radius cylinder point indices precomputed at preprocess time).
  ``__getitem__`` draws the input sphere of radius ``in_radius`` around a centre,
  recentres the coordinates at the centre (KPConv convention), computes the
  shallow features on those centred coordinates, crops the previous-year DTM
  patch, and standardises with ``norm_stats.json``.
* ``make_region_balanced_sampler`` weights spheres so every source cloud (region)
  contributes equally; uniform random is the default (see the trainer).
* ``MultiscaleCollate`` is unchanged - it packs a list of samples into the batch.
"""
from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict
from typing import List, Optional

import numpy as np

from .splitting import effective_split
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

from .augment import augment_tile
from .dtm import (Raster, crop_dtm_patch, MultiRaster, crop_multiraster_patch,
                  _to_dtm_raster, PRIOR_RASTER_CHANNELS)
from .tile_io import load_tile
from ..features.shallow_features import assemble_features
from ..utils.laz_io import IGNORE_LABEL


class _CandView:
    """Sequence of ``(file_idx, within_file_k, center(3,))`` per candidate sphere,
    stored as three CONTIGUOUS NUMPY ARRAYS instead of a Python list of tuples.
    Tuples are materialised on demand so every existing caller
    (``cands[i]``, ``cands[i][0]``, ``enumerate(cands)``, ``len(cands)``) keeps working.

    Why: a forked DataLoader worker inherits the parent's pages copy-on-write, and
    CPython writes a refcount into every Python-object header it touches. Walking a
    list of ~10k tuples (each holding two ndarrays) over an epoch therefore COWs those
    pages into each worker, so host RAM creeps up toward ``num_workers x footprint``
    and - with persistent workers - compounds across epochs into an OOM. Numpy arrays
    are a single object each: indexing reads the data buffer without touching
    per-element refcounts, so the pages stay shared and the creep goes away.
    """
    __slots__ = ("_fi", "_k", "_c")

    def __init__(self, fi, k, centers):
        self._fi, self._k, self._c = fi, k, centers

    def __len__(self):
        return int(self._fi.shape[0])

    def __getitem__(self, i):
        return (int(self._fi[i]), int(self._k[i]), self._c[i])

    def __iter__(self):
        for i in range(int(self._fi.shape[0])):
            yield (int(self._fi[i]), int(self._k[i]), self._c[i])

    @property
    def file_idx(self):
        return self._fi


class SphereDataset(Dataset):
    """KPConv input-sphere dataset. Sampling units are candidate sphere centres."""

    def __init__(self, tile_dir: str, cfg, split: Optional[str] = None,
                 augment: Optional[bool] = None, center_subset: Optional[List[int]] = None):
        if cKDTree is None:
            raise RuntimeError("scipy is required for the sphere dataset")
        self.cfg = cfg
        self.tile_dir = tile_dir
        self.split = split
        if augment is None:
            augment = (split == "train") and bool(getattr(cfg, "use_augmentation", False))
        self.augment = augment
        self.R = float(getattr(cfg, 'tile_stats_radius', 6.0))
        self.min_pts = int(getattr(cfg, 'tile_stats_min_points', 100))

        files = sorted(glob.glob(os.path.join(tile_dir, "*.npz")))
        # Optional TRAIN-tile subset (cfg.max_train_tiles): cap how many train tiles are
        # used so the working set fits in host RAM. Each kept tile retains FULL fidelity
        # (10 cm points, 1 m DTM, every feature) -- we just train on fewer tiles, so there
        # is no per-sample compression or accuracy loss. Deterministic (seeded). val/test
        # always use all tiles so evaluation is unaffected.
        max_tt = int(getattr(cfg, "max_train_tiles", 0) or 0)
        if split == "train" and max_tt > 0:
            train_files = []
            for f in files:
                try:
                    with np.load(f, allow_pickle=True) as d:
                        if effective_split(f, str(d["split"]) if "split" in d else "train", cfg) == "train":
                            train_files.append(f)
                except Exception:
                    pass
            if len(train_files) > max_tt:
                rng = np.random.default_rng(0)
                sel = rng.choice(len(train_files), size=max_tt, replace=False)
                keep = {train_files[i] for i in sel.tolist()}
                files = [f for f in files if f in keep]
        self.files: List[str] = []
        self.file_splits: List[str] = []
        _fi: List[int] = []
        _k: List[int] = []
        _ctr: List[np.ndarray] = []
        for f in files:
            try:
                d = np.load(f, allow_pickle=True)
                fsplit = str(d["split"]) if "split" in d else "train"
                fsplit = effective_split(f, fsplit, cfg)   # --resplit-seed override
                if split is not None and fsplit != split:
                    continue
                centers = d["centers"].astype(np.float32)
            except Exception:
                continue
            fi = len(self.files)
            self.files.append(f)
            self.file_splits.append(fsplit)
            for k in range(centers.shape[0]):
                _fi.append(fi); _k.append(k); _ctr.append(centers[k])
        # Pack per-sample metadata into contiguous arrays (see _CandView for why this
        # matters for multi-worker RAM). The temporary lists above are freed here.
        T = len(_fi)
        cand_fi = np.asarray(_fi, dtype=np.int32)
        cand_k = np.asarray(_k, dtype=np.int32)
        cand_centers = np.stack(_ctr).astype(np.float32) if T else np.zeros((0, 3), np.float32)
        if center_subset is not None:
            sub = np.asarray([i for i in center_subset if 0 <= i < T], dtype=np.int64)
            cand_fi, cand_k, cand_centers = cand_fi[sub], cand_k[sub], cand_centers[sub]
        self.cands = _CandView(cand_fi, cand_k, cand_centers)
        # Bounded LRU of loaded tiles. The unbounded dict here used to hold every
        # tile a worker ever touched (full point array + KD-tree) -> hundreds of GB
        # and OOM at 1000+ tiles. We cap it; with precomputed cylinder indices a
        # cached entry is just the arrays (no KD-tree), so a small cap is cheap.
        self._cache: "OrderedDict[int, dict]" = OrderedDict()
        self._cache_max = max(1, int(getattr(cfg, "tile_cache_size", 4)))

        norm_path = os.path.join(tile_dir, "norm_stats.json")
        if os.path.exists(norm_path):
            with open(norm_path) as fh:
                st = json.load(fh)
            self.mean = np.asarray(st["mean"], dtype=np.float32)
            self.std = np.asarray(st["std"], dtype=np.float32)
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.cands)

    def _load(self, fi: int):
        c = self._cache.get(fi)
        if c is not None:
            self._cache.move_to_end(fi)            # LRU: most-recently used
            return c
        # Big arrays come back as memmaps (shared OS page cache across workers); do
        # NOT .astype() them here - that would materialise the whole array and defeat
        # the mmap. They already carry the right dtype from stage 04; __getitem__
        # slices them (materialising only the sphere's points) and casts the slice.
        t = load_tile(self.files[fi], mmap=True)
        local = t["local"]                              # memmap (N,3) float32
        geo = t["dtm_geo"]
        n_local = local.shape[0]
        intensity = t["intensity"] if t.get("intensity") is not None \
            else np.zeros((n_local,), dtype=np.float32)
        if t.get("ret_ratio") is not None:
            ret_ratio = t["ret_ratio"]
        elif t.get("returns") is not None:              # legacy tiles: mean/mean fallback
            r = np.asarray(t["returns"], dtype=np.float32)
            ret_ratio = r[:, 1] / np.maximum(r[:, 0], 1.0)
        else:
            ret_ratio = np.ones((n_local,), dtype=np.float32)
        # Precomputed cylinder indices (stage 04): __getitem__ slices these instead
        # of rebuilding a whole-tile KD-tree per access, so skip the tree build.
        has_idx = (t.get("cand_off") is not None) and (t.get("cand_idx") is not None)
        c = {
            "local": local,
            "labels": t["labels"],
            "returns": t["returns"],
            "intensity": intensity,
            "ret_ratio": ret_ratio,
            "dtm": Raster(np.asarray(t["dtm_data"], dtype=np.float32),
                          float(geo[0]), float(geo[1]), float(geo[2])),
            "prior": self._load_prior(t, geo),
            "origin": np.asarray(t["file_origin"], dtype=np.float64),
            "cand_off": np.asarray(t["cand_off"], dtype=np.int64) if has_idx else None,
            "cand_idx": t["cand_idx"] if has_idx else None,
            "tree": None if has_idx else cKDTree(np.asarray(local)[:, :2]),  # legacy only
            # precomputed shallow features (stage 04); None on legacy tiles -> per-sphere
            "feat_mean_elev": t.get("feat_mean_elev"),
            "feat_curvature": t.get("feat_curvature"),
            "dtm_height": t.get("dtm_height"),       # (N,) z - prevDTM (use_prev_dtm); None on legacy tiles
        }
        self._cache[fi] = c
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)        # evict least-recently used
        return c

    def _load_prior(self, t, geo):
        """Build the multi-channel previous-year-classification MultiRaster for a
        tile. Prefers the stored 5-channel ``prior_data`` (stage 04 new); if only
        the legacy single-channel ``dtm_data`` exists, synthesise the channels
        (DSM=DTM, nDSM=0, ground_prob=1, coverage=1) so the raster branch still
        runs with a sensible (ground-everywhere) prior."""
        if t.get("prior_data") is not None and t.get("prior_geo") is not None:
            pg = t["prior_geo"]
            return MultiRaster(np.asarray(t["prior_data"], dtype=np.float32),
                               float(pg[0]), float(pg[1]), float(pg[2]),
                               PRIOR_RASTER_CHANNELS)
        dtm = np.asarray(t["dtm_data"], dtype=np.float32)
        H, W = dtm.shape
        data = np.stack([dtm, dtm, np.zeros_like(dtm),
                         np.ones_like(dtm), np.ones_like(dtm)], axis=0)
        return MultiRaster(data, float(geo[0]), float(geo[1]), float(geo[2]),
                           PRIOR_RASTER_CHANNELS)

    def __getitem__(self, i: int):
        fi, ck, center = self.cands[i]
        f = self._load(fi)
        center = center.astype(np.float64).copy()
        jit = np.zeros(2, dtype=np.float64)
        if self.augment:                                  # xy jitter of the centre
            j = float(getattr(self.cfg, "sphere_center_jitter", self.R * 0.25))
            jit = np.random.uniform(-j, j, size=2)

        if f["cand_off"] is not None:
            # precomputed in_radius cylinder around the grid centre; apply jitter as
            # a translation of the input region - equivalent to
            # KPConv's centre jitter for augmentation, and it keeps the >= min_pts
            # guarantee that preprocessing already enforced.
            off = f["cand_off"]
            idx = f["cand_idx"][int(off[ck]):int(off[ck + 1])]
            center[:2] += jit
        else:                                             # legacy tiles: jitter then query
            center[:2] += jit
            idx = f["tree"].query_ball_point(center[:2], self.R)
            if len(idx) < self.min_pts:
                kk = min(self.min_pts, f["local"].shape[0])
                idx = np.atleast_1d(f["tree"].query(center[:2], k=kk)[1])
            idx = np.asarray(idx, dtype=np.int64)

        sph = f["local"][idx]
        centered = (sph - center).astype(np.float32)      # KPConv-centred input region
        labels = f["labels"][idx]
        nr = f["returns"][idx, 0]
        rn = f["returns"][idx, 1]
        it = f["intensity"][idx]
        rr = f["ret_ratio"][idx]

        ps = int(getattr(self.cfg, "dtm_patch_size", 64))
        nchan = int(getattr(self.cfg, "prior_raster_channels", 5))
        if getattr(self.cfg, "use_dtm_raster", False):
            cw = center + f["origin"]                     # world centre
            dtm_patch = crop_multiraster_patch(f["prior"], cw[0] - self.R, cw[1] - self.R,
                                               2.0 * self.R, ps, origin_z=float(cw[2]))
        else:
            dtm_patch = np.zeros((nchan, ps, ps), dtype=np.float32)

        if self.augment:
            rng = np.random.default_rng()
            centered, dtm_patch = augment_tile(centered, dtm_patch, self.cfg, rng,
                                               2.0 * self.R, pivot=0.0)

        # Precomputed shallow features (stage 04): slice to the sphere; re-centre the
        # mean elevation by the region-centre z (exact: the weighted mean is
        # translation-equivariant). None on legacy tiles -> assemble_features computes
        # them per sphere (the original, slower path).
        mep = mcp = None
        if f.get("feat_mean_elev") is not None:
            mep = (f["feat_mean_elev"][idx] - np.float32(center[2])).astype(np.float32)
        if f.get("feat_curvature") is not None:
            mcp = f["feat_curvature"][idx].astype(np.float32)
        dh = None if f.get("dtm_height") is None else np.asarray(f["dtm_height"])[idx].astype(np.float32)

        feats = assemble_features(centered, self.cfg,
                                  num_returns=nr, return_number=rn,
                                  intensity=it, return_ratio=rr,
                                  mean_elev_precomp=mep, curvature_precomp=mcp,
                                  dtm_height=dh).astype(np.float32)
        if self.mean is not None and feats.shape[1] == self.mean.shape[0]:
            feats = (feats - self.mean) / self.std

        return {
            "points": centered,
            "features": feats,
            "labels": labels.astype(np.int64),
            "dtm_patch": dtm_patch.astype(np.float32),
            "origin": (center + f["origin"]).astype(np.float64),   # world centre
            "path": f"{os.path.splitext(os.path.basename(self.files[fi]))[0]}_c{i}",
        }

    def gallery_center_indices(self, n_want: int = 6) -> List[int]:
        """Candidate indices for the per-epoch visual gallery, chosen to cover a
        RANGE OF REGIONAL AREAS rather than any scene-type taxonomy.

        Spheres are grouped by their source cloud (each cloud is a distinct NZ
        region/survey). We round-robin across clouds so the gallery never collapses
        onto one terrain type (e.g. forest); within each cloud the picks are spread
        spatially (farthest-point order over the candidate centres) so adjacent,
        near-identical spheres are not all chosen. Deterministic given the tile set.
        """
        n = len(self)
        if n == 0:
            return []
        fi = np.asarray(self.cands.file_idx)
        ctr = np.asarray(self.cands._c)[:, :2].astype(np.float64)
        # per-cloud, order candidates by a farthest-point sweep for spatial spread
        order_by_file = {}
        for f in np.unique(fi):
            idx = np.where(fi == f)[0]
            if idx.size <= 2:
                order_by_file[int(f)] = list(idx)
                continue
            pts = ctr[idx]
            chosen = [0]
            d = np.linalg.norm(pts - pts[0], axis=1)
            while len(chosen) < idx.size:
                j = int(np.argmax(d))
                chosen.append(j)
                d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
            order_by_file[int(f)] = [int(idx[c]) for c in chosen]
        # round-robin across clouds (regional diversity)
        files = sorted(order_by_file.keys())
        out, depth = [], 0
        while len(out) < min(n_want, n):
            progressed = False
            for f in files:
                seq = order_by_file[f]
                if depth < len(seq):
                    out.append(seq[depth]); progressed = True
                    if len(out) >= min(n_want, n):
                        break
            depth += 1
            if not progressed:
                break
        return out

    def candidate_point_counts(self) -> np.ndarray:
        """Input-point count of every candidate's cylinder, for KPConv's variable
        batch budget. Computed once and cached. When tiles carry precomputed cylinder
        offsets (stage 04) this is a free lookup; otherwise it falls back to a per-file
        KD-tree pass (memory-safe: each file is loaded only transiently). Counts use the
        same >= ``sphere_min_points`` floor as ``__getitem__``."""
        if getattr(self, "_pt_counts", None) is not None:
            return self._pt_counts
        from collections import defaultdict
        from ..utils.progress import progress
        counts = np.zeros(len(self.cands), dtype=np.int64)
        by_file = defaultdict(list)
        for gi, (fi, ck, center) in enumerate(self.cands):
            by_file[fi].append((gi, ck, np.asarray(center, dtype=np.float64)))
        R, mp = float(self.R), int(self.min_pts)
        for fi in progress(list(by_file.keys()), desc="[setup] cylinder sizes"):
            items = by_file[fi]
            try:
                d = np.load(self.files[fi], allow_pickle=True)
                if "cand_off" in d:                        # precomputed -> O(1) per candidate
                    off = d["cand_off"].astype(np.int64)
                    for gi, ck, _c in items:
                        counts[gi] = max(int(off[ck + 1] - off[ck]), mp)
                    continue
                local = d["local"].astype(np.float32)      # legacy fallback
                tree = cKDTree(local[:, :2])
                centers = np.stack([c[:2] for _, _, c in items])
                lists = tree.query_ball_point(centers, R)
            except Exception:
                for gi, _ck, _c in items:
                    counts[gi] = mp
                continue
            for (gi, _ck, _c), lst in zip(items, lists):
                counts[gi] = max(int(len(lst)), mp)
        self._pt_counts = counts
        return counts


def calibrate_batch_limit(dataset, cfg, weights=None) -> int:
    """Pick the total-input-points-per-batch budget so a variable batch averages
    ``cfg.batch_num`` spheres (KPConv supplementary, Sec. A). The per-sphere mean is
    taken under the sampling distribution when ``weights`` (the skew-sampler weights)
    are supplied, else uniform."""
    counts = np.asarray(dataset.candidate_point_counts(), dtype=np.float64)
    bn = max(int(getattr(cfg, "batch_num", 10)), 1)
    if counts.size == 0:
        return 1
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        mean_pts = float((counts * (w / w.sum())).sum()) if (w.size == counts.size and w.sum() > 0) \
            else float(counts.mean())
    else:
        mean_pts = float(counts.mean())
    # +0.5 sphere of slack: greedy "stop before exceeding the budget" packing lands
    # about half a sphere under the limit, so this centres the average batch on
    # batch_num (rather than ~0.5 under it).
    return max(int(round((bn + 0.5) * mean_pts)), 1)


def calibrate_neighbor_limit(dataset, cfg, n_spheres: int = 48, percentile: float = 95.0,
                             num_layers: int = None, max_limit: int = None,
                             verbose: bool = True) -> int:
    """Measure the per-layer radius-neighbour-count distribution on a sample of input
    spheres and return a single ``neighbor_limit`` that covers ``percentile`` of
    neighbourhoods at *every* layer (KPConv's calibration philosophy, adapted to this
    repo's single global cap), CLAMPED to ``max_limit`` (default ``cfg.neighbor_limit_max``)
    so a large conv_radius can't hand back an OOM-inducing value. ``num_layers`` defaults
    to ``cfg.num_layers``.

    Mirrors ``build_multiscale_batch`` geometry exactly: per layer ``l`` the grid is
    ``dl0 * 2^l`` and the conv/pool radius is ``conv_radius * dl_l``. For each sampled
    sphere we grid-subsample to every level and, per query point, count radius
    neighbours among the same-level support and among the finer support gathered by
    the coarser (pooled) queries - the two searches the network actually does.

    Returns the chosen limit; prints a per-layer table when ``verbose``.
    """
    from .subsampling import grid_subsample
    if num_layers is None:
        num_layers = int(getattr(cfg, "num_layers", 5))
    if cKDTree is None or len(dataset) == 0:
        return int(getattr(cfg, "neighbor_limit", 50) or 50)
    dl0 = float(cfg.first_subsampling_dl)
    conv_r = float(cfg.conv_radius)
    gs = float(getattr(cfg, "grid_scaling", 2.0))   # match build_multiscale_batch
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))
    sel = rng.choice(len(dataset), size=min(int(n_spheres), len(dataset)), replace=False)

    same = [[] for _ in range(num_layers)]      # same-resolution conv neighbour counts
    pool = [[] for _ in range(num_layers - 1)]  # strided/pool neighbour counts (l -> l+1)
    for si in sel:
        pts = np.asarray(dataset[int(si)]["points"], dtype=np.float32)
        # build the resolution pyramid for this single cloud
        levels = []
        sp, _, _ = grid_subsample(pts, None, None, dl0)
        levels.append(sp if sp.shape[0] else pts[:1])
        for l in range(1, num_layers):
            sp, _, _ = grid_subsample(levels[-1].astype(np.float32), None, None, dl0 * (gs ** l))
            levels.append(sp if sp.shape[0] else levels[-1][:1])
        for l in range(num_layers):
            r_l = conv_r * dl0 * (gs ** l)
            P = levels[l]
            t = cKDTree(P)
            same[l].append(np.array([len(x) for x in t.query_ball_point(P, r_l)], dtype=np.int64))
            if l < num_layers - 1:
                Pc = levels[l + 1]
                # coarse queries gather fine supports within the fine radius r_l
                pool[l].append(np.array([len(x) for x in t.query_ball_point(Pc, r_l)], dtype=np.int64))

    def _pct(chunks):
        if not chunks:
            return (0, 0, 0)
        a = np.concatenate(chunks)
        if a.size == 0:
            return (0, 0, 0)
        return (int(np.percentile(a, 50)), int(np.percentile(a, percentile)), int(a.max()))

    per_layer = []
    for l in range(num_layers):
        s50, sp, smx = _pct(same[l])
        per_layer.append((l, conv_r * dl0 * (gs ** l), s50, sp, smx))
    pool_pct = [_pct(pool[l])[1] for l in range(num_layers - 1)]
    p95_cap = int(max([p[3] for p in per_layer] + pool_pct + [1]))
    if max_limit is None:
        max_limit = int(getattr(cfg, "neighbor_limit_max", 0) or 0)
    clamped = (max_limit > 0 and p95_cap > max_limit)
    chosen = max_limit if clamped else p95_cap
    ksum = int(sum(getattr(cfg, "kernel_sizes", [7, 13, 15])))   # differences tensor ~ M x H x ksum x 3
    bytes_per_pt_per_H = ksum * 3 * 4                            # fp32

    if verbose:
        print(f"[calib] neighbour counts over {len(sel)} spheres "
              f"(conv_radius={conv_r}, dl0={dl0}); columns: median / p{int(percentile)} / max")
        for (l, r_l, s50, sp, smx) in per_layer:
            cur = int(getattr(cfg, "neighbor_limit", 50) or 0)
            trunc = "" if cur <= 0 else f"  (cap {cur} keeps {min(100.0, 100.0*cur/max(sp,1)):.0f}% of p{int(percentile)})"
            print(f"[calib]  layer {l}: radius {r_l:5.2f} m   conv {s50:4d}/{sp:4d}/{smx:4d}{trunc}")
        chosen_gb = chosen * bytes_per_pt_per_H * 100_000 / 1e9
        if clamped:
            full_gb = p95_cap * bytes_per_pt_per_H * 100_000 / 1e9
            print(f"[calib]  WARNING: p{int(percentile)} cap is {p95_cap} (~{full_gb:.0f} GB layer-0 "
                  f"differences per 100k-pt batch) -> CLAMPED to neighbor_limit_max={max_limit}.")
        print(f"[calib]  -> neighbor_limit = {chosen} (p{int(percentile)}; "
              f"~{chosen_gb:.0f} GB layer-0 differences per 100k-pt batch, scales linearly with the cap "
              f"and with batch_limit - see the [setup] line for your batch size)")
    return int(chosen)


class PointBudgetBatchSampler:
    """KPConv variable batch size (supplementary, Sec. A). Groups a stream of sample
    indices into batches whose summed input-point counts stay <= ``batch_limit``
    (always at least one sphere per batch; a single oversized sphere is emitted on
    its own). With ``batch_limit`` set by :func:`calibrate_batch_limit`, batches
    average ``cfg.batch_num`` spheres - so the paper's batch size is preserved while
    the per-batch point count, and thus GPU memory and step time, stays roughly
    constant across varying point density."""

    def __init__(self, index_sampler, point_counts, batch_limit, num_batches=None):
        self.index_sampler = index_sampler
        self.point_counts = np.asarray(point_counts, dtype=np.int64)
        self.batch_limit = int(batch_limit)
        self._num_batches = int(num_batches) if num_batches else None

    def __iter__(self):
        batch, cur, lim, pc, n = [], 0, self.batch_limit, self.point_counts, self.point_counts.shape[0]
        for idx in self.index_sampler:
            idx = int(idx)
            c = int(pc[idx]) if 0 <= idx < n else 0
            if batch and cur + c > lim:
                yield batch
                batch, cur = [], 0
            batch.append(idx)
            cur += c
        if batch:
            yield batch

    def __len__(self):
        if self._num_batches is not None:
            return max(self._num_batches, 1)
        try:
            n_idx = len(self.index_sampler)
        except TypeError:
            n_idx = self.point_counts.shape[0]
        mean = max(float(self.point_counts.mean()), 1.0)
        per = max(self.batch_limit / mean, 1.0)
        return max(int(round(n_idx / per)), 1)


class _WeightedReplacementSampler(torch.utils.data.Sampler):
    """Weighted sampling WITH replacement, free of ``torch.multinomial``'s 2**24
    category cap. Draws ``num_samples`` indices per epoch by inverse-CDF lookup, so
    it matches ``WeightedRandomSampler(..., replacement=True)`` while scaling to the
    tens of millions of candidate spheres produced by large/dense surveys (PNOA's
    2 km tiles at 2.8 pts/m^2 yield ~10^7-10^8 spheres). Re-seeded each epoch."""

    def __init__(self, weights, num_samples: int, seed: int = 0):
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            w = np.ones_like(w)
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        cdf[-1] = 1.0                      # guard fp drift so searchsorted stays in range
        self._cdf = cdf
        self._n = int(num_samples)
        self._seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        u = rng.random(self._n)
        idx = np.searchsorted(self._cdf, u, side="right")
        np.clip(idx, 0, self._cdf.shape[0] - 1, out=idx)
        return iter(idx.tolist())


def make_region_balanced_sampler(dataset, cfg):
    """Weighted sampler that gives every SOURCE CLOUD (region/survey) equal expected
    mass, so large or dense clouds do not dominate an epoch.

    Each candidate sphere's weight is ``1 / (#candidates in its source cloud)``,
    normalised to sum to 1. Sampling with replacement then draws clouds uniformly
    and, within a cloud, its spheres uniformly. This is the regional-diversity
    sampler; the trainer uses it only when ``use_region_balanced_sampler=True``
    (uniform-over-all-spheres is the default).

    Returns a replacement sampler robust to very large candidate pools (see
    ``_WeightedReplacementSampler``); ``torch.multinomial`` cannot be used directly
    because it rejects more than 2**24 categories.
    """
    fi = np.asarray(getattr(getattr(dataset, "cands", None), "_fi", None))
    n = int(fi.shape[0]) if fi is not None and fi.ndim == 1 else len(dataset)
    nsamp = int(getattr(cfg, "epoch_steps", 0)) * int(getattr(cfg, "batch_num", 1))
    if nsamp <= 0:
        nsamp = n
    if n == 0:
        return None
    if fi is None or fi.ndim != 1:
        weights = np.ones(n, dtype=np.float64)
    else:
        counts = np.bincount(fi.astype(np.int64))
        per_file_w = np.divide(1.0, counts, out=np.zeros_like(counts, dtype=np.float64),
                               where=counts > 0)
        weights = per_file_w[fi.astype(np.int64)]
    if weights.sum() <= 0:
        weights = np.ones(n, dtype=np.float64)
    weights /= weights.sum()
    return _WeightedReplacementSampler(weights, nsamp, seed=int(getattr(cfg, "seed", 0)))


# Backwards-compatible alias for any external caller / older import.
make_skew_sampler = make_region_balanced_sampler


# The PTv3 collate lives in ptv3_collate.py; re-export under the old name for
# any external caller that still imports MultiscaleCollate from here.
from .ptv3_collate import PTv3Collate as MultiscaleCollate  # noqa: E402,F401
