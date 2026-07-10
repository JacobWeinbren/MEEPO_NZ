"""Full-scene dataset for PTv3 / MEEPO.

This replaces the KPConv-era ``SphereDataset`` (which cropped many small
``in_radius`` cylinders per tile and packed them into a points-budget batch).
PTv3 is serialization-based and has no ball-query neighbourhoods, so it ingests
a whole scene directly. Here **one item = one whole preprocessed tile**; if a
tile is larger than ``scene_max_points`` (so one forward pass would not fit), a
single large axis-aligned block is drawn from it (PTv3's "cap by point count,
not radius" convention) - never a 5 m sphere.

The per-sample dict matches what ``PTv3Collate`` already expects
(``points`` / ``features`` / ``labels`` / ``ndsm`` / ``origin`` / ``path``), so the
collate, model, MoE, height-aware loss and DTM-RMSE metric
are all unchanged. The previous-year raster branch (the per-sphere 64x64 CNN
patch) is *sphere machinery* and is intentionally not used here - scene mode is
point-only (xyz + return-count + return-ratio + intensity), which is the
PTv3-native form and now carries far more spatial context than the old 5 m
cylinder ever could.
"""
from __future__ import annotations

import glob
import os
from collections import OrderedDict
from typing import List, Optional

import numpy as np
from torch.utils.data import Dataset

from .augment import augment_tile
from .dtm import MultiRaster, crop_multiraster_patch, PRIOR_RASTER_CHANNELS
from .splitting import effective_split
from .tile_io import load_tile
from ..features.shallow_features import assemble_features
from ..utils.laz_io import IGNORE_LABEL


class _SceneCandView:
    """Gallery-compatibility shim: yields ``(file_idx, 0, centre)`` per tile so the
    trainer's per-epoch visualiser renders whole scenes instead of spheres."""

    def __init__(self, ds: "SceneDataset"):
        self._ds = ds

    def __len__(self):
        return len(self._ds.files)

    def __getitem__(self, i):
        return (int(i), 0, self._ds._centroid(i))

    @property
    def file_idx(self):
        return np.arange(len(self._ds.files), dtype=np.int64)

    @property
    def _c(self):
        return np.stack([self._ds._centroid(i) for i in range(len(self._ds.files))]) \
            if len(self._ds.files) else np.zeros((0, 3), np.float32)


class SceneDataset(Dataset):
    """One whole tile (or one large block of an oversized tile) per item."""

    def __init__(self, tile_dir: str, cfg, split: Optional[str] = None,
                 augment: Optional[bool] = None, center_subset: Optional[List[int]] = None):
        self.cfg = cfg
        self.tile_dir = tile_dir
        self.split = split
        if augment is None:
            augment = (split == "train") and bool(getattr(cfg, "use_augmentation", False))
        self.augment = augment
        self.max_points = int(getattr(cfg, "scene_max_points", 1_500_000))
        self.block_size = float(getattr(cfg, "scene_block_size", 64.0))
        self.min_points = int(getattr(cfg, "scene_min_points",
                                       getattr(cfg, "sphere_min_points", 100)))
        self.mean = None
        self.std = None
        self._cache: "OrderedDict[int, dict]" = OrderedDict()
        self._cache_max = int(getattr(cfg, "scene_cache_tiles", 4))

        files = sorted(glob.glob(os.path.join(tile_dir, "*.npz")))
        self.files: List[str] = []
        for f in files:
            try:
                with np.load(f, allow_pickle=True) as d:
                    fsplit = str(d["split"]) if "split" in d else "train"
                    fsplit = effective_split(f, fsplit, cfg)   # --resplit-seed override
                    if split is not None and fsplit != split:
                        continue
            except Exception:
                continue
            self.files.append(f)
        if center_subset is not None:
            keep = [i for i in center_subset if 0 <= i < len(self.files)]
            self.files = [self.files[i] for i in keep]

        # per-channel standardisation (same norm_stats.json as the sphere pipeline;
        # scene mode uses the identical per-point features, so the stats transfer)
        import json
        norm_path = os.path.join(tile_dir, "norm_stats.json")
        try:
            with open(norm_path) as fh:
                st = json.load(fh)
            self.mean = np.asarray(st["mean"], dtype=np.float32)
            self.std = np.asarray(st["std"], dtype=np.float32)
        except Exception:
            self.mean = None
            self.std = None

    # ----------------------------------------------------------------- length
    def __len__(self) -> int:
        return len(self.files)

    # ------------------------------------------------------------ tile loading
    def _load(self, fi: int) -> dict:
        if fi in self._cache:
            self._cache.move_to_end(fi)
            return self._cache[fi]
        t = load_tile(self.files[fi], mmap=True)
        local = t["local"]
        n_local = local.shape[0]
        intensity = t["intensity"] if t.get("intensity") is not None \
            else np.zeros((n_local,), dtype=np.float32)
        if t.get("ret_ratio") is not None:
            ret_ratio = t["ret_ratio"]
        elif t.get("returns") is not None:
            r = np.asarray(t["returns"], dtype=np.float32)
            ret_ratio = r[:, 1] / np.maximum(r[:, 0], 1.0)
        else:
            ret_ratio = np.ones((n_local,), dtype=np.float32)
        c = {
            "local": local,
            "labels": t["labels"],
            "returns": t["returns"],
            "intensity": intensity,
            "ret_ratio": ret_ratio,
            "origin": np.asarray(t["file_origin"], dtype=np.float64),
            "feat_mean_elev": t.get("feat_mean_elev"),
            "feat_curvature": t.get("feat_curvature"),
            "dtm_height": t.get("dtm_height"),       # (N,) z - prevDTM (use_prev_dtm); None on legacy tiles
            "prior": self._load_prior(t),    # 5-ch previous-year prior MultiRaster (Deviation A)
            "dtm": None,
        }
        self._cache[fi] = c
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return c

    def _load_prior(self, t):
        """Build the previous-year prior MultiRaster from a tile (same as the sphere
        dataset): prefer the stored 5-channel ``prior_data``; fall back to a legacy
        single-channel ``dtm_data`` (DSM=DTM, nDSM=0, gprob=1, cover=1); else None."""
        if t.get("prior_data") is not None and t.get("prior_geo") is not None:
            pg = np.asarray(t["prior_geo"], dtype=np.float64)
            return MultiRaster(np.asarray(t["prior_data"], dtype=np.float32),
                               float(pg[0]), float(pg[1]), float(pg[2]), PRIOR_RASTER_CHANNELS)
        if t.get("dtm_data") is not None and t.get("dtm_geo") is not None:
            dtm = np.asarray(t["dtm_data"], dtype=np.float32)
            g = np.asarray(t["dtm_geo"], dtype=np.float64)
            data = np.stack([dtm, dtm, np.zeros_like(dtm), np.ones_like(dtm), np.ones_like(dtm)], 0)
            return MultiRaster(data, float(g[0]), float(g[1]), float(g[2]), PRIOR_RASTER_CHANNELS)
        return None

    def _centroid(self, fi: int) -> np.ndarray:
        c = self._load(fi)
        xy = np.asarray(c["local"], dtype=np.float64)
        ctr = xy.mean(0) if xy.shape[0] else np.zeros(3)
        return ctr.astype(np.float64)

    # ---------------------------------------------------------- block cropping
    def _crop_block(self, xyz: np.ndarray, seed: int):
        """Indices of one point-budget region (<= max_points) of an oversized tile, at
        FULL (native) density -- Pointcept's SphereCrop(point_max): take the ``max_points``
        points NEAREST a centre. This is the SAME region predict_scene feeds at inference
        (it tiles the cloud into ~max_points full-density blocks), so train / val / test /
        inference all operate on identically-sized, identical-density regions, as KPConv
        (fixed in_radius) and Pointcept (SphereCrop point_max) prescribe.

        Centre selection is AREA-uniform (a random location in the tile extent), NOT a
        random *point*. A random point is density-weighted: it would over-sample dense
        regions (forest/buildings) and tile interiors and under-sample sparse areas and
        tile EDGES -- a spatial training bias, since predict_scene covers every block
        uniformly at test. Area-uniform centres cover the tile evenly (KPConv uses a
        potential field for the same goal) and match the prior validated runs. Full-
        density nearest-N then fixes the density bug (the old code took a random *subset*
        of a 300 m window -> ~1/10 the density predict_scene feeds full-resolution)."""
        n = xyz.shape[0]
        xy = xyz[:, :2]
        lo = xy.min(0); hi = xy.max(0)
        if self.augment:
            cx, cy = np.random.default_rng(None).uniform(lo, hi)   # AREA-uniform centre
        else:
            cx, cy = 0.5 * (lo + hi)                               # deterministic centre (eval)
        k = int(min(self.max_points, n))
        if k >= n:
            return np.arange(n)
        d2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
        return np.argpartition(d2, k - 1)[:k]                  # nearest max_points

    # --------------------------------------------------------- gallery support
    @property
    def cands(self):
        return _SceneCandView(self)

    def gallery_center_indices(self, n_want: int = 6) -> List[int]:
        """Pick a gallery covering a RANGE OF REGIONAL AREAS: in scene mode one
        tile == one whole scene, so we spread the picks spatially across distinct
        tiles (farthest-point order over tile centroids) rather than by any
        scene-type taxonomy. Deterministic given the tile set."""
        n = len(self.files)
        if n == 0:
            return []
        if n <= n_want:
            return list(range(n))
        ctr = np.stack([self._centroid(i)[:2] for i in range(n)]).astype(np.float64)
        chosen = [0]
        d = np.linalg.norm(ctr - ctr[0], axis=1)
        while len(chosen) < min(n_want, n):
            j = int(np.argmax(d))
            chosen.append(j)
            d = np.minimum(d, np.linalg.norm(ctr - ctr[j], axis=1))
        return chosen

    def candidate_point_counts(self) -> np.ndarray:
        # only consulted by the (unused) KPConv variable-batch path
        return np.ones(len(self.files), dtype=np.int64)

    # ----------------------------------------------------------------- getitem
    def __getitem__(self, i: int) -> dict:
        t = self._load(i)
        xyz = np.asarray(t["local"], dtype=np.float32)
        labels = np.asarray(t["labels"], dtype=np.int64)
        ret = np.asarray(t["returns"])
        nr = ret[:, 0]; rn = ret[:, 1]
        it = np.asarray(t["intensity"], dtype=np.float32)
        rr = np.asarray(t["ret_ratio"], dtype=np.float32)
        mep_all = t.get("feat_mean_elev")
        mcp_all = t.get("feat_curvature")
        dh_all = t.get("dtm_height")

        # RandomDropout (MEEPO: dropout_ratio=0.2, dropout_application_ratio=0.2) -- with some
        # per-sample probability, drop a random fraction of points. A density-robustness aug that
        # does NOT move (x,y), so the per-point prior-raster sampling stays exact. Train only.
        if self.augment:
            dro = float(getattr(self.cfg, "augment_dropout_ratio", 0.0))
            dap = float(getattr(self.cfg, "augment_dropout_prob", 0.0))
            if dro > 0.0 and dap > 0.0 and xyz.shape[0] > self.min_points \
                    and np.random.default_rng().random() < dap:
                rngd = np.random.default_rng()
                keep = rngd.random(xyz.shape[0]) >= dro
                if int(keep.sum()) >= self.min_points:
                    xyz = xyz[keep]; labels = labels[keep]
                    nr = nr[keep]; rn = rn[keep]; it = it[keep]; rr = rr[keep]
                    if mep_all is not None: mep_all = np.asarray(mep_all)[keep]
                    if mcp_all is not None: mcp_all = np.asarray(mcp_all)[keep]
                    if dh_all is not None: dh_all = np.asarray(dh_all)[keep]

        # oversized tile -> one large block (PTv3 cap-by-count, not a 5 m sphere)
        if xyz.shape[0] > self.max_points:
            sel = self._crop_block(xyz, seed=i)
            xyz = xyz[sel]; labels = labels[sel]
            nr = nr[sel]; rn = rn[sel]; it = it[sel]; rr = rr[sel]
            if mep_all is not None: mep_all = np.asarray(mep_all)[sel]
            if mcp_all is not None: mcp_all = np.asarray(mcp_all)[sel]
            if dh_all is not None: dh_all = np.asarray(dh_all)[sel]

        # centre the scene at its centroid (translation-equivariant; keeps coords small)
        center = xyz.mean(0).astype(np.float64) if xyz.shape[0] else np.zeros(3)
        centered = (xyz - center).astype(np.float32)

        # --- Deviation A integrated into the whole-scene pipeline ---
        # Crop the previous-year prior raster to THIS block window (side = scene_block_size,
        # centred at the scene centroid), resample to raster_scene_patch_size px, offset the
        # height channels by the centre z. The fully-convolutional GrounDiff CNN runs ONCE on
        # this patch inside the model forward (trains end-to-end); each point bilinearly
        # samples it (tile_size = scene_block_size). The identical path is used at val/inference.
        use_rast = bool(getattr(self.cfg, "use_dtm_raster", True))
        T = float(self.block_size)
        if use_rast:
            ps = int(getattr(self.cfg, "raster_scene_patch_size", 128))
            wc = center + np.asarray(t["origin"], dtype=np.float64)          # world centre (x,y,z)
            dtm_patch = crop_multiraster_patch(t.get("prior"), wc[0] - T / 2.0, wc[1] - T / 2.0,
                                               T, ps, origin_z=float(wc[2]))
        else:
            dtm_patch = None

        if self.augment:
            rng = np.random.default_rng()
            centered, dtm_patch = augment_tile(centered, dtm_patch, self.cfg, rng,
                                               tile_size=T, pivot=0.0)

        mep = None if mep_all is None else (np.asarray(mep_all) - np.float32(center[2])).astype(np.float32)
        mcp = None if mcp_all is None else np.asarray(mcp_all).astype(np.float32)
        dh = None if dh_all is None else np.asarray(dh_all).astype(np.float32)
        feats = assemble_features(centered, self.cfg,
                                  num_returns=nr, return_number=rn,
                                  intensity=it, return_ratio=rr,
                                  mean_elev_precomp=mep, curvature_precomp=mcp,
                                  dtm_height=dh).astype(np.float32)
        if self.mean is not None and feats.shape[1] == self.mean.shape[0]:
            feats = (feats - self.mean) / self.std

        # GrounDiff continuous nDSM regression target (height above bare earth, m);
        # NaN where invalid/ignored so the loss skips those points.
        if getattr(self.cfg, "use_groundiff_regression", False):
            from .dtm import height_above_ground
            ndsm = height_above_ground(
                centered.astype(np.float64), labels.astype(np.int64),
                res=float(getattr(self.cfg, "ndsm_dtm_res", 1.0)),
                min_ground=int(getattr(self.cfg, "ndsm_min_ground", 8)))
        else:
            ndsm = np.full(centered.shape[0], np.nan, dtype=np.float32)

        out = {
            "points": centered,
            "features": feats,
            "labels": labels.astype(np.int64),
            "ndsm": ndsm.astype(np.float32),
            "origin": (center + t["origin"]).astype(np.float64),
            "path": os.path.splitext(os.path.basename(self.files[i]))[0],
        }
        if use_rast and dtm_patch is not None:
            out["dtm_patch"] = np.asarray(dtm_patch, dtype=np.float32)
        # learned SPAG-DC: per-scene "oracle" regime globals from the GT-ground terrain.
        if getattr(self.cfg, "spag_learned", False):
            from ..inference.spag_dc import oracle_regime_globals
            out["regime"] = oracle_regime_globals(
                centered[labels == 1].astype(np.float64)).astype(np.float32)
        return out
