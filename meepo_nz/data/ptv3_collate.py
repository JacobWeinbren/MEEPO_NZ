"""PTv3 collate: pack a list of spheres into the batch dict the MEEPO model
and the (reused) Trainer/visualizer expect.

Replaces the KPConv ``MultiscaleCollate`` + ``build_multiscale_batch`` (which
built a 5-level kd-tree neighbour pyramid).  PTv3 needs none of that: it works
on voxel coordinates and a serialized order computed inside the model.  So this
collate only has to:

  * **grid-sample** each sphere to one point per voxel at ``grid_size``
    (= ``first_subsampling_dl``), keeping a mapping so per-point labels are
    majority-voted into voxels and predictions can be propagated back;
  * concatenate clouds and emit ``coord`` / ``grid_coord`` / ``feat`` / ``offset``
    (PTv3 convention) plus ``labels``;
  * stack the per-cloud previous-year-classification raster patches into
    ``dtm_patches`` (B, C, H, W);
  * carry ``points`` / ``lengths`` / ``origins`` / ``paths`` so
    the unchanged visualizer and per-epoch gallery keep working.

The output dict is consumed unchanged by the reused ``training/trainer.py`` and
``data/batch.py:move_batch``.
"""
from __future__ import annotations

from typing import List

import numpy as np
import random as _random
import torch

from ..utils.laz_io import IGNORE_LABEL


def _voxelize(coord, grid_size):
    """Map points to integer voxels; return (grid_coord_int, inverse) where
    ``inverse[i]`` is the voxel index of point i. One representative point per
    voxel is kept by the caller via segment reductions."""
    gmin = coord.min(0)
    gc = np.floor((coord - gmin) / grid_size).astype(np.int64)
    # unique voxels + inverse mapping
    key = gc - gc.min(0)
    span = key.max(0) + 1
    flat = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    uniq, inverse = np.unique(flat, return_inverse=True)
    return gc, inverse


def _majority_label(labels, inverse, n_vox, num_classes, ignore_index):
    """Per-voxel majority label over the points that fell in it (ignoring
    ignore_index where possible)."""
    out = np.full(n_vox, ignore_index, dtype=np.int64)
    # counts[v, c]
    valid = labels != ignore_index
    if valid.any():
        cc = np.zeros((n_vox, num_classes), dtype=np.int64)
        np.add.at(cc, (inverse[valid], labels[valid]), 1)
        has = cc.sum(1) > 0
        out[has] = cc[has].argmax(1)
    return out


def _voxel_reduce_mean(values, inverse, n_vox):
    """Mean-reduce per-point ``values`` (N, C) into voxels (n_vox, C). Uses ``np.bincount``
    (a fast C-level scatter-add) per channel rather than ``np.add.at``, which is markedly
    slower at 100k+ points. Bit-identical to the add.at result; the per-channel loop keeps
    peak memory low (no N*C flat-index array), which matters in the RAM-bound dataloader."""
    v = values.reshape(len(values), -1).astype(np.float64)
    C = v.shape[1]
    cnt = np.bincount(inverse, minlength=n_vox).astype(np.float64)
    acc = np.empty((n_vox, C), dtype=np.float64)
    for c in range(C):
        acc[:, c] = np.bincount(inverse, weights=v[:, c], minlength=n_vox)
    return (acc / np.maximum(cnt[:, None], 1.0)).astype(np.float32)


class PTv3Collate:
    """Collate spheres -> PTv3 batch dict."""

    def __init__(self, cfg, device=None, neighbor_limit=None, mix_prob=0.0, **_ignored):
        # neighbor_limit / extra kwargs accepted for drop-in compatibility with the
        # old KPConv MultiscaleCollate call sites; PTv3 voxelisation doesn't use them.
        self.cfg = cfg
        self.grid_size = float(getattr(cfg, "first_subsampling_dl", 0.1))
        self.num_classes = int(getattr(cfg, "num_classes", 2))
        self.device = device
        self.mix_prob = float(mix_prob)        # Mix3D (Pointcept point_collate_fn); 0 disables (val/test)

    def __call__(self, samples: List[dict]) -> dict:
        coords, feats, labels, gcoords = [], [], [], []
        lengths0 = []
        offset_acc = 0
        offsets = []
        patches = []
        points_per_cloud = []   # for the visualizer's per-sphere path
        ndsms = []
        regimes = []
        has_patch = ("dtm_patch" in samples[0])
        has_ndsm = ("ndsm" in samples[0])
        has_regime = ("regime" in samples[0])

        for s in samples:
            pts = np.asarray(s["points"], dtype=np.float32)        # tile-local xyz
            f = np.asarray(s["features"], dtype=np.float32)
            lab = np.asarray(s["labels"], dtype=np.int64)
            gc, inv = _voxelize(pts, self.grid_size)
            n_vox = int(inv.max()) + 1 if len(inv) else 0

            # one representative per voxel: mean coord/feat, majority label,
            # representative grid_coord (first point's voxel coords)
            vcoord = _voxel_reduce_mean(pts, inv, n_vox)
            vfeat = _voxel_reduce_mean(f, inv, n_vox)
            vlab = _majority_label(lab, inv, n_vox, self.num_classes, int(IGNORE_LABEL))
            # representative integer grid coords per voxel
            vgrid = np.zeros((n_vox, 3), dtype=np.int64)
            vgrid[inv] = gc  # last write wins; all points in a voxel share gc up to grid floor
            # (points in the same voxel have identical floor(gc) by construction)

            coords.append(vcoord)
            feats.append(vfeat)
            gcoords.append(vgrid)
            labels.append(vlab)
            lengths0.append(n_vox)
            offset_acc += n_vox
            offsets.append(offset_acc)
            points_per_cloud.append(vcoord)
            if has_patch:
                patches.append(np.asarray(s["dtm_patch"], dtype=np.float32))
            if has_regime:
                regimes.append(np.asarray(s["regime"], dtype=np.float32).reshape(-1))
            if has_ndsm:
                # NaN-aware per-voxel mean of the continuous nDSM target; voxels with
                # no finite member stay NaN (skipped by the GrounDiff regression loss).
                nd = np.asarray(s["ndsm"], dtype=np.float32).reshape(-1)
                fin = np.isfinite(nd)
                sums = np.zeros(n_vox, np.float64)
                cnts = np.zeros(n_vox, np.float64)
                if fin.any():
                    np.add.at(sums, inv[fin], nd[fin].astype(np.float64))
                    np.add.at(cnts, inv[fin], 1.0)
                vnd = np.where(cnts > 0.0, sums / np.maximum(cnts, 1.0), np.nan)
                ndsms.append(vnd.astype(np.float32))

        coord = np.concatenate(coords, 0) if coords else np.zeros((0, 3), np.float32)
        feat = np.concatenate(feats, 0) if feats else np.zeros((0, 1), np.float32)
        grid_coord = np.concatenate(gcoords, 0) if gcoords else np.zeros((0, 3), np.int64)
        label = np.concatenate(labels, 0) if labels else np.zeros((0,), np.int64)

        # Mix3d (Nekrasov et al. 2021; Pointcept point_collate_fn): with prob mix_prob,
        # MERGE clouds pairwise by rewriting offsets only -- cat(offsets[1:-1:2], offsets[-1]).
        # The backbone then treats each pair as ONE scene; points/labels/features are untouched
        # and cloud_lengths_0 stays per-ORIGINAL-cloud, so the per-point prior-raster gather and
        # the per-scene regime head are unaffected. Train only (mix_prob=0 for val/test).
        if self.mix_prob > 0.0 and len(offsets) >= 2 and _random.random() < self.mix_prob:
            offsets = offsets[1:-1:2] + [offsets[-1]]

        def t(x, dtype):
            tt = torch.from_numpy(np.ascontiguousarray(x)).to(dtype)
            return tt.to(self.device) if self.device is not None else tt

        batch = {
            "coord": t(coord, torch.float32),
            "grid_coord": t(grid_coord, torch.long),
            "feat": t(feat, torch.float32),
            "offset": t(np.asarray(offsets, dtype=np.int64), torch.long),
            "labels": t(label, torch.long),
            "cloud_lengths_0": t(np.asarray(lengths0, dtype=np.int64), torch.long),
            # visualizer compatibility: per-cloud level-0 points + lengths
            "points": [t(np.concatenate(points_per_cloud, 0), torch.float32)] if points_per_cloud
                      else [t(np.zeros((0, 3), np.float32), torch.float32)],
            "lengths": [t(np.asarray(lengths0, dtype=np.int64), torch.long)],
            "origins": [s["origin"] for s in samples],
            "paths": [s["path"] for s in samples],
        }
        if has_patch and patches:
            # patches are (C, H, W) (multi-channel prior raster) or (H, W) (legacy DTM)
            arr = np.stack([p if p.ndim == 3 else p[None, :, :] for p in patches], 0)
            batch["dtm_patches"] = t(arr.astype(np.float32), torch.float32)
            # whole-scene mode: the patch covers a scene_block_size window centred on the scene,
            # so the sampler maps centred coords (+T/2) over [0, T]. In sphere mode the dataset
            # centres coords in [-R, R]; the model defaults raster_tile_size to 2*in_radius, so
            # only set this override when running the whole-scene (scene_mode) pipeline.
            if bool(getattr(self.cfg, "scene_mode", True)):
                batch["raster_tile_size"] = float(getattr(self.cfg, "scene_block_size", 64.0))
        if has_ndsm and ndsms:
            batch["ndsm"] = t(np.concatenate(ndsms, 0).astype(np.float32), torch.float32)
        if has_regime and regimes:
            batch["regime"] = t(np.stack(regimes, 0).astype(np.float32), torch.float32)  # (B, n_globals)
        return batch
