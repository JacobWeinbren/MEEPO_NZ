"""Previous-year *classification* raster branch (Deviation A).

An earlier build fed the previous survey only as a single-channel
DTM (bare-earth height) raster, or as a per-point scalar ``z - DTM_prev``.  This
branch instead consumes a **multi-channel raster of the previous year's
classification**, rasterised at 1 m from last year's *classified* point cloud:

    channel 0 : prior DTM        - mean elevation of prior GROUND returns
    channel 1 : prior DSM        - max elevation of all prior returns (top surface)
    channel 2 : prior nDSM       - DSM - DTM  (height of non-ground = veg/structure)
    channel 3 : prior ground-prob- fraction of prior returns classified as ground
    channel 4 : coverage         - 1 where the cell had any prior return, else 0

Design follows **GrounDiff** (Dhaouadi et al., 2025, *Diffusion-Based Ground
Surface Generation from DSMs*) and the DSM->DTM raster-CNN line (DSM2DTM, the
ALS2DTM rasterisation GAN, multi-scale DTM fusion, physically-informed
autoencoders).  GrounDiff's key findings that we adopt:

  * **residual, not absolute** - predict a correction to a height prior rather
    than absolute elevation (their ablation: +17%).  We mean-centre the height
    channels (offset-invariance) and the head predicts a *residual* to the prior
    DTM; we also pass the per-point ``z - DTM_prev`` scalar elsewhere.
  * **dual head + confidence gating** - GrounDiff outputs a residual AND a
    per-pixel ground-confidence, fused by ``G = sigma(l)*s + (1-sigma(l))*(s - r)``
    (removing the gate caused a 12x error blow-up).  Here ``s`` is the prior DTM,
    so the gated *refined* bare-earth height is the faithful Eq. 5 form
    ``g_ref = s - (1-sigma(l))*r = sigma(l)*DTM + (1-sigma(l))*(DTM - resid)`` -
    the prior DTM anchor is always present and the learned residual only corrects
    it where the cell is not confidently unchanged ground.

A small fully-convolutional encoder produces, per cell, (i) ``out_dim`` terrain
context features, (ii) a refined ground-height ``g_ref`` (relative to the patch
mean), and (iii) the ground-confidence ``sigma(l)``.  All three are bilinearly
sampled at every point's planar position and concatenated to the per-point
features before the PTv3 stem - giving the network a strong, terrain-aware prior
on *where the ground should be* (exactly the user's intent), plus local slope /
valley / ridge / canopy context that a single height number cannot convey.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm2d(ch, norm="bn", momentum=0.02):
    """2-D norm for the raster CNN. BatchNorm2d by default; a batch-independent
    GroupNorm when norm in {ln, gn} (safe under micro-batch 1 + grad-accum)."""
    if str(norm).lower() in ("ln", "layer", "layernorm", "gn", "group", "groupnorm"):
        g = next((x for x in (8, 4, 2, 1) if ch % x == 0), 1)
        return nn.GroupNorm(num_groups=g, num_channels=ch)
    return nn.BatchNorm2d(ch, momentum=momentum)


class _ResBlock(nn.Module):
    def __init__(self, ch, use_bn=True, bn_momentum=0.02, norm="bn"):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=not use_bn)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=not use_bn)
        self.n1 = _norm2d(ch, norm, bn_momentum) if use_bn else nn.Identity()
        self.n2 = _norm2d(ch, norm, bn_momentum) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(x + h)


class PriorRasterEncoder(nn.Module):
    """2-D CNN over the multi-channel previous-year-classification raster.

    Parameters
    ----------
    in_channels : number of raster channels (default 5: DTM, DSM, nDSM, gprob, cover)
    out_dim     : number of terrain-context feature channels produced
    height_channels : indices of the height-like channels to mean-centre for
                      offset invariance (default DTM, DSM, nDSM)
    dtm_channel : index of the prior-DTM channel used as the gating prior ``s``
    """

    def __init__(self, in_channels: int = 5, out_dim: int = 8, mid: int = 32,
                 use_bn: bool = True, bn_momentum: float = 0.02,
                 height_channels: Sequence[int] = (0, 1, 2), dtm_channel: int = 0,
                 use_gating: bool = True, norm: str = "bn"):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.height_channels = tuple(height_channels)
        self.dtm_channel = dtm_channel
        self.use_gating = use_gating
        # total sampled width per point = terrain features + (g_ref, confidence) if gating
        self.sample_dim = out_dim + (2 if use_gating else 0)

        def conv(cin, cout):
            layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(_norm2d(cout, norm, bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.stem = nn.Sequential(*conv(in_channels, mid))
        self.body = nn.Sequential(_ResBlock(mid, use_bn, bn_momentum, norm),
                                   _ResBlock(mid, use_bn, bn_momentum, norm))
        self.head_feat = nn.Conv2d(mid, out_dim, 1)
        if use_gating:
            self.head_conf = nn.Conv2d(mid, 1, 1)    # ground-confidence logits (GrounDiff l)
            self.head_resid = nn.Conv2d(mid, 1, 1)   # residual correction to prior DTM (GrounDiff r)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, sample_dim, H, W) terrain feature map."""
        if patches.dim() == 3:
            patches = patches.unsqueeze(1)
        x = patches.clone()
        # offset-invariance: subtract each patch's own mean from the height channels
        # (fp32 for AMP stability). Encodes terrain *shape*, robust to a constant
        # vertical bias between surveys and to the augmentation's vertical scaling.
        hidx = [c for c in self.height_channels if c < x.shape[1]]
        if hidx:
            hsel = x[:, hidx, :, :].float()
            hmean = hsel.mean(dim=(-2, -1), keepdim=True)
            x[:, hidx, :, :] = (hsel - hmean).to(x.dtype)
            dtm_rel = x[:, self.dtm_channel:self.dtm_channel + 1, :, :]  # prior DTM (mean-centred)
        else:
            dtm_rel = x[:, :1, :, :] * 0.0

        h = self.body(self.stem(x))
        feat = self.head_feat(h)
        if not self.use_gating:
            return feat
        conf = torch.sigmoid(self.head_conf(h))      # sigma(l): ground confidence in [0,1]
        resid = self.head_resid(h)                   # r: residual correction
        # GrounDiff gate (Eq. 5) with s = prior DTM:
        #   G = sigma(l)*s + (1-sigma(l))*(s - r) = s - (1-sigma(l))*r
        # The prior-DTM anchor `s` is ALWAYS present; the learned residual only
        # *corrects* it, weighted by (1 - confidence). sigma->1 trusts the prior
        # DTM as-is (ground unchanged since last survey); sigma->0 applies the full
        # correction (ground changed, e.g. erosion / earthworks). This is the
        # faithful Eq. 5 form -- gating `s` by sigma instead (the previous
        # `conf*dtm_rel - (1-conf)*resid`) dropped the prior anchor entirely in
        # low-confidence cells, contradicting both Eq. 5 and the "trust prior where
        # confident, correct where not" intent. (GrounDiff ablation: removing the
        # gate is ~12x worse; predicting a residual rather than absolute height ~+17%.)
        g_ref = dtm_rel - (1.0 - conf) * resid
        return torch.cat([feat, g_ref, conf], dim=1)


def sample_raster_features(feat_map: torch.Tensor, points0: torch.Tensor,
                           lengths0: torch.Tensor, tile_size: float) -> torch.Tensor:
    """Bilinearly sample per-cloud raster features at each point's (x, y).

    ``feat_map`` : (B, C, H, W)   one patch per cloud in the batch
    ``points0``  : (N_total, 3)   tile-local coords (x,y centred in [-R, R] or [0, tile])
    ``lengths0`` : (B,)           points contributed by each cloud
    returns      : (N_total, C)

    Manual bilinear gather (numerically equal to grid_sample with
    mode='bilinear', padding_mode='border', align_corners=True), vectorised
    across all points/clouds via a per-point batch index so it stays
    torch.compile-safe. ``points0`` x,y are mapped from [0, tile_size] to pixels;
    callers centre coordinates so x_local in [0, tile_size].
    """
    B, C, H, W = feat_map.shape
    device = feat_map.device
    if points0.shape[0] == 0:
        return points0.new_zeros((0, C))
    lengths = lengths0.to(device=device, dtype=torch.long)
    bidx = torch.repeat_interleave(torch.arange(B, device=device), lengths)

    xy = points0[:, :2].to(feat_map.dtype)
    px = (xy[:, 0] / float(tile_size)) * (W - 1)
    py = (xy[:, 1] / float(tile_size)) * (H - 1)
    px = px.clamp(0.0, W - 1.0)
    py = py.clamp(0.0, H - 1.0)

    x0 = torch.floor(px).long(); x1 = (x0 + 1).clamp(max=W - 1)
    y0 = torch.floor(py).long(); y1 = (y0 + 1).clamp(max=H - 1)
    wx = (px - x0.to(px.dtype)).unsqueeze(1)
    wy = (py - y0.to(py.dtype)).unsqueeze(1)

    f00 = feat_map[bidx, :, y0, x0]
    f01 = feat_map[bidx, :, y0, x1]
    f10 = feat_map[bidx, :, y1, x0]
    f11 = feat_map[bidx, :, y1, x1]
    top = f00 * (1.0 - wx) + f01 * wx
    bot = f10 * (1.0 - wx) + f11 * wx
    return top * (1.0 - wy) + bot * wy
