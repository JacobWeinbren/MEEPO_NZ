"""MEEPO segmentation model for binary ground / non-ground extraction.

Wraps the clean PTv3+MoE backbone with:
  * the previous-year-classification raster branch (Deviation A), whose per-point
    sampled terrain features are concatenated to the per-point input features
    (xyz [paper] + return count + return ratio + intensity = Deviation B) before the stem;
  * a plain linear segmentation head over 2 classes (no CLIP language head: a
    single fixed ground/non-ground taxonomy from one sensor, so the label-space
    bridging the official MEEPO head provides is unnecessary).

The Mixture-of-Experts router is **label-free** (faithful to the paper): no
terrain-type id is given, so experts self-organise across the diverse NZ
terrains in the corpus purely from the data.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .meepo import Meepo
from .point_structure import Point
from .prior_raster_encoder import PriorRasterEncoder, sample_raster_features


class MeepoSeg(nn.Module):
    def __init__(self, cfg, in_features_dim: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(getattr(cfg, "num_classes", 2))
        # Full-scene (PTv3-native) mode is point-only: the previous-year raster
        # branch is a per-sphere 64x64 CNN patch (sphere machinery), so it is
        # disabled when scene_mode is on -> the stem is sized for the per-point
        # features alone (no raster sample_dim).
        scene_mode = bool(getattr(cfg, "scene_mode", False))
        # raster branch (Deviation A) is now integrated into the whole-scene pipeline too:
        # in scene mode the prior raster is cropped per block and run through this same CNN.
        self.use_raster = bool(getattr(cfg, "use_dtm_raster", True))
        self.in_radius = float(getattr(cfg, "in_radius", 5.0))

        sample_dim = 0
        if self.use_raster:
            self.raster_encoder = PriorRasterEncoder(
                in_channels=int(getattr(cfg, "prior_raster_channels", 5)),
                out_dim=int(getattr(cfg, "dtm_feat_dim", 8)),
                mid=int(getattr(cfg, "dtm_cnn_mid", 32)),
                use_bn=bool(getattr(cfg, "use_batch_norm", True)),
                bn_momentum=float(getattr(cfg, "batch_norm_momentum", 0.02)),
                use_gating=bool(getattr(cfg, "prior_raster_gating", True)),
                norm=str(getattr(cfg, "norm", "bn")),
            )
            sample_dim = self.raster_encoder.sample_dim
        self.sample_dim = sample_dim

        backbone_in = in_features_dim + sample_dim
        order = tuple(getattr(cfg, "moe_order",
                              ("z", "z-trans", "hilbert", "hilbert-trans")))
        # MEEPO (CNN-Mamba) backbone -- replaces PTv3/LitePT + MoE. Hyperparameters
        # adopted verbatim from the MEEPO ScanNet config (semseg-meepo.py).
        self.backbone = Meepo(
            in_channels=backbone_in, order=order,
            stride=tuple(getattr(cfg, "enc_stride", (2, 2, 2, 2))),
            enc_depths=tuple(getattr(cfg, "meepo_enc_depths", (2, 2, 2, 6, 2))),
            enc_channels=tuple(getattr(cfg, "meepo_enc_channels", (32, 64, 128, 256, 512))),
            dec_depths=tuple(getattr(cfg, "meepo_dec_depths", (2, 2, 2, 2))),
            dec_channels=tuple(getattr(cfg, "meepo_dec_channels", (64, 64, 128, 256))),
            mamba_state_dim=int(getattr(cfg, "mamba_state_dim", 1)),
            mamba_conv_dim=int(getattr(cfg, "mamba_conv_dim", 4)),
            mamba_expand_factor=int(getattr(cfg, "mamba_expand_factor", 3)),
            mlp_ratio=float(getattr(cfg, "mlp_ratio", 3)),
            drop_path=float(getattr(cfg, "drop_path_rate", 0.3)),
            shuffle_orders=bool(getattr(cfg, "shuffle_orders", True)),
            stem_kernel_size=int(getattr(cfg, "stem_kernel_size", 5)),
            n_directions=int(getattr(cfg, "mamba_directions", 2)),
            ssm_backend=str(getattr(cfg, "ssm_backend", "auto")),
            grad_checkpointing=bool(getattr(cfg, "grad_checkpointing", False)),
            checkpoint_granularity=str(getattr(cfg, "checkpoint_granularity", "block")),
            norm=str(getattr(cfg, "norm", "ln")),
        )
        head_drop = float(getattr(cfg, "head_dropout", 0.0))
        self.seg_head = nn.Sequential(
            nn.Dropout(head_drop) if head_drop > 0 else nn.Identity(),
            nn.Linear(self.backbone.out_channels, self.num_classes),
        )
        # GrounDiff auxiliary regression head (Dhaouadi et al., 2025): predicts the
        # CONTINUOUS nDSM r = z - DTM(x,y) (height above bare earth) per point. The
        # dense height target has no majority-class shortcut, so regressing it forces
        # the shared features to encode height-above-ground and prevents the
        # predict-all-ground collapse of a classification-only loss. Train-only.
        self.use_gdreg = bool(getattr(cfg, "use_groundiff_regression", False))
        if self.use_gdreg:
            self.reg_head = nn.Linear(self.backbone.out_channels, 1)
        self._reg_pred = None
        # LEARNED SPAG-DC regime head: pools the backbone's per-scene features and
        # regresses the SPAG-DC control globals (theta0/alpha/beta/n_sigma/base_res/
        # min_floor), squashed into their valid box. Supervised by an "oracle" target
        # from each scene's GT-ground terrain (see inference/spag_dc.oracle_regime_globals).
        # Runs in train AND eval (inference feeds the predicted globals to spag_dc_refine).
        self.spag_learned = bool(getattr(cfg, "spag_learned", False))
        # The regime head is fed BOTH the data (pooled backbone features) AND the model's
        # own outputs (per-scene prediction statistics: predicted-ground fraction, height
        # spread of predicted-ground vs all points, density, confidence -- see _pred_stats).
        # This lets a black-box (REINFORCE) calibrator learn per-scene SPAG-DC globals that
        # minimise DTM-RMSE (scripts/10_fit_spag_rl.py); the oracle smooth-L1 is only a warm
        # start. n_pred_stats is fixed by _pred_stats below.
        self._spag_n_pred_stats = 8
        if self.spag_learned:
            from ..inference.spag_dc import SPAG_N_GLOBALS, SPAG_GLOBAL_LO, SPAG_GLOBAL_HI
            hid = int(getattr(cfg, "spag_regime_hidden", 64))
            in_dim = int(self.backbone.out_channels) + self._spag_n_pred_stats
            self.regime_head = nn.Sequential(
                nn.Linear(in_dim, hid), nn.GELU(),
                nn.Linear(hid, int(SPAG_N_GLOBALS)))
            self.register_buffer("_spag_lo", torch.tensor(SPAG_GLOBAL_LO, dtype=torch.float32))
            self.register_buffer("_spag_hi", torch.tensor(SPAG_GLOBAL_HI, dtype=torch.float32))
        self._regime_pred = None
        self._regime_logits = None        # pre-squash head output (policy mean for REINFORCE)
        self._regime_pooled = None        # cached pooled feats (detached) for offline calibration
        self._regime_pred_stats = None    # cached prediction-output stats (detached)

    def squash_globals(self, logits):
        """Map raw head logits -> SPAG-DC globals inside [lo, hi] (the inference mapping)."""
        return self._spag_lo + torch.sigmoid(logits) * (self._spag_hi - self._spag_lo)

    def regime_logits(self, pooled, pred_stats):
        """Head forward on [pooled features ; prediction stats] -> raw globals logits.
        Used by the REINFORCE calibrator on cached (detached) per-scene inputs so only the
        head receives gradient."""
        x = torch.cat([pooled, pred_stats.to(pooled.dtype)], dim=1)
        return self.regime_head(x)

    @staticmethod
    def _pred_stats(logits, coord, counts, bidx):
        """Per-original-cloud summary of the model's OWN predictions + geometry, sync-free.
        Returns (B, 8): [pred-ground fraction, (mean z of pred-ground - scene mean z)/10,
        std z pred-ground /10, std z all /10, z-range/10, centred log point-count,
        mean P(ground), roughness contrast std_zg/std_zall]. Scale/elevation-robust."""
        B = int(counts.numel())
        dev = logits.device
        z = coord[:, 2].to(torch.float32)
        predg = (logits.argmax(dim=-1) == 1).to(torch.float32)           # ground == class 1
        prob_g = torch.softmax(logits.float(), dim=-1)[:, 1]
        cntf = counts.to(torch.float32).clamp(min=1.0)
        zeros = torch.zeros(B, device=dev, dtype=torch.float32)
        cnt_g = zeros.clone().index_add_(0, bidx, predg)
        cnt_gc = cnt_g.clamp(min=1.0)
        sum_zg = zeros.clone().index_add_(0, bidx, predg * z)
        sq_zg = zeros.clone().index_add_(0, bidx, predg * z * z)
        sum_z = zeros.clone().index_add_(0, bidx, z)
        sq_z = zeros.clone().index_add_(0, bidx, z * z)
        sum_pg = zeros.clone().index_add_(0, bidx, prob_g)
        zmax = torch.full((B,), -1e9, device=dev).scatter_reduce(0, bidx, z, reduce="amax", include_self=False)
        zmin = torch.full((B,), 1e9, device=dev).scatter_reduce(0, bidx, z, reduce="amin", include_self=False)
        mean_zg = sum_zg / cnt_gc
        mean_z = sum_z / cntf
        std_zg = (sq_zg / cnt_gc - mean_zg * mean_zg).clamp(min=0.0).sqrt()
        std_z = (sq_z / cntf - mean_z * mean_z).clamp(min=0.0).sqrt()
        frac_g = cnt_g / cntf
        zrange = (zmax - zmin).clamp(min=0.0, max=1e4)
        feats = torch.stack([
            frac_g,
            (mean_zg - mean_z) / 10.0,
            std_zg / 10.0,
            std_z / 10.0,
            zrange / 10.0,
            torch.log(cntf) - 12.0,
            sum_pg / cntf,
            std_zg / (std_z + 1e-3),
        ], dim=1)
        return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch):
        feat = batch["feat"]
        coord = batch["coord"]
        grid_coord = batch["grid_coord"]
        offset = batch["offset"]

        if self.use_raster and batch.get("raster_feats_precomputed", None) is not None:
            # Whole-scene path: per-point raster features were computed upstream the
            # SAME way training builds them (local 2R windows, mean-centred per patch),
            # so a raster-branch model can run PTv3-native whole-scene inference. Just
            # concatenate; do not re-sample from a (nonexistent) per-cloud patch.
            feat = torch.cat([feat, batch["raster_feats_precomputed"].to(feat.dtype)], dim=1)
        elif self.use_raster and ("dtm_patches" in batch) and (batch["dtm_patches"] is not None):
            fmap = self.raster_encoder(batch["dtm_patches"])          # (B, sample_dim, H, W)
            lengths0 = batch["cloud_lengths_0"] if "cloud_lengths_0" in batch \
                else torch.diff(torch.cat([offset.new_zeros(1), offset]))
            # patch spans [center-T/2, center+T/2]; tile-local coords are centred in
            # [-T/2, T/2] -> shift to [0, T] for the sampler. T defaults to 2*in_radius
            # (sphere mode); callers may pass raster_tile_size for other geometries.
            tile = float(batch.get("raster_tile_size") or (2.0 * self.in_radius))
            xy = coord.clone()
            xy[:, 0] = xy[:, 0] + tile / 2.0
            xy[:, 1] = xy[:, 1] + tile / 2.0
            sampled = sample_raster_features(fmap, xy, lengths0, tile)  # (N, sample_dim)
            feat = torch.cat([feat, sampled.to(feat.dtype)], dim=1)

        point = Point(coord=coord, grid_coord=grid_coord, feat=feat, offset=offset)
        if "contexts" in batch and batch["contexts"] is not None:
            point.contexts = batch["contexts"]
        point = self.backbone(point)
        logits = self.seg_head(point.feat)
        # auxiliary GrounDiff nDSM regression: only needed by the loss during training.
        reg_pred = self.reg_head(point.feat).squeeze(-1) if (self.use_gdreg and self.training) else None
        self._reg_pred = reg_pred
        # learned SPAG-DC regime globals: per-scene mean-pool -> head -> squash to box.
        if self.spag_learned:
            # pool per ORIGINAL cloud via cloud_lengths_0 (not offset) so the regime head is
            # robust to Mix3D's pairwise-merged offset (B preds match the B per-cloud targets).
            if batch.get("cloud_lengths_0", None) is not None:
                counts = batch["cloud_lengths_0"].to(offset.device).clamp(min=1)           # (B,)
            else:
                counts = torch.diff(torch.cat([offset.new_zeros(1), offset])).clamp(min=1)  # (B,)
            bidx = torch.repeat_interleave(torch.arange(counts.numel(), device=offset.device), counts)
            pooled = point.feat.new_zeros(counts.numel(), point.feat.shape[1]).index_add_(0, bidx, point.feat)
            pooled = pooled / counts.unsqueeze(1).to(pooled.dtype)
            pred_stats = self._pred_stats(logits.detach(), coord, counts, bidx)   # model's own outputs
            raw = self.regime_head(torch.cat([pooled, pred_stats.to(pooled.dtype)], dim=1))  # (B, n_globals)
            self._regime_logits = raw
            self._regime_pooled = pooled.detach()
            self._regime_pred_stats = pred_stats.detach()
            self._regime_pred = self.squash_globals(raw)
        else:
            self._regime_pred = None
        return logits


def build_meepo(cfg):
    """Instantiate :class:`MeepoSeg` from a config, sizing the per-point input
    feature dimension from the active feature switches."""
    from ..features.shallow_features import expected_feature_dim
    fpp = expected_feature_dim(cfg)
    cfg.in_features_dim = fpp
    return MeepoSeg(cfg, in_features_dim=fpp)
