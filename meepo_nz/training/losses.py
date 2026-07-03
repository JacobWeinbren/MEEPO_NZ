"""
Loss for binary ground / non-ground segmentation.

This IS the MEEPO / Pointcept segmentation criteria: ``CrossEntropyLoss``
(loss_weight 1) + ``LovaszLoss`` (multiclass, loss_weight 1). The CE term is a
plain per-point softmax cross-entropy with mean reduction -- the same unweighted
``reduce_mean`` KPConv's ``segmentation_loss`` uses, hence the historical label,
but it is ``torch.nn.CrossEntropyLoss``, not a KPConv-specific loss. Class
weights are optional and
**off by default**: inverse-frequency weighting up-weights the minority class,
and since ground is usually the majority in NZ terrain it biases the model
toward non-ground (collapsing ground recall). ``inverse_frequency_weights`` is
kept for datasets that explicitly opt in (``loss_class_balance="inverse"``).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..utils.laz_io import IGNORE_LABEL


def inverse_frequency_weights(label_counts: Sequence[float], num_classes: int = 2,
                              clip: float = 10.0) -> torch.Tensor:
    counts = np.asarray(label_counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    w = 1.0 / freq
    w = w / w.mean()                 # normalise so mean weight ~ 1
    w = np.clip(w, 1.0 / clip, clip)
    return torch.tensor(w, dtype=torch.float32)


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovász extension of the Jaccard loss (Berman et al. 2018)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Multi-class Lovász-softmax over the classes PRESENT in the batch.

    ``probs``  (N, C) per-point softmax probabilities; ``labels`` (N,) class ids.
    Directly optimises the IoU surrogate, which is why PTv3 / LitePT train with
    CrossEntropy + Lovász for semantic segmentation."""
    if probs.numel() == 0:
        return probs.sum() * 0.0
    C = probs.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:                      # class absent -> skip ("present" mode)
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


class SegLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 lovasz_weight: float = 0.0):
        super().__init__()
        # ignore_index=IGNORE_LABEL: unclassified points (kept as geometric context)
        # contribute NO loss - they are never trained on.
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=int(IGNORE_LABEL))
        self.lovasz_weight = float(lovasz_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                aux_pred: Optional[torch.Tensor] = None,
                aux_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        # aux_* accepted for a uniform criterion signature (GrounDiffLoss uses them); ignored here.
        loss = self.ce(logits, labels)
        if self.lovasz_weight > 0.0:
            mask = labels != int(IGNORE_LABEL)
            if mask.any():
                probs = torch.softmax(logits[mask].float(), dim=1)
                loss = loss + self.lovasz_weight * lovasz_softmax_flat(probs, labels[mask])
        return loss


GROUND_INDEX = 1                              # binary labels: ground = 1, non-ground = 0


# ---------------------------------------------------------------------------
# GrounDiff loss  (Dhaouadi et al., 2025, "GrounDiff", Eqs. 11-12, 14).
#
# GrounDiff reformulates ground-surface extraction as a primarily REGRESSION
# task: the network regresses the nDSM r = z - DTM(x,y) (height above the bare
# earth) and only secondarily predicts a ground mask. Its objective is
#
#       L = lambda1*L1 + lambda2*L2 + lambda_grad*L_grad + lambda_c*Lc     (Eq. 11)
#       L1 = ||r_hat - r||_1 ,   L2 = ||r_hat - r||_2^2                    (Eq. 12)
#       Lc = BCE(sigmoid(l), ground_mask)                                  (Eq. 14)
#       lambda1 = lambda2 = 1.0,  lambda_grad = 0.1,  lambda_c = 0.1
#
# WHY THIS FIXES THE COLLAPSE.  A pure classification (mask) loss on an
# imbalanced binary task -- which is all CE+Lovasz is -- has a strong
# "predict the majority class everywhere" local minimum (mean CE ~ the base
# rate), so the model collapses to all-ground (non-ground IoU -> 0). The
# continuous nDSM regression has NO such shortcut: predicting 0 for a 15 m tree
# costs ||15||_1 / ||15||_2^2, so the shared features are forced to encode
# height-above-ground, which is precisely what separates ground from non-ground.
#
# ADAPTATION TO A POINT CLASSIFIER (documented deviations from the paper):
#   * GrounDiff is a raster U-Net diffusion model; we transfer only the
#     regression-dominant LOSS (the part that prevents collapse), not the
#     diffusion process or the gating fusion G (Eq. 5), which are specific to
#     iterative raster generation.
#   * The classification term reuses MEEPO's CE(+Lovasz) (the backbone
#     paper) as Lc; we keep it at full weight by default (``cls_weight=1.0``)
#     rather than the paper's 0.1, because per-point IoU is OUR deliverable
#     metric and the mask head must train well. Set ``groundiff_cls_weight=0.1``
#     for the strict-GrounDiff weighting.
#   * The edge-aware regulariser L_grad (Eq. 13) is a raster-grid gradient term
#     (||grad g||) with no analogue on unstructured points, and is the smallest
#     term (lambda_grad=0.1); it is omitted.
#   * The nDSM target is normalised by ``ndsm_scale`` so L1/L2 are O(1) and
#     balanced against CE -- GrounDiff likewise regresses normalised elevations.
# ---------------------------------------------------------------------------
class GrounDiffLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 lovasz_weight: float = 0.0, l1_weight: float = 1.0,
                 l2_weight: float = 1.0, cls_weight: float = 1.0,
                 ndsm_scale: float = 10.0):
        super().__init__()
        self.cls = SegLoss(class_weights=class_weights, lovasz_weight=lovasz_weight)
        self.l1_weight = float(l1_weight)
        self.l2_weight = float(l2_weight)
        self.cls_weight = float(cls_weight)
        self.ndsm_scale = float(ndsm_scale)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                reg_pred: Optional[torch.Tensor] = None,
                reg_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Lc: MEEPO per-point classification (CE [+ Lovasz]) = GrounDiff mask term.
        loss = self.cls_weight * self.cls(logits, labels)
        if reg_pred is None or reg_target is None:
            return loss
        # L1 + L2 on the normalised nDSM, over points with a valid (finite) target.
        r_hat = reg_pred.reshape(-1).float()
        r = reg_target.reshape(-1).float()
        mask = torch.isfinite(r) & (labels.reshape(-1) != int(IGNORE_LABEL))
        if not bool(mask.any()):
            return loss
        diff = (r_hat[mask] - r[mask]) / self.ndsm_scale
        l1 = diff.abs().mean()
        l2 = (diff * diff).mean()
        reg = self.l1_weight * l1 + self.l2_weight * l2
        if torch.isfinite(reg):
            loss = loss + reg
        return loss
