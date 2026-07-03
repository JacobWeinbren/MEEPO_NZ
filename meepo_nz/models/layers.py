"""Minimal layer utilities vendored to avoid a timm dependency.

PTv3 uses exactly one timm primitive - ``DropPath`` (stochastic depth) - so we
inline the standard implementation rather than depend on the whole library.
Identical behaviour to ``timm.layers.DropPath``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False,
              scale_by_keep: bool = True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast over all but batch dim
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample (when applied in the main path
    of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"
