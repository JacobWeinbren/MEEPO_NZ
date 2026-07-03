"""Batch device movement for the MEEPO pipeline.

The KPConv multi-scale neighbour-pyramid builder that used to live here is gone:
PTv3 needs no kd-tree neighbour pyramid (it works on voxel coords + a serialized
order computed inside the model), so PTv3Collate produces a flat batch dict and
this module only moves it to the device. ``move_batch`` is imported unchanged by
the reused ``training/trainer.py``.
"""
from __future__ import annotations

from typing import Dict

import torch


def move_batch(batch: Dict, device: torch.device, cfg=None) -> Dict:
    """Move every tensor (and tensors inside list values) in ``batch`` to
    ``device`` (async for pinned CUDA). Non-tensor entries pass through."""
    nb = device.type == "cuda"
    out = {}
    for k, v in batch.items():
        if isinstance(v, list):
            out[k] = [t.to(device, non_blocking=nb) if torch.is_tensor(t) else t for t in v]
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=nb)
        else:
            out[k] = v
    return out
