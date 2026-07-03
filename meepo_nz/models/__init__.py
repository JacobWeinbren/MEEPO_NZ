"""Clean-PyTorch point-cloud ground-segmentation model package.

``build_meepo(cfg)`` returns a :class:`MeepoSeg`: the MEEPO (CNN-Mamba)
backbone + previous-year-classification prior-raster branch + 2-class head,
running on CPU and on Blackwell GPUs without spconv / flash-attn / torch_scatter.
"""
from .segmentation_model import MeepoSeg, build_meepo

__all__ = ["MeepoSeg", "build_meepo"]
