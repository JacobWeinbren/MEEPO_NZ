"""MEEPO for NZ LiDAR ground extraction.

A clean-PyTorch reimplementation of MEEPO (Point Transformer V3 + sparse
Mixture-of-Experts, ICLR 2026) for binary ground / non-ground segmentation of
New Zealand aerial LiDAR, with a previous-year-classification raster branch
(Deviation A, GrounDiff-informed) and per-point intensity + return-count features
(Deviation B). Runs on Blackwell (sm_120) GPUs and on CPU - it deliberately avoids
spconv / flash-attn / torch_scatter / timm.
"""
__version__ = "1.0.0"
