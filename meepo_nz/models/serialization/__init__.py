"""Space-filling-curve serialization for point clouds (Z-order + Hilbert).

Copied verbatim from PTv3 / MEEPO (pure PyTorch, no CUDA dependency).
``encode`` maps integer ``grid_coord`` (+ batch id) to a 1-D sort key; sorting
by that key linearises the 3-D voxels into a locality-preserving sequence that
the serialized attention then partitions into fixed-size patches.
"""
from .default import (
    encode,
    decode,
    z_order_encode,
    z_order_decode,
    hilbert_encode,
    hilbert_decode,
)

__all__ = [
    "encode", "decode",
    "z_order_encode", "z_order_decode",
    "hilbert_encode", "hilbert_decode",
]
