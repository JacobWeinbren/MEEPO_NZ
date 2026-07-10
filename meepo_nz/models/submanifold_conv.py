"""Pure-PyTorch submanifold sparse 3-D convolution.

PTv3 uses ``spconv.SubMConv3d`` in exactly two places: the embedding stem
(kernel 5) and the conditional positional encoding (xCPE, kernel 3) inside every
transformer block.  ``spconv`` is a heavy CUDA extension whose prebuilt wheels
lag new GPU architectures (a real problem on a Blackwell RTX PRO 6000 / sm_120)
and which cannot run on CPU for a smoke test.  To keep the model buildable
everywhere we re-implement the one operation we need.

A *submanifold* conv keeps the active-site set fixed (output computed only at
input voxels, no dilation).  For an odd kernel of size ``K`` there are ``K**3``
integer offsets; for each offset ``o`` and each output voxel ``v`` we gather the
feature of voxel ``v + o`` if it exists in the active set and accumulate
``W_o @ feat[v+o]``.  Voxels are looked up via a packed-key + ``searchsorted``
hash (each ``(batch, gx, gy, gz)`` is unique after grid sampling), so the whole
thing is ``O(N * K**3)`` gathers with no Python-level per-point work.

Numerically this matches a standard submanifold conv with zero padding for
missing neighbours; it is device-agnostic (CPU/CUDA) and ``torch.compile``-safe.
"""
from __future__ import annotations

import itertools
import math
import os

import torch
import torch.nn as nn

from .point_structure import Point, PointModule
from .scatter_gather import gather_rows

# Use real spconv when it is importable (on Blackwell that means building the
# RayYoh/cumm + kenomo/spconv forks from source -- see setup_blackwell.sh). Its
# rulebook kernels are ~10-50x faster than any pure-PyTorch sparse conv. When
# spconv is absent (CPU smoke test, or no Blackwell build) we transparently fall
# back to the vectorised clean-PyTorch path below. Set POINT_MOE_DISABLE_SPCONV=1
# to force the clean path even when spconv is installed.
try:
    import spconv.pytorch as _spconv  # type: ignore
    _HAS_SPCONV = True
except Exception:
    _spconv = None
    _HAS_SPCONV = False
_USE_SPCONV = _HAS_SPCONV and os.environ.get("POINT_MOE_DISABLE_SPCONV", "") not in ("1", "true", "True")


def _pack_keys(batch: torch.Tensor, grid: torch.Tensor):
    """Pack (batch, x, y, z) integer coords into a single sortable int64 key.

    Returns ``(keys, base, mins)`` where ``base`` is the per-axis multiplier span
    so neighbour keys can be recomputed from shifted coords with the same packing.
    """
    g = grid.long()
    mins = g.min(dim=0).values
    g0 = g - mins  # non-negative
    span = (g0.max(dim=0).values + 1).clamp(min=1)
    # span per axis; build mixed-radix base [x, y, z] then prepend batch
    sx, sy, sz = int(span[0]), int(span[1]), int(span[2])
    b = batch.long()
    # key = ((batch * sx + x) * sy + y) * sz + z
    keys = ((b * sx + g0[:, 0]) * sy + g0[:, 1]) * sz + g0[:, 2]
    base = (sx, sy, sz)
    return keys, base, mins


def _neighbour_index(sorted_keys, sort_idx, query_keys):
    """Map ``query_keys`` to indices into the original (unsorted) point array,
    returning ``(idx, valid)`` where ``idx`` is the source point index (0 where
    invalid) and ``valid`` a boolean mask of which queries hit an active voxel."""
    pos = torch.searchsorted(sorted_keys, query_keys)
    pos = pos.clamp(max=sorted_keys.numel() - 1)
    hit = sorted_keys[pos] == query_keys
    idx = sort_idx[pos]
    idx = torch.where(hit, idx, torch.zeros_like(idx))
    return idx, hit


class SubMConv3d(PointModule):
    """Submanifold sparse 3-D convolution (kernel ``K**3``), pure PyTorch.

    Drop-in for ``spconv.pytorch.SubMConv3d`` for the cases PTv3 needs: same
    ``in_channels``/``out_channels``/``kernel_size``/``bias`` semantics, operating
    on a :class:`Point` (reads ``point.grid_coord``/``batch``/``feat``).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 indice_key=None, **_ignored):
        super().__init__()
        assert kernel_size % 2 == 1, "submanifold kernel size must be odd"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.use_spconv = _USE_SPCONV

        if self.use_spconv:
            # Real spconv submanifold conv: one fused rulebook kernel for all K**3
            # offsets, rulebook reused across calls sharing `indice_key` (per stage).
            self.sp = _spconv.SubMConv3d(
                in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key
            )
            return

        # ---- clean-PyTorch fallback (CPU / no Blackwell spconv build) ----------
        offsets = list(itertools.product(range(-(kernel_size // 2), kernel_size // 2 + 1), repeat=3))
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long), persistent=False
        )
        self.K3 = len(offsets)
        # weight: (K3, out, in); init like spconv/torch conv (Kaiming-uniform)
        self.weight = nn.Parameter(torch.empty(self.K3, out_channels, in_channels))
        fan_in = in_channels * self.K3
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def _neighbour_idx(self, g0_s, b_s, off, sorted_keys, sort_idx, sx, sy, sz, feat_dtype):
        """Per-row submanifold neighbour lookup for a slice of points.

        g0_s: (c, 3) grid coords (mins-subtracted); b_s: (c,) batch ids.
        Returns (idx, valid): idx (c, K3) row ids into ``feat`` (0 where missing),
        valid (c, K3) {0,1} mask. Identical maths whether ``c == N`` or a chunk.
        """
        c = g0_s.shape[0]
        nbr = g0_s.unsqueeze(1) + off.unsqueeze(0)                 # (c, K3, 3)
        nx, ny, nz = nbr[..., 0], nbr[..., 1], nbr[..., 2]
        inb = (nx >= 0) & (nx < sx) & (ny >= 0) & (ny < sy) & (nz >= 0) & (nz < sz)
        qkeys = ((b_s.unsqueeze(1) * sx + nx) * sy + ny) * sz + nz   # (c, K3)
        qkeys = torch.where(inb, qkeys, torch.full_like(qkeys, -1))
        flat_q = qkeys.reshape(-1)
        pos = torch.searchsorted(sorted_keys, flat_q).clamp(max=sorted_keys.numel() - 1)
        hit = sorted_keys[pos] == flat_q
        idx = torch.where(hit, sort_idx[pos], torch.zeros_like(pos)).reshape(c, self.K3)
        valid = (hit.reshape(c, self.K3) & inb).to(feat_dtype)      # (c, K3)
        return idx, valid

    def _conv(self, grid_coord, batch, feat):
        N = feat.shape[0]
        device = feat.device
        keys, (sx, sy, sz), mins = _pack_keys(batch, grid_coord)
        sort_idx = torch.argsort(keys)
        sorted_keys = keys[sort_idx]
        g0 = grid_coord.long() - mins
        b = batch.long()
        off = self.offsets.to(device)                          # (K3, 3)
        # weight (K3, out, in) -> (K3*in, out) so out[n,o] = sum_{k,i} gathered[n,k,i]*W[k,o,i]
        w = self.weight.permute(0, 2, 1).reshape(self.K3 * self.in_channels, self.out_channels)

        # We build all K3 neighbour queries together (ONE searchsorted, no per-offset
        # `bool(hit.any())` GPU->CPU sync -- the previous looped version synced K3 times
        # per conv block). The neighbour tensor (N, K3, in) is, however, the single
        # largest allocation in the whole model: at full-resolution decoder stages
        # (N~2e5, K3 up to 125, in up to 512) it is several GB and is precisely what
        # forces grad-checkpointing on a 96 GB card (it is the tensor that OOMs first
        # with --no-grad-checkpoint). So when it would be large we TILE the gather+GEMM
        # over points: each chunk materialises only (chunk, K3, in) then does one GEMM.
        # This is a row-block-decomposed matmul -> bit-for-bit the same result as the
        # one-shot path, still sync-free, just a handful more launches. Small layers
        # stay on the single-GEMM fast path, so nothing slows down where memory is fine.
        max_elems = int(getattr(self, "conv_chunk_max_elems", 192_000_000))
        if N * self.K3 * self.in_channels <= max_elems:
            idx, valid = self._neighbour_idx(g0, b, off, sorted_keys, sort_idx, sx, sy, sz, feat.dtype)
            gathered = gather_rows(feat, idx) * valid.unsqueeze(-1)         # (N, K3, in)
            out = gathered.reshape(N, self.K3 * self.in_channels) @ w
        else:
            chunk = max(1, max_elems // (self.K3 * self.in_channels))
            outs = []
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                idx, valid = self._neighbour_idx(g0[s:e], b[s:e], off, sorted_keys, sort_idx, sx, sy, sz, feat.dtype)
                gathered = gather_rows(feat, idx) * valid.unsqueeze(-1)     # (c, K3, in)
                outs.append(gathered.reshape(e - s, self.K3 * self.in_channels) @ w)
            out = torch.cat(outs, 0)
        if self.bias is not None:
            out = out + self.bias
        return out

    @torch.compiler.disable
    def _spconv_forward(self, point):
        # Wrap the Point as a spconv SparseConvTensor, run the rulebook conv, unwrap.
        # SubMConv3d preserves the active-site order, so output features align 1:1
        # with point.feat (no re-scatter). Symmetric 3x3x3 kernel -> axis order is
        # irrelevant to a learned conv, so [batch, x, y, z] is fine.
        feat = point.feat
        if feat.shape[0] == 0:
            # Empty input (e.g. a degenerate val crop): spconv's algorithm tuner
            # raises "can't find suitable algorithm for 0" on a zero-point conv.
            return feat.new_zeros((0, self.out_channels))
        in_dtype = feat.dtype
        gc = point.grid_coord
        gc = gc.int() if gc.dtype != torch.int32 else gc
        gc = gc - gc.min(dim=0).values                      # ensure non-negative indices
        spatial = (gc.max(dim=0).values + 1)
        spatial_shape = [int(spatial[0]), int(spatial[1]), int(spatial[2])]
        bidx = point.batch.to(torch.int32).view(-1, 1)
        indices = torch.cat([bidx, gc], dim=1).contiguous()
        bsz = int(point.batch.max().item()) + 1
        # spconv's implicit-GEMM kernels are tuned for fp32/fp16. Under bf16 autocast
        # the tuner can fail to find any algorithm (-> the eval-time crash) and bf16
        # accumulation in the sparse GEMM is numerically fragile (-> mid-epoch NaNs).
        # Run the conv in fp32 with autocast disabled, then cast back; the rest of the
        # network stays in the autocast dtype. Sparse conv is memory-bound, so the
        # fp32 cost here is small relative to correctness.
        with torch.amp.autocast(feat.device.type, enabled=False):
            x = _spconv.SparseConvTensor(feat.float(), indices, spatial_shape, bsz)
            out = self.sp(x).features
        return out.to(in_dtype)

    def forward(self, point):
        if not isinstance(point, Point):
            raise TypeError("SubMConv3d expects a Point")
        if self.use_spconv and point.feat.is_cuda:
            point.feat = self._spconv_forward(point)
        elif self.use_spconv:
            # spconv implicit-GEMM is GPU-only and we hold only spconv weights here,
            # so give an actionable error rather than spconv's cryptic assertion.
            raise RuntimeError(
                "spconv SubMConv3d requires CUDA tensors (implicit GEMM is GPU-only). "
                "Run with --device cuda, or set POINT_MOE_DISABLE_SPCONV=1 to use the "
                "clean-PyTorch conv (the CPU smoke test does this automatically)."
            )
        else:
            feat = point.feat
            # LEVEL-BELOW-'layer' CHECKPOINTING (accuracy-exact): the neighbour
            # gather (N, K3, in) is the model's single largest saved-for-backward
            # tensor and tiling only bounds the FORWARD transient -- autograd still
            # keeps every tile's GEMM input. When that total is big, recompute the
            # whole gather+GEMM in backward instead of storing it. Same math,
            # ~one extra conv-forward per backward. POINT_MOE_CONV_CKPT=0 disables.
            big = point.feat.shape[0] * self.K3 * self.in_channels > int(
                getattr(self, "conv_ckpt_min_elems", 64_000_000))
            if (big and torch.is_grad_enabled() and feat.requires_grad
                    and os.environ.get("POINT_MOE_CONV_CKPT", "1") != "0"):
                from torch.utils.checkpoint import checkpoint as _ckpt
                # offset2batch runs under torch.inference_mode() -> point.batch is an
                # inference tensor, which checkpoint refuses to save as an input.
                # Clone the (small, integer) index tensors out of inference mode.
                gc = point.grid_coord.clone() if point.grid_coord.is_inference() else point.grid_coord
                bt = point.batch.clone() if point.batch.is_inference() else point.batch
                point.feat = _ckpt(self._conv, gc, bt, feat, use_reentrant=False)
            else:
                point.feat = self._conv(point.grid_coord, point.batch, feat)
        return point
