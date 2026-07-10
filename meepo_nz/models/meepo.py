"""MEEPO backbone -- clean-PyTorch reimplementation of the CNN-Mamba architecture
from "Exploring Contextual Modeling with Linear Complexity for Point Cloud
Segmentation" (ICLR 2025), adapted to this ground-filtering pipeline.

MEEPO keeps PTv3's meta-architecture verbatim -- voxelization, alternating
Z-order/Hilbert serialization, a submanifold-sparse-conv embedding, GridPooling/
GridUnpooling, and a 5-stage encoder / 4-stage decoder -- and replaces the
attention block with a **CNN-Mamba block** (paper Fig. 6a):

    feat += SubMConv3d(feat)            # xCPE: 3x3x3 sparse conv -> local geometry
    feat += BiMamba(RMSNorm(feat))      # Bidirectional SSM -> linear-complexity context
    feat += MLP(LayerNorm(feat))

The Mamba mixer (paper Fig. 6b, "Bidirectional Strided SSM") runs the selective
scan over the serialized token sequence in two directions (forward + backward),
with a non-causal depthwise conv and a gated branch. The single heavy primitive
is the selective scan, dispatched through :mod:`models.ssm` (fused mamba_ssm CUDA
kernel when available, else a pure-PyTorch fallback) -- so the model trains on
Blackwell whether or not the kernel compiles.

Two faithful deviations from the *released* MEEPO code, both for correctness here:
  * the released ``SerializedMamba`` scans ``point.feat`` as one batch-1 sequence
    over ALL points, which (a) crosses cloud boundaries -- the SSM state leaks
    between scenes in a multi-cloud training batch -- and (b) ignores the
    serialization order the paper specifies. We instead reorder by
    ``serialized_order[order_index]`` and scan **per cloud** (via ``offset``),
    then scatter back. Single-cloud inference is identical to the reference.
  * the released code's strided scan directions (paper's "strided") are dead code
    (its loop is ``range(2)``); we match the runnable bidirectional behaviour and
    expose ``n_directions`` (2 = bidirectional, 4 = + strided) for completeness.

No Mixture-of-Experts: MEEPO has none, and this is a single-domain task.
Normalisation is LayerNorm + RMSNorm throughout (the reference uses no BatchNorm),
so the backbone is inherently micro-batch-1 / grad-accum safe.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import DropPath
from .point_structure import Point, PointModule, PointSequential, offset2bincount
from .submanifold_conv import SubMConv3d
from .ssm import selective_scan
from .scatter_gather import gather_rows


# ----------------------------------------------------------------------------- #
#  Shared PTv3 infrastructure (serialization-order pooling, embedding, MLP).
#  Reused verbatim from the PTv3 meta-architecture MEEPO is built on.
# ----------------------------------------------------------------------------- #
def _segment_reduce(src, cluster, n_seg, reduce):
    """Reduce rows of ``src`` into ``n_seg`` groups given per-row ``cluster`` ids."""
    redmap = {"sum": "sum", "mean": "mean", "max": "amax", "min": "amin"}
    op = redmap[reduce]
    out = src.new_zeros((n_seg, src.shape[1]))
    index = cluster.unsqueeze(1).expand(-1, src.shape[1])
    out.scatter_reduce_(0, index, src, reduce=op, include_self=False)
    return out


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm (batch-independent). MEEPO's pre-Mamba norm."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dt)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ----------------------------------------------------------------------------- #
#  Bidirectional Mamba (the SSM mixer core), mirroring MEEPO's Mamba module.
# ----------------------------------------------------------------------------- #
class BiMamba(nn.Module):
    """Multi-directional selective-scan mixer over a single token sequence.

    Operates on (B, L, D). Mirrors the reference MEEPO ``Mamba`` module:
    ``in_proj`` (no bias) maps D -> expand*D and splits into x / z streams of
    ``d_inner//2`` channels; for each scan direction a depthwise NON-causal conv
    (+SiLU) feeds an ``x_proj`` -> (dt, B, C) selective scan; outputs and gates
    are summed across directions, concatenated, and ``out_proj``'d back to D.
    """

    def __init__(self, d_model, d_state=1, d_conv=4, expand=3, n_directions=2,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, dt_scale=1.0,
                 ssm_backend="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.half = self.d_inner // 2
        self.dt_rank = math.ceil(d_model / 16)
        self.n_directions = n_directions
        self.ssm_backend = ssm_backend

        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.A_logs = nn.ParameterList()
        self.Ds = nn.ParameterList()
        self.dt_projs = nn.ModuleList()
        self.x_projs = nn.ModuleList()
        self.conv1d_xs = nn.ModuleList()
        self.conv1d_zs = nn.ModuleList()
        for _ in range(n_directions):
            dt_proj = nn.Linear(self.dt_rank, self.half, bias=True)
            dt_init_std = self.dt_rank ** -0.5 * dt_scale
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand(self.half) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))      # softplus^-1, so softplus(bias)=dt
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True
            self.dt_projs.append(dt_proj)

            self.x_projs.append(nn.Linear(self.half, self.dt_rank + 2 * d_state, bias=False))

            A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.half, 1).contiguous()
            A_log = nn.Parameter(torch.log(A))
            A_log._no_weight_decay = True
            self.A_logs.append(A_log)

            D = nn.Parameter(torch.ones(self.half))
            D._no_weight_decay = True
            self.Ds.append(D)

            # depthwise, NON-causal (padding='same'), no bias (reference: conv_bias//2 == 0)
            self.conv1d_xs.append(nn.Conv1d(self.half, self.half, kernel_size=d_conv,
                                            groups=self.half, padding="same", bias=False))
            self.conv1d_zs.append(nn.Conv1d(self.half, self.half, kernel_size=d_conv,
                                            groups=self.half, padding="same", bias=False))

    def _reindex(self, t, idx):
        # t: (B, half, L); idx: (L,) -> gather along the length axis
        return t[:, :, idx]

    def forward(self, hidden):
        # hidden: (B, L, D)
        B, L, _ = hidden.shape
        xz = self.in_proj(hidden).transpose(1, 2)            # (B, d_inner, L)
        x, z = xz.chunk(2, dim=1)                            # each (B, half, L)

        ys, zs = [], []
        use_dir_ckpt = (torch.is_grad_enabled() and x.requires_grad
                        and L > int(os.environ.get("POINT_MOE_DIR_CKPT_MIN_L", "32768"))
                        and os.environ.get("POINT_MOE_DIR_CKPT", "1") != "0")
        for i in range(self.n_directions):
            if use_dir_ckpt:
                # LEVEL-BELOW-'layer', part 3 (accuracy-exact): recompute this whole
                # direction's internals (convs, SiLUs, reorder copies, scan slices)
                # in backward; only (x, z) and the direction's outputs stay saved.
                from torch.utils.checkpoint import checkpoint as _ckpt
                yi, zi = _ckpt(self._direction, x, z,
                               torch.tensor(i), use_reentrant=False)
                ys.append(yi)
                zs.append(zi)
                continue
            yi, zi = self._direction(x, z, torch.tensor(i))
            ys.append(yi)
            zs.append(zi)

        y = torch.cat([sum(ys), sum(zs)], dim=1)             # (B, d_inner, L)
        return self.out_proj(y.transpose(1, 2))              # (B, L, D)

    def _direction(self, x, z, i_t):
        i = int(i_t)
        B, L = x.shape[0], x.shape[-1]
        if True:
            xi = F.silu(self.conv1d_xs[i](x))                # (B, half, L)
            zi = F.silu(self.conv1d_zs[i](z))
            idx = None
            if i == 1:                                       # backward
                xi, zi = xi.flip(-1), zi.flip(-1)
            elif i == 2:                                     # strided forward (stride-2 interleave)
                idx = torch.cat([torch.arange(0, L, 2, device=xi.device),
                                 torch.arange(1, L, 2, device=xi.device)])
                xi, zi = self._reindex(xi, idx), self._reindex(zi, idx)
            elif i == 3:                                     # strided backward
                idx = torch.cat([torch.arange(0, L, 2, device=xi.device),
                                 torch.arange(1, L, 2, device=xi.device)]).flip(0)
                xi, zi = self._reindex(xi, idx), self._reindex(zi, idx)

            A = -torch.exp(self.A_logs[i].float())                                           # (half,N)

            def _proj_scan(xi_s, h_in, _i=i):
                # pointwise projections + scan on one sequence slice; EXACT under
                # slicing because the state h is carried across slice boundaries.
                Ls = xi_s.shape[-1]
                x_dbl = self.x_projs[_i](xi_s.transpose(1, 2).reshape(B * Ls, self.half))
                dt, Bp, Cp = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt = self.dt_projs[_i](dt).reshape(B, Ls, self.half).transpose(1, 2).contiguous()
                Bp = Bp.reshape(B, Ls, self.d_state).transpose(1, 2).contiguous()
                Cp = Cp.reshape(B, Ls, self.d_state).transpose(1, 2).contiguous()
                return selective_scan(xi_s, dt, A, Bp, Cp, self.Ds[_i].float(), z=None,
                                      delta_bias=self.dt_projs[_i].bias.float(),
                                      delta_softplus=True, backend=self.ssm_backend,
                                      h0=h_in, return_last_state=True)

            slice_len = int(os.environ.get("POINT_MOE_SEQ_SLICE", "0"))
            if slice_len > 0 and L > slice_len and torch.is_grad_enabled() and xi.requires_grad:
                # LEVEL-BELOW-'layer', part 2 (accuracy-exact): the scan's fp32 stream
                # tensors are the largest per-segment allocation; slicing the sequence
                # with checkpointed slices + exact (B,half,N) state carry makes them
                # slice-sized. Gradients flow through the carried state (exact BPTT).
                from torch.utils.checkpoint import checkpoint as _ckpt
                h = xi.new_zeros((B, self.half, self.d_state), dtype=torch.float32)
                y_parts = []
                for s0 in range(0, L, slice_len):
                    e0 = min(s0 + slice_len, L)
                    y_s, h = _ckpt(_proj_scan, xi[:, :, s0:e0].contiguous(), h,
                                   use_reentrant=False)
                    y_parts.append(y_s)
                yi = torch.cat(y_parts, dim=-1)                                              # (B,half,L)
            else:
                yi, _ = _proj_scan(xi, None)                                                 # (B,half,L)

            if i == 1:
                yi, zi = yi.flip(-1), zi.flip(-1)
            elif i in (2, 3):
                rev = torch.argsort(idx)
                yi, zi = self._reindex(yi, rev), self._reindex(zi, rev)
            return yi, zi


class SerializedMamba(PointModule):
    """Mamba mixer over the serialized point sequence, scanned PER CLOUD.

    Reorders ``point.feat`` by ``serialized_order[order_index]`` (a space-filling
    curve), scans each cloud's sub-sequence independently (no cross-cloud state
    leakage), and scatters the result back to the original order.
    """

    def __init__(self, channels, mamba_state_dim, mamba_conv_dim, mamba_expand_factor,
                 order_index=0, n_directions=2, ssm_backend="auto", mixer="mamba1"):
        super().__init__()
        self.channels = channels
        self.order_index = order_index
        if str(mixer).lower() == "mamba3":
            # MEEPO-3: the Mamba-3 recurrence inside MEEPO's scaffold (meepo3.py)
            from .meepo3 import BiMamba3
            self.mamba = BiMamba3(channels, d_state=mamba_state_dim, d_conv=mamba_conv_dim,
                                  expand=mamba_expand_factor, n_directions=n_directions,
                                  ssm_backend=ssm_backend)
        else:
            self.mamba = BiMamba(channels, d_state=mamba_state_dim, d_conv=mamba_conv_dim,
                                 expand=mamba_expand_factor, n_directions=n_directions,
                                 ssm_backend=ssm_backend)

    def forward(self, point):
        oi = self.order_index % point.serialized_order.shape[0]
        order = point.serialized_order[oi]
        inverse = point.serialized_inverse[oi]
        feat = gather_rows(point.feat, order)                # (N, C), clouds contiguous
        # per-cloud boundaries (offset groups clouds; serialized order keeps them contiguous)
        bounds = torch.cumsum(offset2bincount(point.offset), dim=0).tolist()
        out = torch.empty_like(feat)
        start = 0
        for end in bounds:
            if end > start:
                out[start:end] = self.mamba(feat[start:end].unsqueeze(0)).squeeze(0)
            start = end
        point.feat = gather_rows(out, inverse)
        return point


# ----------------------------------------------------------------------------- #
#  CNN-Mamba block (paper Fig. 6a).
# ----------------------------------------------------------------------------- #
class Block(PointModule):
    def __init__(self, channels, mamba_state_dim, mamba_conv_dim, mamba_expand_factor,
                 mlp_ratio=3.0, proj_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU, pre_norm=True, order_index=0, cpe_indice_key=None,
                 n_directions=2, ssm_backend="auto", mixer="mamba1"):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        # xCPE: 3x3x3 submanifold conv -> norm -> GELU (conditional positional encoding)
        self.cpe = PointSequential(
            SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=cpe_indice_key),
            norm_layer(channels),
            act_layer(),
        )
        self.norm1 = PointSequential(RMSNorm(channels))      # pre-Mamba: RMSNorm (MEEPO)
        self.mixer = SerializedMamba(channels, mamba_state_dim, mamba_conv_dim,
                                     mamba_expand_factor, order_index=order_index,
                                     n_directions=n_directions, ssm_backend=ssm_backend,
                                     mixer=mixer)
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(in_channels=channels, hidden_channels=int(channels * mlp_ratio),
                out_channels=channels, act_layer=act_layer, drop=proj_drop)
        )
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point: Point):
        if self.training and getattr(self, "grad_checkpointing", False):
            if getattr(self, "checkpoint_granularity", "block") == "layer":
                # finest: recompute each heavy sub-module (xCPE conv, Mamba mixer, MLP)
                # in its OWN segment, so only one is ever materialised during backward.
                return self._forward_body(point, ckpt_submodules=True)
            # 'block' (default, unchanged): one checkpoint around the whole block body.
            def _run(feat):
                pc = Point(point)
                pc.feat = feat
                return self._forward_body(pc).feat
            point.feat = checkpoint(_run, point.feat, use_reentrant=False)
            return point
        return self._forward_body(point)

    def _forward_body(self, point: Point, ckpt_submodules: bool = False):
        def run(mod, pt):
            if ckpt_submodules and self.training:
                def _r(feat):
                    pc = Point(pt)
                    pc.feat = feat
                    return mod(pc).feat
                pt.feat = checkpoint(_r, pt.feat, use_reentrant=False)
                return pt
            return mod(pt)
        shortcut = point.feat
        point = run(self.cpe, point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(run(self.mixer, point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(run(self.mlp, point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        return point


class _StageCheckpoint(PointModule):
    """Coarsest granularity: recompute a whole stage's same-resolution Blocks as ONE
    segment. Blocks within a stage share N / coord / offset (pooling is outside the group),
    so threading feat through them under a single checkpoint is exact. Fewer recompute
    boundaries than per-block => less recompute time but a higher activation peak."""
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.grad_checkpointing = True

    def forward(self, point: Point):
        if self.training and self.grad_checkpointing:
            def _run(feat):
                pc = Point(point)
                pc.feat = feat
                for blk in self.blocks:
                    pc = blk._forward_body(pc)
                return pc.feat
            point.feat = checkpoint(_run, point.feat, use_reentrant=False)
            return point
        for blk in self.blocks:
            point = blk._forward_body(point)
        return point


class SerializedPooling(PointModule):
    def __init__(self, in_channels, out_channels, stride=2, norm_layer=None,
                 act_layer=None, reduce="max", shuffle_orders=True, traceable=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointSequential(norm_layer(out_channels)) if norm_layer is not None else None
        self.act = PointSequential(act_layer()) if act_layer is not None else None

    @torch.compiler.disable
    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        code = point.serialized_code >> pooling_depth * 3
        _, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True, return_counts=True)
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        n_seg = counts.numel()

        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1, index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1),
        )
        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code, order, inverse = code[perm], order[perm], inverse[perm]

        pooled = Point(
            feat=_segment_reduce(self.proj(point.feat), cluster, n_seg, self.reduce),
            coord=_segment_reduce(point.coord, cluster, n_seg, "mean"),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )
        if self.traceable:
            pooled["pooling_inverse"] = cluster
            pooled["pooling_parent"] = point
        if self.norm is not None:
            pooled = self.norm(pooled)
        if self.act is not None:
            pooled = self.act(pooled)
        return pooled


class SerializedUnpooling(PointModule):
    def __init__(self, in_channels, skip_channels, out_channels, norm_layer=None,
                 act_layer=None, traceable=False):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))
        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))
        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())
        self.traceable = traceable

    @torch.compiler.disable
    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + gather_rows(point.feat, inverse)
        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None,
                 stem_kernel_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.stem = PointSequential(
            conv=SubMConv3d(in_channels, embed_channels, kernel_size=stem_kernel_size,
                            bias=False, indice_key="stem")
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        return self.stem(point)


# ----------------------------------------------------------------------------- #
#  MEEPO backbone.
# ----------------------------------------------------------------------------- #
class Meepo(PointModule):
    """CNN-Mamba backbone. Returns a :class:`Point` whose ``feat`` is the per-point
    decoder feature (``dec_channels[0]`` channels)."""

    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        mamba_state_dim=1,
        mamba_conv_dim=4,
        mamba_expand_factor=3,
        mlp_ratio=3.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        grad_checkpointing=False,
        checkpoint_granularity="block",
        stem_kernel_size=5,
        n_directions=2,
        ssm_backend="auto",
        mixer="mamba1",              # "mamba1" (MEEPO reference) or "mamba3" (MEEPO-3, meepo3.py)
        norm="ln",          # MEEPO is LayerNorm/RMSNorm based (no BatchNorm) -> micro-batch safe
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(dec_depths) + 1

        norm_layer = partial(nn.LayerNorm, eps=1e-5)
        act_layer = nn.GELU

        self.embedding = Embedding(in_channels=in_channels, embed_channels=enc_channels[0],
                                   norm_layer=norm_layer, act_layer=act_layer,
                                   stem_kernel_size=stem_kernel_size)

        block_kw = dict(mamba_state_dim=mamba_state_dim, mamba_conv_dim=mamba_conv_dim,
                        mamba_expand_factor=mamba_expand_factor, mlp_ratio=mlp_ratio,
                        proj_drop=proj_drop, norm_layer=norm_layer, act_layer=act_layer,
                        pre_norm=pre_norm, n_directions=n_directions, ssm_backend=ssm_backend, mixer=mixer)

        enc_dp = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            dp = enc_dp[sum(enc_depths[:s]):sum(enc_depths[:s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(SerializedPooling(in_channels=enc_channels[s - 1],
                                          out_channels=enc_channels[s], stride=stride[s - 1],
                                          norm_layer=norm_layer, act_layer=act_layer), name="down")
            for i in range(enc_depths[s]):
                enc.add(Block(channels=enc_channels[s], drop_path=dp[i],
                              order_index=i % len(self.order), cpe_indice_key=f"stage{s}",
                              **block_kw), name=f"block{i}")
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        dec_dp = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
        self.dec = PointSequential()
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dp = dec_dp[sum(dec_depths[:s]):sum(dec_depths[:s + 1])]
            dp.reverse()
            dec = PointSequential()
            dec.add(SerializedUnpooling(in_channels=dec_channels[s + 1],
                                        skip_channels=enc_channels[s],
                                        out_channels=dec_channels[s],
                                        norm_layer=norm_layer, act_layer=act_layer), name="up")
            for i in range(dec_depths[s]):
                dec.add(Block(channels=dec_channels[s], drop_path=dp[i],
                              order_index=i % len(self.order), cpe_indice_key=f"stage{s}",
                              **block_kw), name=f"block{i}")
            self.dec.add(module=dec, name=f"dec{s}")
        self.out_channels = dec_channels[0]
        # ---- gradient-checkpoint granularity (memory <-> recompute tradeoff) ----
        # 'none'  : store all activations (fastest, most VRAM)
        # 'stage' : recompute a whole stage's blocks as one segment (coarse)
        # 'block' : recompute each block (default; current behaviour)
        # 'layer' : recompute each block's xCPE / Mamba / MLP separately (finest, least VRAM)
        gran = str(checkpoint_granularity or "block").lower()
        if not bool(grad_checkpointing):
            gran = "none"
        self.checkpoint_granularity = gran
        for _m in self.modules():
            if isinstance(_m, Block):
                _m.grad_checkpointing = gran in ("block", "layer")
                _m.checkpoint_granularity = gran
        if gran == "stage":
            for _seq in list(self.enc._modules.values()) + list(self.dec._modules.values()):
                new = OrderedDict()
                grp = []
                for _name, _mod in _seq._modules.items():
                    if isinstance(_mod, Block):
                        grp.append(_mod)
                    else:
                        if grp:
                            new[f"blocks{len(new)}"] = _StageCheckpoint(grp); grp = []
                        new[_name] = _mod
                if grp:
                    new[f"blocks{len(new)}"] = _StageCheckpoint(grp)
                _seq._modules = new

    def forward(self, data_dict):
        point = data_dict if isinstance(data_dict, Point) else Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        return point
