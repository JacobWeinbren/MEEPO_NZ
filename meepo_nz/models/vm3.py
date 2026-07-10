"""VM3 — VoxelMamba-3: a group-free, whole-scene Mamba-3 backbone for outdoor
LiDAR segmentation (``--backbone vm3``).

This backbone REPLACES the MEEPO-3 retrofit (meepo3.py). The retrofit kept
MEEPO's Mamba-1 anatomy (per-channel selective scan, dt_rank bottleneck,
d_state 1-4, per-direction convs, channel-mean-dt rotation) and folded the
Mamba-3 math into it, leaving the layout residuals documented in meepo3.py
("cannot close without multi-head rebuild"). VM3 is that rebuild: the mixer is
the OFFICIAL multi-head Mamba-3 SISO block (Lahoti et al., arXiv:2603.15569;
state-spaces/mamba ``modules/mamba3.py``) used at its native anatomy, and the
host is redesigned around what that block wants:

  * GROUP-FREE WHOLE-SCENE SEQUENCES (Voxel Mamba, Zhang et al., NeurIPS 2024,
    arXiv:2406.10700). Every cloud in the batch is ONE serialized sequence --
    no 1024-token patches, no padding. Batching is packed varlen via
    ``cu_seqlens`` (int32, leading 0), which the official Mamba-3 kernels
    support end-to-end (fwd + bwd; see mamba tests/modules/test_mamba3_varlen).
    PTv3 serialization already keeps clouds contiguous (batch id in the code's
    high bits), so ``cu_seqlens`` falls out of ``offset`` for free. This also
    means Mamba-3's cumulative data-dependent RoPE spans the whole scene
    instead of resetting every patch.

  * NATIVE MAMBA-3 ANATOMY. d_state 64 (paper: matches Mamba-2 at N=128 with
    half the state), headdim 64, expand 2, per-head dt/A/trapezoidal-lambda,
    per-head B/C biases after BCNorm, data-dependent RoPE at rope_fraction 0.5,
    heavy-tail data-dependent A, NO conv inside the mixer. Channels are sized
    so every stage forms >= 4 heads of dim 64 (all stage widths divisible by
    32 with expand 2).

  * DUAL-SCALE SSM BLOCK (DSB; Voxel Mamba Eq. 4). Each mixer op runs a
    forward Mamba-3 branch on the full-resolution sequence and a BACKWARD
    branch on a code-downsampled sequence (stride 2^k per stage), flipped,
    scanned, unflipped and broadcast back to the parents. Bidirectionality +
    hierarchy + a larger effective receptive field at roughly half the cost of
    naive bidirectional scanning.

  * IMPLICIT WINDOW EMBEDDING (IWP/IWE; Voxel Mamba Sec. 3.5). A small MLP of
    (z, window index, in-window offsets, + a half-window-shifted copy) added
    to the branch inputs, shared per stage. Complementary to RoPE: IWE encodes
    absolute 3D position, the data-dependent RoPE encodes dynamics along the
    traversal. (+0.8 mAP in Voxel Mamba Tab. 6b/6d.)

  * SPATIAL LOCALITY OUTSIDE THE MIXER (UniMamba, Jin et al., CVPR 2025,
    arXiv:2503.12009, "SLM"). The stem is a k5 submanifold conv and each block
    keeps an optional xCPE (k3 subconv) residual BEFORE the mixer. UniMamba
    Tab. 4: with conv-provided locality even random ordering is within 0.2 mAP
    of Hilbert. This reconciles MEEPO's +0.5 mIoU conv evidence with Mamba-3's
    conv deletion: the deleted conv is the *in-recurrence causal* conv; the
    spatial conv lives in the host. Disable per ablation with use_cpe=False.

  * LOCAL-GLOBAL VIA PER-HEAD DECAY BANDING (replaces UniMamba's LGSA
    modules). Mamba-3's A and dt are per-head and data-dependent; we
    initialize dt_bias in deterministic log-spaced bands across
    [dt_min, dt_max] so half the heads start fast-decay (local receptive
    field) and half slow (global), and let the data-dependent per-head A
    learn the split that UniMamba hard-wires as separate LSE/GSE channel
    groups (their Tab. 5: the local path is worth ~+0.7 mAP over GSE-only).

  * MIMO OFF. Segmentation is one parallel pass (prefill-like); Mamba-3 Tab. 7
    shows MIMO costs ~20% prefill for a decode-side benefit we never use.

Backend selection (``ssm_backend``):
  * 'cuda'          : REQUIRE the official mamba_ssm Mamba3 (Triton kernels).
  * 'auto' / 'ssd'  : official if importable AND CUDA is available, else the
                      pure-torch reference below (CPU smoke tests).
  * 'torch'         : force the pure-torch reference (exact math, slow).

The pure-torch reference (`Mamba3TorchRef`) mirrors the official parameter
layout (same state_dict keys for the SISO / is_outproj_norm=True
configuration) and the official semantics: heavy-tail A, softplus(dt+bias),
sigmoid trapezoidal lambda, BCNorm -> per-head bias -> cumulative
(dt x angle, mod 2pi) rotation of the first rope_fraction of the state,
3-term exponential-trapezoidal recurrence (Prop. 1/4), D skip, grouped
(per-head) gated RMS output norm. It exists so the whole stack smoke-tests on
CPU before GPU time is spent; training runs use the official Triton kernels.
"""
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import DropPath
from .meepo import (Embedding, MLP, RMSNorm, SerializedPooling,
                    SerializedUnpooling, _segment_reduce)
from .point_structure import (Point, PointModule, PointSequential,
                              offset2bincount)
from .scatter_gather import gather_rows
from .submanifold_conv import SubMConv3d


# --------------------------------------------------------------------------- #
#  Official Mamba-3 import (lazy, guarded).
# --------------------------------------------------------------------------- #
_OFFICIAL_MAMBA3 = None
_OFFICIAL_ERR = None


def _official_mamba3():
    """Import the official Mamba-3 module once; cache the class or the error."""
    global _OFFICIAL_MAMBA3, _OFFICIAL_ERR
    if _OFFICIAL_MAMBA3 is not None or _OFFICIAL_ERR is not None:
        return _OFFICIAL_MAMBA3
    try:
        from mamba_ssm.modules.mamba3 import Mamba3 as _M3  # noqa: WPS433
        _OFFICIAL_MAMBA3 = _M3
    except Exception as e:  # ImportError, triton absent, etc.
        _OFFICIAL_ERR = e
        _OFFICIAL_MAMBA3 = None
    return _OFFICIAL_MAMBA3


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """Official Mamba-3 data-dependent-A activation: f(x)=1+x (x>=0), 1/(1-x) (x<0)."""
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


def _inv_softplus(dt: torch.Tensor) -> torch.Tensor:
    """b with softplus(b)=dt, official parameterization: dt + log(-expm1(-dt))."""
    return dt + torch.log(-torch.expm1(-dt))


def apply_decay_bands(mixer: nn.Module, dt_min: float, dt_max: float) -> None:
    """Deterministic per-head dt bands: heads span [dt_min, dt_max] log-spaced.

    Low-dt heads decay slowly (long/global horizon), high-dt heads decay fast
    (local horizon) -- the learnable, Mamba-3-native version of UniMamba's
    hard LSE/GSE channel split. Overwrites the official random log-uniform
    init IN PLACE (both inits target the same softplus parameterization).
    """
    dt_bias = getattr(mixer, "dt_bias", None)
    if dt_bias is None:
        return
    h = dt_bias.numel()
    with torch.no_grad():
        dt = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), h,
                                      dtype=torch.float32, device=dt_bias.device))
        dt = dt.clamp(min=1e-4)
        dt_bias.copy_(_inv_softplus(dt).to(dt_bias.dtype))


# --------------------------------------------------------------------------- #
#  Pure-torch Mamba-3 reference (SISO, is_outproj_norm=True layout).
# --------------------------------------------------------------------------- #
class _RefRMSNorm(nn.Module):
    """RMSNorm over the last dim (state_dict-compatible with RMSNormGated: 'weight')."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class _RefGroupedGatedNorm(nn.Module):
    """Grouped (per-head) RMSNorm, norm-before-gate: RMS_head(y)*w * silu(z).

    Mirrors RMSNormGated(d_inner, group_size=headdim, norm_before_gate=True);
    state_dict key: 'weight' of shape (d_inner,).
    """

    def __init__(self, d_inner, group_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.ones(d_inner))

    def forward(self, y, z):
        dt = y.dtype
        shp = y.shape
        yg = y.float().reshape(*shp[:-1], -1, self.group_size)
        yg = yg * torch.rsqrt(yg.pow(2).mean(-1, keepdim=True) + self.eps)
        y = yg.reshape(shp) * self.weight.float()
        return (y * F.silu(z.float())).to(dt)


class Mamba3TorchRef(nn.Module):
    """Pure-torch Mamba-3 SISO reference. Same parameter layout / keys as the
    official ``mamba_ssm.modules.mamba3.Mamba3`` with ``is_mimo=False,
    ngroups=1, is_outproj_norm=True``. Sequential scan; CPU-safe; for smoke
    tests and as a last-resort fallback -- NOT for GPU training speed.
    """

    def __init__(self, d_model, d_state=64, expand=2, headdim=64,
                 rope_fraction=0.5, dt_min=0.001, dt_max=0.1,
                 dt_init_floor=1e-4, A_floor=1e-4, chunk_size=64,
                 device=None, dtype=None, **_):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.A_floor = A_floor
        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = 1

        assert rope_fraction in (0.5, 1.0)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0, "d_state too small for RoPE pairs"

        d_in_proj = (2 * self.d_inner + 2 * self.d_state
                     + 3 * self.nheads + self.num_rope_angles)
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        _dt = torch.exp(torch.rand(self.nheads, dtype=torch.float32)
                        * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        _dt = torch.clamp(_dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(_inv_softplus(_dt))
        self.dt_bias._no_weight_decay = True

        self.B_bias = nn.Parameter(1 + torch.zeros(self.nheads, 1, self.d_state))
        self.C_bias = nn.Parameter(1 + torch.zeros(self.nheads, 1, self.d_state))
        self.B_norm = _RefRMSNorm(self.d_state)
        self.C_norm = _RefRMSNorm(self.d_state)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True
        self.norm = _RefGroupedGatedNorm(self.d_inner, group_size=headdim)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    # -- one packed varlen forward ---------------------------------------- #
    def forward(self, u, cu_seqlens=None, seq_idx=None, inference_params=None):
        assert u.dim() == 3 and u.shape[0] == 1, "reference expects packed (1, L, D)"
        L = u.shape[1]
        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, L], dtype=torch.int32, device=u.device)

        zxBCdtAtrap = self.in_proj(u[0])  # (L, d_in_proj)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [self.d_inner, self.d_inner, self.d_state, self.d_state,
             self.nheads, self.nheads, self.nheads, self.num_rope_angles],
            dim=-1)
        H, P, N = self.nheads, self.headdim, self.d_state
        x = x.view(L, H, P).float()
        z = z.view(L, H, P)

        A = -heavy_tail_activation(dd_A.float()).clamp(max=-self.A_floor)   # (L, H)
        DT = F.softplus(dd_dt.float() + self.dt_bias.float())               # (L, H)
        alpha = torch.exp(A * DT)                                           # (L, H)
        lam = torch.sigmoid(trap.float())                                   # (L, H)
        gamma = lam * DT                                                    # (L, H)
        beta = (1.0 - lam) * DT * alpha                                     # (L, H)

        Bn = self.B_norm(B.float()) + 0.0                                   # (L, N)
        Cn = self.C_norm(C.float())
        # per-head bias after norm  ->  (L, H, N)
        Bh = Bn.unsqueeze(1) + self.B_bias.float().squeeze(1).unsqueeze(0)
        Ch = Cn.unsqueeze(1) + self.C_bias.float().squeeze(1).unsqueeze(0)

        # cumulative data-dependent rotation: phi_t = sum_{i<=t} DT_i * angle_i
        S2 = self.num_rope_angles
        ang = angles.float().unsqueeze(1) * DT.unsqueeze(-1)                # (L, H, S2)
        y = torch.empty(L, H, P, dtype=torch.float32, device=u.device)

        cs = cu_seqlens.tolist()
        for s, e in zip(cs[:-1], cs[1:]):
            if e <= s:
                continue
            phi = torch.cumsum(ang[s:e], dim=0)
            phi = torch.remainder(phi, 2 * math.pi)
            cosp, sinp = torch.cos(phi), torch.sin(phi)                     # (l, H, S2)

            def _rot(v):
                rot, rest = v[..., :self.split_tensor_size], v[..., self.split_tensor_size:]
                a = rot[..., 0::2]
                b = rot[..., 1::2]
                ra = a * cosp - b * sinp
                rb = a * sinp + b * cosp
                out = torch.empty_like(rot)
                out[..., 0::2] = ra
                out[..., 1::2] = rb
                return torch.cat([out, rest], dim=-1)

            Bt = _rot(Bh[s:e])                                              # (l, H, N)
            Ct = _rot(Ch[s:e])
            h = torch.zeros(H, N, P, dtype=torch.float32, device=u.device)
            u_prev = torch.zeros(H, N, P, dtype=torch.float32, device=u.device)
            for t in range(e - s):
                u_t = torch.einsum("hn,hp->hnp", Bt[t], x[s + t])
                h = (alpha[s + t].view(H, 1, 1) * h
                     + beta[s + t].view(H, 1, 1) * u_prev
                     + gamma[s + t].view(H, 1, 1) * u_t)
                u_prev = u_t
                y[s + t] = torch.einsum("hn,hnp->hp", Ct[t], h)

        y = y + self.D.float().view(1, H, 1) * x
        y = y.reshape(L, self.d_inner)
        y = self.norm(y, z.reshape(L, self.d_inner))
        return self.out_proj(y.to(u.dtype)).unsqueeze(0)


# --------------------------------------------------------------------------- #
#  Packed-varlen mixer wrapper: (N, C) + cu_seqlens -> (N, C).
# --------------------------------------------------------------------------- #
class PackedMamba3(nn.Module):
    """One Mamba-3 SISO mixer over a packed varlen sequence.

    Dispatch:
      backend='cuda'        -> official mamba_ssm Mamba3 (raise if unavailable)
      backend='torch'       -> pure-torch reference
      backend='auto'/'ssd'  -> official if importable and CUDA present, else ref
    """

    def __init__(self, d_model, d_state=64, headdim=64, expand=2,
                 chunk_size=64, backend="auto", dt_min=0.001, dt_max=0.1,
                 decay_bands=True, rope_fraction=0.5):
        super().__init__()
        backend = str(backend or "auto").lower()
        official = _official_mamba3()
        use_official = False
        if backend == "cuda":
            if official is None:
                raise ImportError(
                    "ssm_backend='cuda' requires the official mamba package "
                    "(pip install -e mamba-main --no-deps; needs triton+einops+transformers). "
                    f"Import error: {_OFFICIAL_ERR!r}")
            use_official = True
        elif backend in ("auto", "ssd"):
            use_official = official is not None and torch.cuda.is_available()
        # 'torch' -> ref

        kw = dict(d_model=d_model, d_state=d_state, expand=expand,
                  headdim=headdim, rope_fraction=rope_fraction,
                  dt_min=dt_min, dt_max=dt_max, chunk_size=chunk_size)
        if use_official:
            self.mixer = official(ngroups=1, is_mimo=False,
                                  is_outproj_norm=True, **kw)
            self.impl = "mamba3-official"
        else:
            self.mixer = Mamba3TorchRef(**kw)
            self.impl = "mamba3-torch-ref"
        if decay_bands:
            apply_decay_bands(self.mixer, dt_min, dt_max)

    def forward(self, feat, cu_seqlens):
        # feat: (N, C) packed, clouds contiguous; cu_seqlens: (B+1,) int32.
        out = self.mixer(feat.unsqueeze(0).contiguous(),
                         cu_seqlens=cu_seqlens.to(torch.int32))
        return out.squeeze(0)


# --------------------------------------------------------------------------- #
#  Implicit Window Embedding (Voxel Mamba Sec. 3.5).
# --------------------------------------------------------------------------- #
class IWE(nn.Module):
    """MLP(concat(z, x//w, y//h, x%w, y%h)  for shift 0 and w//2) -> (N, C).

    Encodes 3D position inside AND across implicit windows without explicit
    partition. Shared per stage. Inputs are float-normalized for stability.
    """

    def __init__(self, channels, window=16, mid=32):
        super().__init__()
        self.window = int(window)
        self.mlp = nn.Sequential(
            nn.Linear(10, mid), nn.GELU(), nn.Linear(mid, channels))

    def forward(self, grid_coord):
        w = self.window
        g = grid_coord.to(torch.float32)
        feats = []
        for shift in (0, w // 2):
            xs = g[:, 0] + shift
            ys = g[:, 1] + shift
            feats += [g[:, 2] / 32.0,
                      torch.floor(xs / w) / 64.0,
                      torch.floor(ys / w) / 64.0,
                      torch.remainder(xs, w) / w,
                      torch.remainder(ys, w) / w]
        return self.mlp(torch.stack(feats, dim=-1))


# --------------------------------------------------------------------------- #
#  Sequence helpers (packed varlen).
# --------------------------------------------------------------------------- #
def _cu_from_batch(batch, num_batches=None):
    counts = torch.bincount(batch, minlength=int(num_batches or 0))
    cu = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
    return cu.to(torch.int32)


def _flip_index(cu_seqlens, device):
    """Index that reverses each sequence of a packed layout (involution)."""
    cu = cu_seqlens.to(torch.long)
    counts = torch.diff(cu)
    n = int(cu[-1].item())
    seq_id = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device), counts.to(device))
    starts = cu[:-1].to(device)[seq_id]
    ends = cu[1:].to(device)[seq_id]
    idx = torch.arange(n, device=device)
    return starts + ends - 1 - idx


# --------------------------------------------------------------------------- #
#  Dual-scale Mamba-3 op on a Point (Voxel Mamba Eq. 4, DSB).
# --------------------------------------------------------------------------- #
class VM3DSB(PointModule):
    """Fe_delta = LN(FwdSSM(seq(F + IWE))) + Up(LN(BwdSSM(flip(Down(F) + IWE'))))

    The enclosing VM3Block supplies the ``+ F`` residual. ``down`` is the
    backward-branch stride 2^k over the serialized code (k=0 -> plain flipped
    full-resolution backward scan). Both branches are Mamba-3 SISO mixers with
    independent parameters, run as packed varlen over ALL clouds at once.
    """

    def __init__(self, channels, d_state=64, headdim=64, expand=2,
                 order_index=0, down=1, iwe: IWE | None = None,
                 chunk_size=64, backend="auto", dt_min=0.001, dt_max=0.1,
                 decay_bands=True):
        super().__init__()
        assert down >= 1 and (down & (down - 1)) == 0, "down must be a power of 2"
        self.order_index = order_index
        self.down = int(down)
        self.iwe = iwe
        mk = dict(d_state=d_state, headdim=headdim, expand=expand,
                  chunk_size=chunk_size, backend=backend,
                  dt_min=dt_min, dt_max=dt_max, decay_bands=decay_bands)
        self.fwd_ssm = PackedMamba3(channels, **mk)
        self.bwd_ssm = PackedMamba3(channels, **mk)
        self.ln_f = nn.LayerNorm(channels)
        self.ln_b = nn.LayerNorm(channels)

    @torch.compiler.disable
    def forward(self, point: Point):
        oi = self.order_index % point.serialized_order.shape[0]
        order = point.serialized_order[oi]
        inverse = point.serialized_inverse[oi]
        feat = point.feat
        dev = feat.device
        cu = _cu_from_batch(point.batch, num_batches=int(point.offset.numel()))

        # ---- forward branch: full-resolution whole-scene sequence -------- #
        pe = self.iwe(point.grid_coord) if self.iwe is not None else None
        x_f = feat + pe if pe is not None else feat
        xs = gather_rows(x_f, order)
        yf = self.fwd_ssm(xs, cu)
        Ff = gather_rows(self.ln_f(yf), inverse)

        # ---- backward branch: downsampled (or full-res) flipped scan ----- #
        k = int(math.log2(self.down))
        if k == 0 or (3 * k) >= int(point.serialized_depth) * 3:
            xb = gather_rows(x_f, order)
            rev = _flip_index(cu, dev)
            yb = self.bwd_ssm(xb[rev], cu)[rev]
            Fb = gather_rows(self.ln_b(yb), inverse)
        else:
            code = point.serialized_code[oi] >> (3 * k)
            _, cluster, counts = torch.unique(
                code, sorted=True, return_inverse=True, return_counts=True)
            n_seg = counts.numel()
            _, indices = torch.sort(cluster)
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
            head_indices = indices[idx_ptr[:-1]]
            # pooled sequence is already in serialized (ascending-code) order,
            # clouds contiguous (batch id lives in the code's high bits).
            pooled = _segment_reduce(feat, cluster, n_seg, "mean")
            if self.iwe is not None:
                pooled = pooled + self.iwe(point.grid_coord[head_indices] >> k)
            cu_p = _cu_from_batch(point.batch[head_indices],
                                  num_batches=int(point.offset.numel()))
            rev = _flip_index(cu_p, dev)
            yb = self.bwd_ssm(pooled[rev], cu_p)[rev]
            Fb = gather_rows(self.ln_b(yb), cluster)  # Up: broadcast to parents

        point.feat = Ff + Fb
        return point


# --------------------------------------------------------------------------- #
#  SwiGLU MLP (Mamba-3 paper block alternation).
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    def __init__(self, channels, ratio=3.0, drop=0.0):
        super().__init__()
        hidden = int(channels * ratio)
        self.w12 = nn.Linear(channels, 2 * hidden)
        self.w3 = nn.Linear(hidden, channels)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.w3(F.silu(a) * b))


# --------------------------------------------------------------------------- #
#  VM3 block: [xCPE] -> RMSNorm -> DSB(Mamba-3 fwd/bwd) -> +res -> LN -> SwiGLU -> +res
# --------------------------------------------------------------------------- #
class VM3Block(PointModule):
    def __init__(self, channels, d_state=64, headdim=64, expand=2,
                 mlp_ratio=3.0, proj_drop=0.0, drop_path=0.0,
                 order_index=0, down=1, iwe: IWE | None = None,
                 use_cpe=True, cpe_indice_key=None, chunk_size=64,
                 backend="auto", dt_min=0.001, dt_max=0.1, decay_bands=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-5), act_layer=nn.GELU):
        super().__init__()
        self.channels = channels
        self.use_cpe = bool(use_cpe)
        if self.use_cpe:
            # xCPE: spatial locality OUTSIDE the mixer (UniMamba SLM / MEEPO xCPE).
            self.cpe = PointSequential(
                SubMConv3d(channels, channels, kernel_size=3, bias=True,
                           indice_key=cpe_indice_key),
                norm_layer(channels),
                act_layer(),
            )
        self.norm1 = PointSequential(RMSNorm(channels))
        self.mixer = VM3DSB(channels, d_state=d_state, headdim=headdim,
                            expand=expand, order_index=order_index, down=down,
                            iwe=iwe, chunk_size=chunk_size, backend=backend,
                            dt_min=dt_min, dt_max=dt_max, decay_bands=decay_bands)
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(SwiGLU(channels, ratio=mlp_ratio, drop=proj_drop))
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity())
        self.grad_checkpointing = False

    def forward(self, point: Point):
        if self.training and getattr(self, "grad_checkpointing", False):
            def _run(feat):
                pc = Point(point)
                pc.feat = feat
                return self._forward_body(pc).feat
            point.feat = checkpoint(_run, point.feat, use_reentrant=False)
            return point
        return self._forward_body(point)

    def _forward_body(self, point: Point):
        if self.use_cpe:
            shortcut = point.feat
            point = self.cpe(point)
            point.feat = shortcut + point.feat
        shortcut = point.feat
        point = self.norm1(point)
        point = self.drop_path(self.mixer(point))
        point.feat = shortcut + point.feat

        shortcut = point.feat
        point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        return point


# --------------------------------------------------------------------------- #
#  The VoxelMamba-3 backbone.
# --------------------------------------------------------------------------- #
class VoxelMamba3(PointModule):
    """Group-free whole-scene Mamba-3 U-Net over PTv3 serialization.

    Returns a :class:`Point` whose ``feat`` has ``dec_channels[0]`` channels
    (per input point). Every stage width must satisfy (expand*C) % headdim == 0.
    """

    def __init__(
        self,
        in_channels=6,
        order=("hilbert", "hilbert-trans", "z", "z-trans"),
        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 2),
        enc_channels=(128, 256, 384, 512),
        dec_depths=(1, 1, 1),
        dec_channels=(128, 256, 384),
        d_state=64,
        headdim=64,
        expand=2,
        mlp_ratio=3.0,
        proj_drop=0.0,
        drop_path=0.3,
        dsb_down=(1, 2, 4, 4),
        iwe_window=16,
        use_cpe=True,
        decay_bands=True,
        chunk_size=64,
        dt_min=0.001,
        dt_max=0.1,
        shuffle_orders=True,
        stem_kernel_size=5,
        ssm_backend="auto",
        grad_checkpointing=False,
        checkpoint_granularity="block",
        norm="ln",  # accepted for host-API parity; VM3 is LN/RMSNorm-only
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_depths) + 1
        assert len(dec_channels) == self.num_stages - 1
        for c in list(enc_channels) + list(dec_channels):
            assert (int(expand) * int(c)) % int(headdim) == 0, (
                f"stage width {c}: expand*C={expand * c} must be divisible by headdim={headdim}")

        norm_layer = partial(nn.LayerNorm, eps=1e-5)
        act_layer = nn.GELU
        dsb_down = tuple(dsb_down) + (dsb_down[-1],) * max(0, self.num_stages - len(dsb_down))

        self.embedding = Embedding(in_channels=in_channels,
                                   embed_channels=enc_channels[0],
                                   norm_layer=norm_layer, act_layer=act_layer,
                                   stem_kernel_size=stem_kernel_size)

        # One IWE per stage RESOLUTION, keyed by channel width per level; the
        # decoder stage s reuses the encoder-stage-s embedding resolution but
        # has its own channel width, so it owns its own IWE at that width.
        self.enc_iwe = nn.ModuleList(
            [IWE(enc_channels[s], window=iwe_window) for s in range(self.num_stages)])
        self.dec_iwe = nn.ModuleList(
            [IWE(dec_channels[s], window=iwe_window) for s in range(self.num_stages - 1)])

        blk_kw = dict(d_state=d_state, headdim=headdim, expand=expand,
                      mlp_ratio=mlp_ratio, proj_drop=proj_drop,
                      use_cpe=use_cpe, chunk_size=chunk_size,
                      backend=ssm_backend, dt_min=dt_min, dt_max=dt_max,
                      decay_bands=decay_bands, norm_layer=norm_layer,
                      act_layer=act_layer)

        enc_dp = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            dp = enc_dp[sum(enc_depths[:s]):sum(enc_depths[:s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(SerializedPooling(in_channels=enc_channels[s - 1],
                                          out_channels=enc_channels[s],
                                          stride=stride[s - 1],
                                          norm_layer=norm_layer,
                                          act_layer=act_layer,
                                          shuffle_orders=shuffle_orders), name="down")
            for i in range(enc_depths[s]):
                enc.add(VM3Block(channels=enc_channels[s], drop_path=dp[i],
                                 order_index=i % len(self.order),
                                 down=dsb_down[s], iwe=self.enc_iwe[s],
                                 cpe_indice_key=f"vm3stage{s}", **blk_kw),
                        name=f"block{i}")
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        dec_dp = [x.item() for x in torch.linspace(0, drop_path, max(1, sum(dec_depths)))]
        self.dec = PointSequential()
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dp = dec_dp[sum(dec_depths[:s]):sum(dec_depths[:s + 1])]
            dp.reverse()
            dec = PointSequential()
            dec.add(SerializedUnpooling(in_channels=dec_channels[s + 1],
                                        skip_channels=enc_channels[s],
                                        out_channels=dec_channels[s],
                                        norm_layer=norm_layer,
                                        act_layer=act_layer), name="up")
            for i in range(dec_depths[s]):
                dec.add(VM3Block(channels=dec_channels[s], drop_path=dp[i],
                                 order_index=i % len(self.order),
                                 down=dsb_down[s], iwe=self.dec_iwe[s],
                                 cpe_indice_key=f"vm3stage{s}", **blk_kw),
                        name=f"block{i}")
            self.dec.add(module=dec, name=f"dec{s}")
        self.out_channels = dec_channels[0]

        gran = str(checkpoint_granularity or "block").lower()
        if not bool(grad_checkpointing):
            gran = "none"
        elif gran in ("stage", "layer"):
            gran = "block"  # VM3 supports block granularity; coarser/finer map to it
        self.checkpoint_granularity = gran
        for m in self.modules():
            if isinstance(m, VM3Block):
                m.grad_checkpointing = gran == "block"

    # -- introspection ----------------------------------------------------- #
    def mixer_impl(self) -> str:
        for m in self.modules():
            if isinstance(m, PackedMamba3):
                return m.impl
        return "none"

    def forward(self, data_dict):
        point = data_dict if isinstance(data_dict, Point) else Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        return point
