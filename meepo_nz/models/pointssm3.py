"""PointSSM backbone (Li et al., IJAEG 144:104830, 2025) with **Mamba-3**
(Lahoti et al., arXiv:2603.15569) as the sequence mixer -- the "point-ssm-mamba"
backbone (``--backbone pointssm``).

HOST (PointSSM, per their paper + Table 1):
  * MambaConv block: LN -> bidirectional SSM (residual) -> POST-conv propagator,
    their Eq. 9:  Y = phi(Conv(Mamba(X))) + Mamba(X).  Their Table 9 shows the
    post-conv carries the long-range-collapse fix (raw 68.5 -> post-conv 78.1).
  * DSamba down-sampling: Linear -> Mamba over the Hilbert sequence -> select the
    LAST point of each pooled grid cell in serialized order (their Fig. 8), so the
    kept token's hidden state summarizes its cell (and, selectively, its past).
  * Displaced-order Hilbert serialization (xyz / yzx / zxy, their Fig. 7,
    Table 13: displaced 78.1 > PTv3 mix 77.2), switched per block.
  * Dims (their Table 1): embedding 64; enc depths [1,1,2,4,1],
    channels [64,64,128,256,512]; dec [1,1,1,1] x [64,64,128,256]; stride 2;
    d_state 32 / expand 1 (MambaConv); d_state 4 / expand 2 (DSamba);
    drop_path 0.1; stem = submanifold conv k5.

MIXER (Mamba-3 SISO, per their Sec. 3 + Fig. 2):
  * Exponential-trapezoidal discretization (their Prop. 1):
        h_t = a_t h_{t-1} + b_t Bb_{t-1} x_{t-1} + g_t Bb_t x_t
    with a_t = exp(dt_t * A), g_t = lam_t * dt_t, b_t = (1-lam_t) * dt_t * a_t,
    lam_t = sigmoid(proj(x_t))  (their default parameterization, Appendix A.3).
  * Complex state via the RoPE trick (their Prop. 4): data-dependent per-pair
    rotation angles, CUMULATIVELY summed and applied to B and C; the recurrence
    then runs as a plain real scan on the rotated Bb, Cb.
  * BCNorm (RMS over the state dim) + learnable B/C biases init to 1.0, added
    AFTER the norm (their Sec. 3.4 / Table 10a).
  * NO internal causal conv (their Table 5a: bias + trapezoidal make it
    redundant) -- propagation is the host's post-conv, per PointSSM's own
    ablation. A is state-independent per channel (scalar decay, the Mamba-2/3
    convention) -- REQUIRED by the two-scan decomposition below.

IMPLEMENTATION -- trapezoidal as TWO standard selective scans (no kernel work):
  the scan is linear in its input stream, so with a state-independent decay
  alpha_t = exp(dt_t * a) the recurrence splits exactly:
      scan1: u1_t = lam_t * x_t                    with B = Bb_t
      scan2: u2_t = (1-lam_t) * alpha_t * x_{t-1}  with B = Bb_{t-1}
  (shift right, zero pad; dt precomputed with softplus OUTSIDE the scan so both
  calls share the identical alpha and the fold into u2 is exact). Both calls go
  through the existing ``selective_scan`` dispatcher, so the fused sm_120 kernel
  and the pure-torch SSD fallback are inherited unchanged. Exactness vs a
  brute-force loop is smoke-checked (smoke [14]).

Bidirectionality: PointSSM's PUBLISHED semantics (their Sec. 3.2 / Fig. 1):
  the flipped sequence is concatenated along the BATCH dim, run through the SAME
  weights in one pass, split, per-direction multiplicatively gated (two
  independent gate projections -- "the other branch of the multiplicative gate
  utilizes two independent weights"), summed, LayerNorm-ed (their Table 12:
  Add + LN best), then linearly projected. Their fused bidirectional kernel is a
  speed optimization of exactly this; we implement the semantics.

Deviations from the PointSSM repo, MINIMIZED (2026-07-09 pass):
  * Norm: cfg.norm='bn' now reproduces their BN&GELU exactly; 'ln' remains the
    micro-batch-1 / grad-accum-safe option (mandatory on the 16 GB path).
  * spconv: NOT a deviation -- SubMConv3d auto-dispatches to real spconv when
    importable (build via setup_blackwell.sh on sm_120) and its pure-torch path
    is numerically the same operator (zero-padded submanifold conv).
  * AUDITED against the official repo (commit f577286, modules/mamba3.py +
    tests/ops/triton/test_mamba3_siso.py step reference), 2026-07-09:
    reference-exact: beta/gamma coefficients + t=0 boundary, norm->bias->rotate
    order, sigmoid(lambda), tanh(angle)*pi bounding, angle-increment = dt*theta
    CUMULATIVE with mod-2pi wrap, rope_fraction=0.5 (half the state rotates),
    data-dependent A via heavy-tail activation (folded exactly into the two-scan).
    Layout residuals (host anatomy, cannot close without multi-head rebuild):
    channel-mean dt in the rotation (theirs per-head), shared B/C biases per
    direction (theirs per-head), dt via the Mamba-1 dt_rank bottleneck (theirs
    direct per-head projection), gate/fusion per the HOST (PointSSM Tab.12 /
    MEEPO), conv KEPT per host evidence (they have none).
  * Bidirectionality: the paper's published batch-concat semantics; their fused
    bidirectional kernel is a speed optimization of the same math.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .mamba3_fused import fused_available, mamba3_fused, warn_fallback_once
from .meepo import (RMSNorm, DropPath, Embedding, SerializedPooling,
                    SerializedUnpooling, _StageCheckpoint, _segment_reduce)
from .point_structure import Point, PointModule, PointSequential, offset2bincount
from .scatter_gather import gather_rows
from .ssm import selective_scan
from .submanifold_conv import SubMConv3d


# --------------------------------------------------------------------------- #
#  Mamba-3 mixer (SISO, trapezoidal + RoPE + BCNorm/biases), (B, L, D) API.
# --------------------------------------------------------------------------- #
class Mamba3(nn.Module):
    def __init__(self, d_model, d_state=32, expand=1, bidirectional=True,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                 ssm_backend="auto"):
        super().__init__()
        assert d_state % 2 == 0, "Mamba-3 complex state needs an even d_state (pairs)"
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.bidirectional = bidirectional
        self.ssm_backend = ssm_backend
        # headdim 16: the BACKWARD kernel's dAinv = tl.dot(V, dO^T) contracts over
        # headdim -> tl.dot K-floor 16 (fourth parity-gate catch). All real widths
        # divide by 16; smaller dims run fallback-only.
        self.headdim = 16 if self.d_inner % 16 == 0 else max(
            p for p in (8, 4, 2, 1) if self.d_inner % p == 0)
        self.nheads = self.d_inner // self.headdim
        self._headdim_kernel_ok = (self.headdim >= 16)
        self.dt_rank = max(1, math.ceil(d_model / 16))
        self._force_euler = False          # smoke hook: lam=1, theta=0 -> exp-Euler

        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        # per-direction multiplicative gates (PointSSM: independent weights per direction)
        self.z_proj_f = nn.Linear(d_model, self.d_inner, bias=False)
        self.z_proj_b = nn.Linear(d_model, self.d_inner, bias=False) if bidirectional else None
        self.fuse_norm = nn.LayerNorm(self.d_inner)          # PointSSM Table 12: Add + LN

        # DIRECT per-head dt projection + separate dt_bias (official mamba3.py
        # scheme; the Mamba-1 dt_rank bottleneck deviation is eliminated)
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.nheads, bias=False)
        nn.init.uniform_(self.dt_proj.weight, -(self.d_inner ** -0.5), (self.d_inner ** -0.5))
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # softplus^-1
        self.dt_bias._no_reinit = True

        # Mamba-3 heads: trapezoidal gate lam (sigmoid; bias 0 -> classical trapezoid
        # lam=1/2 at init), rotation angles theta (init 0 -> real SSM at init).
        self.lam_proj = nn.Linear(self.d_inner, self.nheads, bias=True)
        nn.init.zeros_(self.lam_proj.bias)
        # rope_fraction (reference default 0.5): only the first half of the state
        # pairs rotate; the rest stay real (cos=1/sin=0 padding in the reference).
        if d_state >= 8:
            rot_dims = (int(d_state * 0.5) // 4) * 4       # multiple of 4 -> EVEN angles
            self.rope_fraction = 0.5                        # reference default
        else:
            rot_dims = d_state - (d_state % 2)              # all pairs at tiny states
            self.rope_fraction = 1.0
        self.num_rope_angles = max(1, rot_dims // 2)
        # kernel legality (mamba3_siso_fwd): angle count must be EVEN and <= N//2;
        # tiny states (e.g. d_state=2) rotate on the FALLBACK path only.
        self._rope_kernel_ok = (self.num_rope_angles % 2 == 0
                                and self.num_rope_angles <= d_state // 2)
        self.theta_proj = nn.Linear(self.d_inner, self.num_rope_angles, bias=False)
        nn.init.zeros_(self.theta_proj.weight)
        # data-dependent A head (reference mamba3.py: dd_A -> -heavy_tail, clamped);
        # zero-init => heavy_tail(0)=1 => A_t = -1 at init, matching the static init.
        self.a_proj = nn.Linear(self.d_inner, self.nheads, bias=False)
        nn.init.zeros_(self.a_proj.weight)

        # BCNorm + biases (Mamba-3 Sec 3.4; bias init 1.0 per their Table 10a)
        self.b_norm = RMSNorm(d_state)
        self.c_norm = RMSNorm(d_state)
        self.b_bias = nn.Parameter(torch.ones(d_state))
        self.c_bias = nn.Parameter(torch.ones(d_state))

        # scalar (state-independent) decay per channel; Mamba-2/3 convention.
        A_log = torch.log(torch.empty(self.d_inner).uniform_(1.0, 16.0))
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

    # ---- exact trapezoidal recurrence via two shared-decay scans -------------
    @staticmethod
    def _scan_core(xt, dt, Bb, Cb, lam, a, D=None, backend="auto", a_t=None):
        """xt/dt/lam: (B, d, L); Bb/Cb: (B, N, L); a: (d,) negative. Returns (B, d, L).

        Implements  h_t = alpha_t h_{t-1} + beta_t Bb_{t-1} x_{t-1} + gamma_t Bb_t x_t,
        y_t = Cb_t . h_t (+ D x_t)  with alpha=exp(dt*a), gamma=lam*dt,
        beta=(1-lam)*dt*alpha, by linearity of the scan in its input stream.

        All scan inputs are cast to float32: that is the vanilla mamba_ssm call
        profile the fused kernel unconditionally supports. Under AMP the mixed
        bf16/fp32 tensors this method receives were rejected by the kernel on the
        first Blackwell run, silently rerouting to the d_state-heavy SSD fallback
        (-> 96 GB OOM). fp32 here costs a few MB and removes that failure mode;
        the output is cast back to the caller's dtype.
        """
        out_dtype = xt.dtype
        xt, dt, Bb, Cb, lam = (t.float().contiguous() for t in (xt, dt, Bb, Cb, lam))
        if a_t is not None:
            # Data-dependent A (reference mamba3.py: A_t = -clamp(heavy_tail(dd_A)));
            # folded EXACTLY into the shared-decay two-scan: pass delta_hat = dt*f
            # with A = -1 (decay exp(-dt*f) = exp(dt*A_t)), and divide the input
            # streams by f so delta_hat*B*u reproduces gamma/beta with the RAW dt.
            f = (-a_t).float().clamp(min=1e-4)          # (B, d, L), positive
            dt_hat = (dt * f).contiguous()
            alpha = torch.exp(dt * a_t.float())
            A = torch.full((xt.shape[1], Bb.shape[1]), -1.0, device=xt.device)
            u1 = (lam * xt / f).contiguous()
            u2 = ((1.0 - lam) * alpha * F.pad(xt, (1, 0))[..., :-1] / f).contiguous()
            B2 = F.pad(Bb, (1, 0))[..., :-1].contiguous()
            y = selective_scan(u1, dt_hat, A, Bb, Cb, None, z=None,
                               delta_bias=None, delta_softplus=False, backend=backend)
            y = y + selective_scan(u2, dt_hat, A, B2, Cb, None, z=None,
                                   delta_bias=None, delta_softplus=False, backend=backend)
            if D is not None:
                y = y + xt * D[None, :, None]
            return y.to(out_dtype)
        A = a[:, None].expand(-1, Bb.shape[1]).contiguous()          # (d, N), state-indep
        alpha = torch.exp(dt * a[None, :, None])                     # (B, d, L)
        u1 = (lam * xt).contiguous()
        # shift-right via pad+slice; force fresh contiguous storage -- the fused
        # kernel's vectorized loads assume aligned base pointers, and a sliced
        # view is the kind of input that can trip cudaErrorMisalignedAddress.
        u2 = ((1.0 - lam) * alpha * F.pad(xt, (1, 0))[..., :-1]).contiguous()
        B2 = F.pad(Bb, (1, 0))[..., :-1].contiguous()
        y = selective_scan(u1, dt, A, Bb, Cb, None, z=None,
                           delta_bias=None, delta_softplus=False, backend=backend)
        y = y + selective_scan(u2, dt, A, B2, Cb, None, z=None,
                               delta_bias=None, delta_softplus=False, backend=backend)
        if D is not None:
            y = y + xt * D[None, :, None]
        return y.to(out_dtype)


    def _fallback_scan(self, xin, xt):
        """Per-head exact two-scan (CPU / no-Triton): each head rotates B/C with
        ITS OWN dt (reference semantics; the old channel-mean approximation is
        gone). xin: (B*, L, d_inner); xt: (B*, d_inner, L)."""
        Bn_, L, _ = xin.shape
        Bp, Cp = torch.split(self.x_proj(xin), [self.d_state, self.d_state], dim=-1)
        Bp = self.b_norm(Bp) + self.b_bias
        Cp = self.c_norm(Cp) + self.c_bias
        dt_lin = F.softplus(self.dt_proj(xin) + self.dt_bias)          # (B*, L, H)
        dt = dt_lin.transpose(1, 2).contiguous()
        if self._force_euler:
            lam = torch.ones_like(dt); a_t = torch.full_like(dt, -1.0)
            theta = None
        else:
            lam = torch.sigmoid(self.lam_proj(xin)).transpose(1, 2).contiguous()
            dd = self.a_proj(xin).transpose(1, 2)
            neg = dd.clamp_max(0); pos = dd.clamp_min(0)
            a_t = -(pos + torch.reciprocal(1 - neg)).clamp(min=1e-4)
            theta = torch.tanh(self.theta_proj(xin)) * math.pi
        P = self.headdim
        y = torch.empty_like(xt)
        for h in range(self.nheads):
            Bh, Ch = Bp, Cp
            if theta is not None:
                ang = torch.cumsum(dt_lin[..., h:h + 1] * theta, dim=1)
                ang = ang - (2 * math.pi) * torch.floor(ang / (2 * math.pi))
                cos, sin = torch.cos(ang), torch.sin(ang)
                n_rot = self.num_rope_angles

                def _rot(Pv):
                    Pv = Pv.view(Bn_, L, self.d_state // 2, 2)
                    p0, p1 = Pv[..., 0].clone(), Pv[..., 1].clone()
                    # R(+ang) = their _rotary verbatim (see meepo3.py note)
                    p0[..., :n_rot] = cos * Pv[..., :n_rot, 0] - sin * Pv[..., :n_rot, 1]
                    p1[..., :n_rot] = sin * Pv[..., :n_rot, 0] + cos * Pv[..., :n_rot, 1]
                    return torch.stack([p0, p1], dim=-1).view(Bn_, L, self.d_state)
                Bh, Ch = _rot(Bp), _rot(Cp)
            sl = slice(h * P, (h + 1) * P)
            y[:, sl] = self._scan_core(
                xt[:, sl], dt[:, h:h + 1].expand(-1, P, -1).contiguous(),
                Bh.transpose(1, 2).contiguous(), Ch.transpose(1, 2).contiguous(),
                lam[:, h:h + 1].expand(-1, P, -1).contiguous(), None,
                D=self.D[h].expand(P).float(), backend=self.ssm_backend,
                a_t=a_t[:, h:h + 1].expand(-1, P, -1).contiguous())
        return y

    def forward(self, hidden):
        # hidden: (B, L, D)
        Bsz, L, _ = hidden.shape
        xin = self.in_proj(hidden)                                   # (B, L, d_inner)
        if self.bidirectional:
            xin2 = torch.cat([xin, xin.flip(1)], dim=0)              # batch-concat flip
        else:
            xin2 = xin
        xt = xin2.transpose(1, 2).contiguous()                       # (B*, d, L)
        if (not self._force_euler) and xt.is_cuda and self._rope_kernel_ok \
                and self._headdim_kernel_ok and fused_available():
            Bp, Cp = torch.split(self.x_proj(xin2), [self.d_state, self.d_state], dim=-1)
            dt_lin = F.softplus(self.dt_proj(xin2) + self.dt_bias)
            dtc = dt_lin.transpose(1, 2).contiguous()
            Bn = self.b_norm(Bp); Cn = self.c_norm(Cp)
            lam_raw = self.lam_proj(xin2).transpose(1, 2)
            theta_raw = self.theta_proj(xin2)
            dd = self.a_proj(xin2).transpose(1, 2)
            neg = dd.clamp_max(0); pos = dd.clamp_min(0)
            a_t = -(pos + torch.reciprocal(1 - neg)).clamp(min=1e-4)
            try:
                y = mamba3_fused(xt, dtc, Bn, Cn, lam_raw, a_t, theta_raw,
                                 self.b_bias, self.c_bias, self.D, headdim=self.headdim)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as _e:
                if str(self.ssm_backend).lower() == "cuda":
                    raise RuntimeError(
                        f"[mamba3] fused kernel REQUIRED (--ssm-backend cuda) but failed: "
                        f"{type(_e).__name__}: {str(_e)[:200]}. Falling back would OOM at "
                        f"this batch config -- fix the geometry or rerun with "
                        f"--ssm-backend auto + grad checkpointing.") from _e
                warn_fallback_once(_e)
                y = self._fallback_scan(xin2, xt)
        else:
            y = self._fallback_scan(xin2, xt)
        y = y.transpose(1, 2)                                        # (B*, L, d)
        if self.bidirectional:
            y_f, y_b = y[:Bsz], y[Bsz:].flip(1)
            g_f = F.silu(self.z_proj_f(hidden))
            g_b = F.silu(self.z_proj_b(hidden))
            fused = self.fuse_norm(y_f * g_f + y_b * g_b)            # add + LN (Tab. 12)
        else:
            fused = self.fuse_norm(y * F.silu(self.z_proj_f(hidden)))
        return self.out_proj(fused)


class SerializedMamba3(PointModule):
    """Mamba-3 mixer over the serialized sequence, scanned PER CLOUD (no cross-cloud
    state leakage) -- same contract as SerializedMamba."""

    def __init__(self, channels, d_state, expand, order_index=0, ssm_backend="auto"):
        super().__init__()
        self.order_index = order_index
        self.mamba = Mamba3(channels, d_state=d_state, expand=expand,
                            bidirectional=True, ssm_backend=ssm_backend)

    @torch.compiler.disable
    def forward(self, point):
        oi = self.order_index % point.serialized_order.shape[0]
        order = point.serialized_order[oi]
        inverse = point.serialized_inverse[oi]
        feat = gather_rows(point.feat, order)
        bounds = torch.cumsum(offset2bincount(point.offset), dim=0).tolist()
        out = torch.empty_like(feat)
        start = 0
        for end in bounds:
            if end > start:
                out[start:end] = self.mamba(feat[start:end].unsqueeze(0)).squeeze(0)
            start = end
        point.feat = gather_rows(out, inverse)
        return point


# --------------------------------------------------------------------------- #
#  MambaConv block (PointSSM Fig. 1 / Eq. 9) with Mamba-3 inside.
# --------------------------------------------------------------------------- #
class MC3Block(PointModule):
    def __init__(self, channels, d_state=32, expand=1, drop_path=0.0,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, order_index=0,
                 indice_key=None, ssm_backend="auto"):
        super().__init__()
        self.channels = channels
        self.norm1 = PointSequential(nn.LayerNorm(channels))   # PointSSM Fig. 1: LN before Mamba
        self.mixer = SerializedMamba3(channels, d_state, expand,
                                      order_index=order_index, ssm_backend=ssm_backend)
        # POST-conv propagator (PointSSM Tab. 9: post-conv >> pre-conv >> raw); two
        # submanifold conv blocks; LN/GELU instead of their BN&GELU (micro-batch safe).
        self.prop = PointSequential(
            SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=indice_key),
            norm_layer(channels), act_layer(),
            SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=indice_key),
            norm_layer(channels), act_layer(),
        )
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point: Point):
        if self.training and getattr(self, "grad_checkpointing", False):
            def _run(feat):
                pc = Point(point)
                pc.feat = feat
                return self._forward_body(pc).feat
            point.feat = checkpoint(_run, point.feat, use_reentrant=False)
            return point
        return self._forward_body(point)

    def _forward_body(self, point: Point, ckpt_submodules: bool = False):
        shortcut = point.feat
        point = self.norm1(point)
        point = self.drop_path(self.mixer(point))
        point.feat = shortcut + point.feat            # m = X + Mamba(LN(X))
        shortcut = point.feat
        point = self.drop_path(self.prop(point))
        point.feat = shortcut + point.feat            # Y = m + phi(Conv(Conv(m)))  (Eq. 9)
        return point


# --------------------------------------------------------------------------- #
#  DSamba (PointSSM Fig. 8): Mamba-based down-sampling -- select the LAST token
#  of each pooled grid cell along the serialized (Hilbert) order.
# --------------------------------------------------------------------------- #
class DSamba(SerializedPooling):
    def __init__(self, in_channels, out_channels, stride=2, norm_layer=None,
                 act_layer=None, shuffle_orders=True, d_state=4, expand=2,
                 ssm_backend="auto"):
        super().__init__(in_channels, out_channels, stride=stride, norm_layer=norm_layer,
                         act_layer=act_layer, reduce="max", shuffle_orders=shuffle_orders,
                         traceable=True)
        # Linear d^{l-1} -> d^l happens in self.proj (inherited); the Mamba pass runs
        # AFTER projection, over the parent-resolution sequence, unidirectionally
        # (last-token selection is a causal read-out; d_state 4 / expand 2, Tab. 1 --
        # a SMALL state so the kept token attends to the recent past, their Sec. 3.3.3).
        self.dsamba = Mamba3(out_channels, d_state=d_state, expand=expand,
                             bidirectional=False, ssm_backend=ssm_backend)

    @torch.compiler.disable
    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        code = point.serialized_code >> pooling_depth * 3
        _, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True,
                                          return_counts=True)
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        n_seg = counts.numel()

        # ---- Mamba pass over the PARENT sequence (per cloud, serialized order 0) ----
        feat = self.proj(point.feat)                                  # d^{l-1} -> d^{l}
        order0 = point.serialized_order[0]
        inv0 = point.serialized_inverse[0]
        seq = gather_rows(feat, order0)
        bounds = torch.cumsum(offset2bincount(point.offset), dim=0).tolist()
        out = torch.empty_like(seq)
        start = 0
        for end in bounds:
            if end > start:
                out[start:end] = self.dsamba(seq[start:end].unsqueeze(0)).squeeze(0)
            start = end
        mfeat = gather_rows(out, inv0)                                # back to point order

        # ---- select last-in-serialized-order per cluster (their Fig. 8b) ----
        rank = inv0                                                   # serial position/point
        key = cluster.to(torch.int64) * (rank.numel() + 1) + rank.to(torch.int64)
        perm = torch.argsort(key)
        cl_sorted = cluster[perm]
        is_last = torch.ones_like(cl_sorted, dtype=torch.bool)
        is_last[:-1] = cl_sorted[1:] != cl_sorted[:-1]
        last_idx = perm[is_last]                                      # one per cluster, in
        pooled_feat = mfeat[last_idx]                                 # cluster-id order

        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1, index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1),
        )
        if self.shuffle_orders:
            permo = torch.randperm(code.shape[0])
            code, order, inverse = code[permo], order[permo], inverse[permo]

        pooled = Point(
            feat=pooled_feat,
            coord=_segment_reduce(point.coord, cluster, n_seg, "mean"),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )
        pooled["pooling_inverse"] = cluster
        pooled["pooling_parent"] = point
        if self.norm is not None:
            pooled = self.norm(pooled)
        if self.act is not None:
            pooled = self.act(pooled)
        return pooled


# --------------------------------------------------------------------------- #
#  PointSSM backbone (5-stage encoder / 4-stage decoder, their Table 1).
# --------------------------------------------------------------------------- #
class PointSSM3(PointModule):
    """Returns a :class:`Point` whose ``feat`` is the per-point decoder feature
    (``dec_channels[0]`` channels) -- same contract as :class:`Meepo`."""

    def __init__(
        self,
        in_channels=6,
        order=("hilbert", "hilbert-yzx", "hilbert-zxy"),   # displaced orders (Fig. 7)
        stride=(2, 2, 2, 2),
        enc_depths=(1, 1, 2, 4, 1),
        enc_channels=(64, 64, 128, 256, 512),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(64, 64, 128, 256),
        d_state=32,
        expand=1,
        dsamba_state=4,
        dsamba_expand=2,
        drop_path=0.1,
        shuffle_orders=True,
        grad_checkpointing=False,
        checkpoint_granularity="block",
        stem_kernel_size=5,
        ssm_backend="auto",
        norm="ln",
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(dec_depths) + 1

        # norm='bn' reproduces PointSSM's BN&GELU exactly (BatchNorm1d over the
        # point/active-site axis, the spconv-net convention). norm='ln' (default in
        # the run commands) is the micro-batch-1 / grad-accum-safe substitute --
        # REQUIRED on the 16 GB path; on the big box with --batch-num >= 2, 'bn'
        # matches PointSSM's own per-GPU BN batch (their DALES: bs 4 over 2 GPUs).
        norm_layer = nn.BatchNorm1d if str(norm).lower() == "bn" else partial(nn.LayerNorm, eps=1e-5)
        act_layer = nn.GELU

        self.embedding = Embedding(in_channels=in_channels, embed_channels=enc_channels[0],
                                   norm_layer=norm_layer, act_layer=act_layer,
                                   stem_kernel_size=stem_kernel_size)

        blk = dict(d_state=d_state, expand=expand, norm_layer=norm_layer,
                   act_layer=act_layer, ssm_backend=ssm_backend)

        enc_dp = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            dp = enc_dp[sum(enc_depths[:s]):sum(enc_depths[:s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(DSamba(in_channels=enc_channels[s - 1], out_channels=enc_channels[s],
                               stride=stride[s - 1], norm_layer=norm_layer, act_layer=act_layer,
                               shuffle_orders=shuffle_orders, d_state=dsamba_state,
                               expand=dsamba_expand, ssm_backend=ssm_backend), name="down")
            for i in range(enc_depths[s]):
                enc.add(MC3Block(channels=enc_channels[s], drop_path=dp[i],
                                 order_index=i % len(self.order),
                                 indice_key=f"ps3stage{s}", **blk), name=f"block{i}")
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
                dec.add(MC3Block(channels=dec_channels[s], drop_path=dp[i],
                                 order_index=i % len(self.order),
                                 indice_key=f"ps3stage{s}", **blk), name=f"block{i}")
            self.dec.add(module=dec, name=f"dec{s}")
        self.out_channels = dec_channels[0]

        gran = str(checkpoint_granularity or "block").lower()
        if not bool(grad_checkpointing):
            gran = "none"
        self.checkpoint_granularity = gran
        for _m in self.modules():
            if isinstance(_m, MC3Block):
                _m.grad_checkpointing = gran in ("block", "layer")
        if gran == "stage":
            for _seq in list(self.enc._modules.values()) + list(self.dec._modules.values()):
                new = OrderedDict()
                grp = []
                for _name, _mod in _seq._modules.items():
                    if isinstance(_mod, MC3Block):
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
