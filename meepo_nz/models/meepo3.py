"""MEEPO-3: the MEEPO backbone (ICLR paper, Tab. 7/10) with the **Mamba-3**
recurrence (Lahoti et al., arXiv:2603.15569) swapped into its multi-directional
mixer -- selected with ``--backbone meepo3``.

WHAT IS KEPT (every MEEPO feature, byte-identical outside the mixer):
  * The whole backbone: xCPE conv blocks, RMSNorm pre-norms, MLPs, grid
    pooling/unpooling, serialization pool + shuffling, drop-path, stem k5,
    depths/channels per MEEPO Tab. 10, checkpointing hierarchy.
  * Inside the mixer, MEEPO's own micro-fixes, both ablation-backed in-domain:
      - causal-free depthwise conv per direction (their Tab. 7c: +0.5 mIoU).
        NOTE this is a deliberate REJECTION of Mamba-3's conv-deletion finding
        (their Tab. 5a) -- that result is from language modeling and was never
        tested on point clouds, whereas MEEPO's +0.5 was measured in-domain.
      - the bidirectional/strided direction scheme, per-direction parameters,
        gate streams, and sum-then-concat fusion (their Tab. 7d/e).

WHAT CHANGES (the recurrence, per direction):
  * Exponential-trapezoidal discretization (Mamba-3 Prop. 1) via the exact
    two-scan decomposition in ``pointssm3.Mamba3._scan_core`` (fp32 kernel
    profile + contiguity hardening -- the code path already validated on the
    Blackwell inside the pointssm host).
  * Complex state via the RoPE trick (Prop. 4): data-dependent per-pair
    rotations, dt-coupled (channel-mean dt * theta), cumulatively applied to
    B and C. Requires an EVEN d_state >= 2; at d_state=1 (MEEPO's native
    value) the rotation is undefined and is gated OFF -- you still get
    trapezoidal + BCNorm + biases, a graded upgrade.
  * BCNorm (RMS over the state dim) + learnable B/C biases init 1.0 (Mamba-3
    Sec. 3.4 / Tab. 10a), applied after the norm.
  * A is a per-channel SCALAR decay (state-independent; required by the
    two-scan fold; the Mamba-2/3 convention). Init A = 1 for every channel --
    exactly MEEPO's arange(1, N+1) init at its native N=1, extended across
    states, so at initialization the decay matches the proven backbone.
  * dt is precomputed (softplus of the biased Linear) and fed to the scan with
    ``delta_softplus=False`` -- required by the fold, and incidentally cleaner
    than BiMamba's historical double-application of the dt bias (Linear bias +
    ``delta_bias``), which is preserved untouched in BiMamba as load-bearing
    history.

STABILITY INITIALIZATION (lesson from the pointssm lr episode, 2026-07-09):
  * theta_proj init to ZERO -> the model starts as a real SSM.
  * lam_proj bias init to +2.0 -> lambda_0 = sigmoid(2) ~ 0.88, i.e. the model
    starts NEAR the exponential-Euler recurrence (lambda=1) that the proven
    MEEPO configuration uses, and learns its way toward the trapezoid. (The
    Mamba-3 paper's implied zero-init gives lambda_0 = 0.5; we bias toward the
    known-stable dynamics instead. No paper anchors this value; provenance =
    stability engineering, documented here.)
  Net effect: at step 0 the recurrence is approximately the MEEPO-Mamba-1
  dynamics that trained stably at lr 6e-3 on this corpus; the Mamba-3 terms
  grow in gradually. This is the argument for keeping the full MEEPO recipe.

d_state: default 4 (two complex pairs). No paper anchors MEEPO+Mamba-3's N;
PointSSM's ablation (8->76.8, 16->76.9, 32->78.1 on ScanNet) and Mamba-3's
half-state-parity claim bracket the trade. Override with --meepo3-state
(even, or 1 for the RoPE-free graded mode).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba3_fused import fused_available, mamba3_fused, warn_fallback_once
from .pointssm3 import Mamba3
from .ssm import selective_scan  # noqa: F401  (re-export convenience for probes)


class BiMamba3(nn.Module):
    """MEEPO's multi-directional mixer scaffold with a Mamba-3 recurrence.

    Drop-in replacement for :class:`meepo.BiMamba` -- same (B, L, D) contract,
    same in/out projections, same per-direction conv + reorder + gate scheme.
    """

    def __init__(self, d_model, d_state=4, d_conv=4, expand=3, n_directions=2,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, dt_scale=1.0,
                 ssm_backend="auto", lam_bias_init=2.0):
        super().__init__()
        assert d_state == 1 or d_state % 2 == 0, \
            "Mamba-3 complex state needs an even d_state (or 1 for the RoPE-free graded mode)"
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.half = self.d_inner // 2
        self.dt_rank = math.ceil(d_model / 16)
        self.n_directions = n_directions
        self.ssm_backend = ssm_backend
        # heads of headdim P (reference granularity; kernel TMA needs P>=8@bf16).
        # All MEEPO widths give half % 8 == 0; tiny smoke dims fall back to P
        # dividing half, with the kernel gated off unless P == 8.
        # headdim 16: the BACKWARD kernel's dAinv = tl.dot(V, dO^T) contracts over
        # headdim -> tl.dot K-floor 16 (fourth parity-gate catch). All real widths
        # divide by 16; smaller dims run fallback-only.
        self.headdim = 16 if self.half % 16 == 0 else max(
            p for p in (8, 4, 2, 1) if self.half % p == 0)
        self.nheads = self.half // self.headdim
        self._headdim_kernel_ok = (self.headdim >= 16)
        self.use_rope = d_state >= 2
        if d_state >= 8:
            rot_dims = (int(d_state * 0.5) // 4) * 4       # multiple of 4 -> EVEN angles
            self.rope_fraction = 0.5                        # reference default
        else:
            rot_dims = d_state - (d_state % 2)              # all pairs at tiny states
            self.rope_fraction = 1.0
        self.num_rope_angles = max(1, rot_dims // 2) if self.use_rope else 1
        # kernel legality (mamba3_siso_fwd): angle count must be EVEN and <= N//2;
        # tiny states (e.g. d_state=2) rotate on the FALLBACK path only.
        self._rope_kernel_ok = (self.num_rope_angles % 2 == 0
                                and self.num_rope_angles <= d_state // 2)
        self._force_euler = False            # smoke hook: lam=1, theta=0

        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.Ds = nn.ParameterList()
        self.dt_projs = nn.ModuleList()
        self.x_projs = nn.ModuleList()
        self.conv1d_xs = nn.ModuleList()
        self.conv1d_zs = nn.ModuleList()
        # Mamba-3 heads, per direction (mirroring MEEPO's per-direction params)
        self.lam_projs = nn.ModuleList()
        self.theta_projs = nn.ModuleList()
        self.b_norms = nn.ModuleList()
        self.c_norms = nn.ModuleList()
        self.b_biases = nn.ParameterList()
        self.c_biases = nn.ParameterList()
        self.a_projs = nn.ModuleList()       # data-dependent A (reference mamba3.py)

        from .meepo import RMSNorm
        self.dt_biases = nn.ParameterList()
        for _ in range(n_directions):
            # DIRECT PER-HEAD dt projection + separate dt_bias (official mamba3.py
            # scheme: DT = softplus(dd_dt + dt_bias)); per-HEAD granularity is the
            # reference's (dt/lambda/A shared across a head's P channels).
            dt_proj = nn.Linear(self.half, self.nheads, bias=False)
            nn.init.uniform_(dt_proj.weight, -(self.half ** -0.5) * dt_scale,
                             (self.half ** -0.5) * dt_scale)
            self.dt_projs.append(dt_proj)
            dt = torch.exp(
                torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            db = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # softplus^-1
            db._no_reinit = True
            self.dt_biases.append(db)

            self.x_projs.append(nn.Linear(self.half, 2 * d_state, bias=False))

            # NOTE: no static A parameter -- A is DATA-DEPENDENT per the audited
            # reference (heavy-tail head below; zero-init => A_t = -1 at init,
            # which equals MEEPO's arange init at its native N=1).

            D = nn.Parameter(torch.ones(self.nheads))
            D._no_weight_decay = True
            self.Ds.append(D)

            self.conv1d_xs.append(nn.Conv1d(self.half, self.half, kernel_size=d_conv,
                                            groups=self.half, padding="same", bias=False))
            self.conv1d_zs.append(nn.Conv1d(self.half, self.half, kernel_size=d_conv,
                                            groups=self.half, padding="same", bias=False))

            lam = nn.Linear(self.half, self.nheads, bias=True)
            nn.init.constant_(lam.bias, float(lam_bias_init))   # near-Euler start
            self.lam_projs.append(lam)
            th = nn.Linear(self.half, self.num_rope_angles, bias=False)
            nn.init.zeros_(th.weight)                            # real SSM at init
            self.theta_projs.append(th)
            ap = nn.Linear(self.half, self.nheads, bias=False)   # data-dependent A head (per-head)
            nn.init.zeros_(ap.weight)                            # heavy_tail(0)=1 -> A_t=-1
            self.a_projs.append(ap)
            self.b_norms.append(RMSNorm(d_state))
            self.c_norms.append(RMSNorm(d_state))
            b_b = nn.Parameter(torch.ones(d_state)); self.b_biases.append(b_b)
            c_b = nn.Parameter(torch.ones(d_state)); self.c_biases.append(c_b)

    def _reindex(self, t, idx):
        return t[:, :, idx]

    def forward(self, hidden):
        # hidden: (B, L, D)
        B, L, _ = hidden.shape
        xz = self.in_proj(hidden).transpose(1, 2)            # (B, d_inner, L)
        x, z = xz.chunk(2, dim=1)                            # each (B, half, L)

        ys, zs = [], []
        for i in range(self.n_directions):
            xi = F.silu(self.conv1d_xs[i](x))                # causal-free conv KEPT (MEEPO Tab.7c)
            zi = F.silu(self.conv1d_zs[i](z))
            idx = None
            if i == 1:
                xi, zi = xi.flip(-1), zi.flip(-1)
            elif i == 2:
                idx = torch.cat([torch.arange(0, L, 2, device=xi.device),
                                 torch.arange(1, L, 2, device=xi.device)])
                xi, zi = self._reindex(xi, idx), self._reindex(zi, idx)
            elif i == 3:
                idx = torch.cat([torch.arange(0, L, 2, device=xi.device),
                                 torch.arange(1, L, 2, device=xi.device)]).flip(0)
                xi, zi = self._reindex(xi, idx), self._reindex(zi, idx)

            xt = xi.transpose(1, 2)                          # (B, L, half)
            Bp, Cp = torch.split(self.x_projs[i](xt), [self.d_state, self.d_state], dim=-1)
            dt_lin = F.softplus(self.dt_projs[i](xt) + self.dt_biases[i])   # (B, L, H)
            dt = dt_lin.transpose(1, 2).contiguous()         # (B, H, L)  PER-HEAD

            # BCNorm (bias handled per-path: kernel adds internally; fallback below)
            Bp = self.b_norms[i](Bp)
            Cp = self.c_norms[i](Cp)
            lam_raw = self.lam_projs[i](xt).transpose(1, 2)              # (B, H, L)
            theta_raw = self.theta_projs[i](xt)                          # (B, L, n_ang)
            dd = self.a_projs[i](xt).transpose(1, 2)
            neg = dd.clamp_max(0); pos = dd.clamp_min(0)
            a_t = -(pos + torch.reciprocal(1 - neg)).clamp(min=1e-4)     # (B, H, L)

            # ---- FUSED PATH: official Triton kernel (reference semantics:
            # per-head dt*tanh(theta)*pi rotation, sigmoid(trap), native bf16,
            # one pass, headdim-8 heads) ----
            if (not self._force_euler) and xi.is_cuda \
                    and str(self.ssm_backend).lower() == "cuda" \
                    and not (self._rope_kernel_ok and self._headdim_kernel_ok and fused_available()):
                from .mamba3_fused import fused_reason
                raise RuntimeError(
                    f"[mamba3] fused kernel REQUIRED (--ssm-backend cuda) but unusable: "
                    f"rope_ok={self._rope_kernel_ok} headdim_ok={self._headdim_kernel_ok} "
                    f"import={fused_reason() or 'ok'}")
            if (not self._force_euler) and xi.is_cuda and self._rope_kernel_ok \
                    and self._headdim_kernel_ok and fused_available():
                try:
                    yi = mamba3_fused(xi, dt, Bp, Cp, lam_raw, a_t, theta_raw,
                                      self.b_biases[i], self.c_biases[i], self.Ds[i],
                                      headdim=self.headdim)
                    if i == 1:
                        yi, zi = yi.flip(-1), zi.flip(-1)
                    elif i in (2, 3):
                        rev = torch.argsort(idx)
                        yi, zi = self._reindex(yi, rev), self._reindex(zi, rev)
                    ys.append(yi); zs.append(zi)
                    continue
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

            # ---- FALLBACK: exact two-scan, PER HEAD (each head rotates B/C with
            # ITS OWN dt -> reference rotation semantics; the old channel-mean
            # approximation is gone). H small; CPU/no-Triton path only. ----
            Bp = Bp + self.b_biases[i]
            Cp = Cp + self.c_biases[i]
            if self._force_euler:
                lam_h = torch.ones_like(dt)
                a_h = torch.full_like(dt, -1.0)
            else:
                lam_h = torch.sigmoid(lam_raw)
                a_h = a_t
            theta = torch.tanh(theta_raw) * math.pi
            P = self.headdim
            yi = torch.empty_like(xi)
            for h in range(self.nheads):
                Bh, Ch = Bp, Cp
                if self.use_rope and not self._force_euler:
                    ang = torch.cumsum(dt_lin[..., h:h + 1] * theta, dim=1)
                    ang = ang - (2 * math.pi) * torch.floor(ang / (2 * math.pi))
                    cos, sin = torch.cos(ang), torch.sin(ang)
                    n_rot = self.num_rope_angles

                    def _rot(Pv):
                        Pv = Pv.view(B, L, self.d_state // 2, 2)
                        p0, p1 = Pv[..., 0].clone(), Pv[..., 1].clone()
                        # R(+ang), matching their _rotary VERBATIM (rotated_0 =
                        # x0*cos - x1*sin). The old R(-ang) is model-equivalent
                        # under theta -> -theta but NOT function-equivalent, which
                        # broke fallback<->kernel checkpoint portability
                        # ([semantic] rel=0.56, caught by parity gate v3).
                        p0[..., :n_rot] = cos * Pv[..., :n_rot, 0] - sin * Pv[..., :n_rot, 1]
                        p1[..., :n_rot] = sin * Pv[..., :n_rot, 0] + cos * Pv[..., :n_rot, 1]
                        return torch.stack([p0, p1], dim=-1).view(B, L, self.d_state)
                    Bh, Ch = _rot(Bp), _rot(Cp)
                sl = slice(h * P, (h + 1) * P)
                yi[:, sl] = Mamba3._scan_core(
                    xi[:, sl], dt[:, h:h + 1].expand(-1, P, -1).contiguous(),
                    Bh.transpose(1, 2).contiguous(), Ch.transpose(1, 2).contiguous(),
                    lam_h[:, h:h + 1].expand(-1, P, -1).contiguous(),
                    None, D=self.Ds[i][h].expand(P).float(),
                    backend=self.ssm_backend,
                    a_t=a_h[:, h:h + 1].expand(-1, P, -1).contiguous())

            dd = self.a_projs[i](xt).transpose(1, 2)          # (B, half, L)
            neg = dd.clamp_max(0); pos = dd.clamp_min(0)
            a_t = -(pos + torch.reciprocal(1 - neg)).clamp(min=1e-4)  # heavy-tail, <= -1e-4

            if i == 1:
                yi, zi = yi.flip(-1), zi.flip(-1)
            elif i in (2, 3):
                rev = torch.argsort(idx)
                yi, zi = self._reindex(yi, rev), self._reindex(zi, rev)
            ys.append(yi)
            zs.append(zi)

        y = torch.cat([sum(ys), sum(zs)], dim=1)             # (B, d_inner, L)
        return self.out_proj(y.transpose(1, 2))              # (B, L, D)
