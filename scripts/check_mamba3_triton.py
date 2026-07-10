#!/usr/bin/env python3
"""GPU parity gate v3. Ground truth = the AUTHORS' OWN reference
(mamba3_siso_fwd_ref, vendored verbatim). Two attributed comparisons:
  [semantic] our two-scan fallback vs their reference -- asserted TIGHT (<=1e-3):
             proves our reading of the Mamba-3 semantics.
  [kernel]   official Triton kernel vs their reference -- asserted at 0.15,
             i.e. THEIR OWN test-policy class (rtol=1e-1, forward assert
             commented out, documented 6-8% output / ~20% angle error from
             fast-math approximations + bf16 internals + cumsum drift).
             The number is REPORTED so you can judge it; a training decision
             at ~5-10% kernel approximation is a documented, revocable choice
             (exact-but-slow alternative: --ssm-backend auto + grad ckpt).
Usage: PYTHONPATH=. python scripts/check_mamba3_triton.py
"""
import math
import sys

import torch
import torch.nn.functional as F

from meepo_nz.models.mamba3_fused import fused_available, fused_reason, mamba3_fused
from meepo_nz.models.pointssm3 import Mamba3
from meepo_nz.ops.triton_mamba3.reference import mamba3_siso_fwd_ref


def loop_ref(xt, dt, Bn, Cn, lam_raw, a_t, theta_raw, b_bias, c_bias, D, n_rot, P):
    Bsz, C, L = xt.shape
    H = C // P
    N = Bn.shape[-1]
    lam = torch.sigmoid(lam_raw)
    theta = torch.tanh(theta_raw) * math.pi                        # (B, L, n_ang)
    alpha = torch.exp(dt * a_t)
    q = Cn + c_bias
    k = Bn + b_bias
    v_all = xt.transpose(1, 2).reshape(Bsz, L, H, P)               # (B, L, H, P)
    ang_state = torch.zeros(Bsz, H, theta.shape[-1], device=xt.device, dtype=torch.float64)
    S = torch.zeros(Bsz, H, N, P, dtype=torch.float64, device=xt.device)
    k_prev = torch.zeros(Bsz, H, N, dtype=torch.float64, device=xt.device)
    v_prev = torch.zeros(Bsz, H, P, dtype=torch.float64, device=xt.device)
    y = torch.zeros(Bsz, L, H, P, dtype=torch.float64, device=xt.device)
    for t in range(L):
        ang_state = torch.remainder(
            ang_state + theta[:, t].double().unsqueeze(1) * dt[:, :, t].double().unsqueeze(-1),
            2 * math.pi)
        cos, sin = torch.cos(ang_state), torch.sin(ang_state)

        def rot(v):  # v: (B, N) -> per-head (B, H, N); rotate first n_rot pairs
            vh = v.double().unsqueeze(1).expand(-1, H, -1).clone()
            P = vh.view(Bsz, H, N // 2, 2)
            p0, p1 = P[..., 0].clone(), P[..., 1].clone()
            P0 = cos[..., :n_rot] * p0[..., :n_rot] + sin[..., :n_rot] * p1[..., :n_rot]
            P1 = -sin[..., :n_rot] * p0[..., :n_rot] + cos[..., :n_rot] * p1[..., :n_rot]
            p0[..., :n_rot], p1[..., :n_rot] = P0, P1
            return torch.stack([p0, p1], dim=-1).view(Bsz, H, N)

        kbar = rot(k[:, t]); qbar = rot(q[:, t])
        g = (lam[:, :, t] * dt[:, :, t]).double()
        b = ((1 - lam[:, :, t]) * dt[:, :, t] * alpha[:, :, t]).double()
        v = v_all[:, t].double()                                    # (B, H, P)
        S = alpha[:, :, t].double()[..., None, None] * S \
            + b[..., None, None] * torch.einsum("bhn,bhp->bhnp", k_prev, v_prev) \
            + g[..., None, None] * torch.einsum("bhn,bhp->bhnp", kbar, v)
        y[:, t] = torch.einsum("bhn,bhnp->bhp", qbar, S) + D.double()[None, :, None] * v
        k_prev, v_prev = kbar, v
    return y.reshape(Bsz, L, C).transpose(1, 2)


def main():
    if not torch.cuda.is_available():
        print("[parity] no CUDA device; run this on the training box."); sys.exit(2)
    if not fused_available():
        print(f"[parity] fused kernel UNAVAILABLE: {fused_reason()}"); sys.exit(1)
    dev = torch.device("cuda")
    torch.manual_seed(3)
    Bsz, H, P, L, N = 2, 6, 16, 517, 16  # N,P=16 = fwd+bwd tl.dot floors; odd L on purpose
    C = H * P
    n_rot = 4                             # rope_fraction 0.5 at N=16 -> 4 angles (even, <= N//2)
    xt = torch.randn(Bsz, C, L, device=dev)
    dt = F.softplus(torch.randn(Bsz, H, L, device=dev))
    Bn = torch.randn(Bsz, L, N, device=dev)
    Cn = torch.randn(Bsz, L, N, device=dev)
    lam_raw = torch.randn(Bsz, H, L, device=dev)          # per-HEAD
    theta_raw = 0.3 * torch.randn(Bsz, L, n_rot, device=dev)
    dd = 0.3 * torch.randn(Bsz, H, L, device=dev)
    a_t = -((dd.clamp_min(0) + torch.reciprocal(1 - dd.clamp_max(0)))).clamp(min=1e-4)
    b_bias = torch.ones(N, device=dev); c_bias = torch.ones(N, device=dev)
    D = torch.ones(H, device=dev)                          # per-HEAD

    yA = mamba3_fused(xt, dt, Bn, Cn, lam_raw, a_t, theta_raw, b_bias, c_bias, D, headdim=P)
    # THEIR reference, called with THEIR layouts (Q/K (B,L,G,N); V (B,L,H,P);
    # ADT/DT/Trap (B,H,L); Angles (B,L,H,n_ang) raw; biases (H,N)):
    Out_ref, _ = mamba3_siso_fwd_ref(
        Cn.unsqueeze(2), Bn.unsqueeze(2),
        xt.transpose(1, 2).reshape(Bsz, L, H, P),
        (a_t * dt), dt, lam_raw,
        c_bias.expand(H, N), b_bias.expand(H, N),
        theta_raw.to(torch.float32).unsqueeze(-2).expand(-1, -1, H, -1),
        D, None, None)
    yB = Out_ref.reshape(Bsz, L, H * P).transpose(1, 2).float()
    # fallback = per-head two-scan with per-head rotation (should now be ~exact)
    lam = torch.sigmoid(lam_raw)
    theta = torch.tanh(theta_raw) * math.pi
    yC = torch.empty_like(xt)
    for h in range(H):
        ang = torch.cumsum(dt.transpose(1, 2)[..., h:h + 1] * theta, dim=1)
        ang = ang - (2 * math.pi) * torch.floor(ang / (2 * math.pi))
        cos, sin = torch.cos(ang), torch.sin(ang)

        def rot_h(Pv):
            Pv = (Pv).view(Bsz, L, N // 2, 2)
            p0, p1 = Pv[..., 0].clone(), Pv[..., 1].clone()
            p0[..., :n_rot] = cos * Pv[..., :n_rot, 0] - sin * Pv[..., :n_rot, 1]
            p1[..., :n_rot] = sin * Pv[..., :n_rot, 0] + cos * Pv[..., :n_rot, 1]
            return torch.stack([p0, p1], dim=-1).view(Bsz, L, N)

        Bb = rot_h(Bn + b_bias).transpose(1, 2).contiguous()
        Cb = rot_h(Cn + c_bias).transpose(1, 2).contiguous()
        sl = slice(h * P, (h + 1) * P)
        yC[:, sl] = Mamba3._scan_core(
            xt[:, sl], dt[:, h:h + 1].expand(-1, P, -1).contiguous(), Bb, Cb,
            lam[:, h:h + 1].expand(-1, P, -1).contiguous(), None,
            D=D[h].expand(P), backend="auto",
            a_t=a_t[:, h:h + 1].expand(-1, P, -1).contiguous())

    eAB = float((yA - yB).abs().max()); rAB = eAB / float(yB.abs().max())
    eCB = float((yC - yB).abs().max()); rCB = eCB / float(yB.abs().max())
    print(f"[parity][semantic] our fallback vs THEIR reference: rel={rCB:.3e}  (assert <= 1e-3)")
    print(f"[parity][kernel]   Triton kernel vs THEIR reference: rel={rAB:.3e}  "
          f"(their own policy class: rtol=1e-1, fwd assert disabled; assert <= 0.15)")
    if rCB > 1e-3:
        print("[parity] FAIL[semantic]: OUR reading of Mamba-3 deviates from the authors' "
              "reference -- do not train; send this output back."); sys.exit(1)
    if rAB > 0.15:
        print("[parity] FAIL[kernel]: Triton kernel exceeds even the authors' own error "
              "class on this GPU -- train with --ssm-backend auto + grad checkpointing "
              "(exact fallback) instead; send this output back."); sys.exit(1)
    print(f"[parity] PASS -- semantics exact ({rCB:.1e}); kernel within its documented "
          f"approximation class ({rAB:.1e}). Training on the kernel accepts that "
          f"approximation for ~3x speed; the exact fallback remains one flag away.")


if __name__ == "__main__":
    main()
