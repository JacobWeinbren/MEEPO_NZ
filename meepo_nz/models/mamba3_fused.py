"""Fused Mamba-3 SISO path: thin adapter onto the OFFICIAL Triton kernels
(vendored verbatim from state-spaces/mamba @ f577286 into ops/triton_mamba3;
only the package-internal import paths were rewritten).

Layout mapping (ours -> kernel):
  channels are heads at headdim P=1, one shared Q/K group (GQA ngroups=1):
    V   (B, half, L)  -> (B, L, half, 1)
    Q/K (B, L, N) post-RMSNorm, PRE-bias -> (B, L, 1, N); biases go IN (kernel
        adds per-head; ours are direction-shared, expanded across heads)
    DT/ADT/Trap (B, half, L) -> (B, half, L)  [kernel wants (b, nheads, l)]
    Angles = RAW theta logits, fp32, expanded per-head (B, L, half, n_ang):
        the kernel applies tanh(.)*pi INTERNALLY, then cumsum(Angles*DT) mod
        2pi with PER-HEAD dt -- the reference rotation semantics (the two-scan
        fallback's channel-mean dt is the documented CPU-only approximation).
    Trap = RAW lambda logits: the kernel applies sigmoid INTERNALLY (batch
        path of the official module passes both raw; the step path activates
        module-side -- easy to conflate, verified against modules/mamba3.py).
  Z=None (host gating stays outside), D per head, chunk_size 64.

Preprocessing parity with the official module (modules/mamba3.py):
  DT = softplus(raw + dt_bias); A_t = -heavy_tail(dd_A) clamped <= -1e-4;
  ADT = A_t * DT; B/C RMS-normed before the call; Trap/Angles RAW (see above).

Availability is resolved lazily and cached: requires CUDA tensors + importable
triton + the vendored package. Any import/runtime failure (except OOM) prints
ONE loud warning with the reason and flips to the two-scan fallback -- the
2026-07-09 silent-fallback OOM taught that lesson.
"""
from __future__ import annotations

import os

import torch

_STATE = {"checked": False, "fn": None, "reason": None, "warned": False}


def fused_available():
    if not _STATE["checked"]:
        _STATE["checked"] = True
        if os.environ.get("POINT_MOE_DISABLE_TRITON3", "") == "1":
            _STATE["reason"] = "disabled via POINT_MOE_DISABLE_TRITON3=1"
        else:
            try:
                from ..ops.triton_mamba3.mamba3_siso_combined import mamba3_siso_combined
                _STATE["fn"] = mamba3_siso_combined
            except Exception as e:  # no triton / no GPU build / import error
                _STATE["reason"] = f"{type(e).__name__}: {str(e)[:120]}"
    return _STATE["fn"] is not None


def fused_reason():
    return _STATE["reason"]


def warn_fallback_once(exc=None):
    if not _STATE["warned"]:
        _STATE["warned"] = True
        why = f"{type(exc).__name__}: {str(exc)[:160]}" if exc is not None else _STATE["reason"]
        print(f"[mamba3] WARNING: official Triton kernel unavailable/failed ({why}) "
              f"-> two-scan fallback (slower; rotation uses channel-mean dt). "
              f"Set POINT_MOE_DISABLE_TRITON3=1 to silence.", flush=True)


def mamba3_fused(xi, dt, Bn, Cn, lam_raw, a_t, theta_raw, b_bias, c_bias, D,
                 headdim=8, chunk_size=64):
    """xi: (B, C, L) channels, grouped into H = C//headdim heads of headdim P
    (the kernel's TMA descriptors require P*elemsize >= 16 bytes -> P >= 8 at
    bf16; P=1 was rejected on sm_120 -- the 2026-07-09 parity-gate catch).
    dt/lam_raw/a_t: (B, H, L) PER-HEAD (the reference granularity);
    Bn/Cn: (B, L, N) post-norm PRE-bias; theta_raw: (B, L, n_ang) RAW logits;
    b_bias/c_bias: (N,); D: (H,). Returns (B, C, L)."""
    fn = _STATE["fn"]
    Bsz, C, L = xi.shape
    P = int(headdim); H = C // P
    N = Bn.shape[-1]
    n_ang = theta_raw.shape[-1]
    # PREFLIGHT: the kernel's hardware floors, learned one parity-gate catch at a
    # time on sm_120 (2026-07-09): tl.dot contraction K >= 16 -> d_state >= 16;
    # TMA descriptor last dim >= 16 bytes -> headdim >= 8 @ bf16; angle count
    # even and <= d_state//2. Fail in English, not in a Triton stack trace.
    problems = []
    if N < 16: problems.append(f"d_state={N} < 16 (tl.dot K-floor)")
    if P < 16: problems.append(f"headdim={P} < 16 (backward tl.dot contracts over headdim)")
    if C % P != 0: problems.append(f"channels={C} not divisible by headdim={P}")
    if n_ang % 2 != 0 or n_ang > N // 2:
        problems.append(f"angles={n_ang} (must be even and <= d_state//2={N//2})")
    if problems:
        raise ValueError("mamba3 kernel geometry: " + "; ".join(problems))
    ADT = (a_t * dt).contiguous()
    angles = theta_raw.to(torch.float32).unsqueeze(-2).expand(-1, -1, H, -1).contiguous()
    V = xi.transpose(1, 2).reshape(Bsz, L, H, P).contiguous()
    y = fn(
        Q=Cn.unsqueeze(2).contiguous(),            # (B, L, 1, N) -- GQA group
        K=Bn.unsqueeze(2).contiguous(),
        V=V,
        ADT=ADT, DT=dt.contiguous(), Trap=lam_raw.contiguous(),
        Q_bias=c_bias.expand(H, N).contiguous(),
        K_bias=b_bias.expand(H, N).contiguous(),
        Angles=angles, D=D.contiguous(), Z=None,
        chunk_size=chunk_size, Input_States=None, return_final_states=False,
    )
    return y.reshape(Bsz, L, C).transpose(1, 2).contiguous()
