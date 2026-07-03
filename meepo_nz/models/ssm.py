"""Selective-scan SSM core for the MEEPO backbone (Mamba block).

MEEPO's Bidirectional Strided SSM (paper Fig. 6b) is a standard Mamba-1 selective
scan run over several token orderings. The ONE heavy primitive is the scan itself,
``y = SSM(u, delta, A, B, C, D)``. The reference code uses the fused CUDA kernel
``mamba_ssm.ops.selective_scan_interface.selective_scan_fn`` (compiled extension).
On Blackwell (sm_120) that kernel may or may not build, so this module exposes a
single ``selective_scan`` entry point with a backend switch:

    backend="cuda"  -> require the fused kernel (raises if unavailable)
    backend="torch" -> pure-PyTorch reference (works on ANY device incl. CPU)
    backend="auto"  -> use the kernel if importable, else fall back to torch

The pure-PyTorch path is mathematically identical to the kernel (same discretized
recurrence) -- only slower and more memory-hungry, since it materializes the
per-step state. It is what lets the model train on Blackwell with no compiled
dependency and what makes the CPU smoke test possible.

Signature matches ``selective_scan_fn`` exactly so the two are interchangeable:
    u, delta : (B, D, L)        input and (pre-softplus) timestep
    A        : (D, N)           state matrix (real, typically -exp(A_log))
    B, C     : (B, N, L)        input/output projections (selective => time-varying)
    D        : (D,)  or None    skip connection
    z        : (B, D, L) or None gating branch (SiLU), fused into the kernel
    delta_bias : (D,) or None   added to delta before softplus
    delta_softplus : bool       apply softplus(delta + delta_bias)
returns y : (B, D, L)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["selective_scan", "selective_scan_ref", "selective_scan_ssd", "kernel_available"]


def kernel_available() -> bool:
    """True iff the fused mamba_ssm CUDA kernel can be imported."""
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
        return True
    except Exception:
        return False


def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False):
    """Pure-PyTorch selective scan. Differentiable, device-agnostic.

    Mirrors mamba_ssm's reference implementation (selective_scan_ref): discretize
    with zero-order hold (deltaA = exp(delta * A)), run the linear recurrence
    h_t = deltaA_t * h_{t-1} + (delta_t * B_t) * u_t, read out y_t = C_t . h_t,
    then add the D skip and optional SiLU(z) gate.
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, seqlen = u.shape
    n = A.shape[1]

    # Time-varying discretization (selective): deltaA (B,D,L,N), deltaBu (B,D,L,N).
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A.float()))
    if B.dim() == 3:                                   # (B, N, L) selective
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B.float(), u)
    else:                                              # (D, N) time-invariant
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B.float(), u)

    x = u.new_zeros((batch, dim, n))
    ys = []
    for t in range(seqlen):
        x = deltaA[:, :, t] * x + deltaB_u[:, :, t]    # (B, D, N)
        if C.dim() == 3:                               # (B, N, L)
            y = torch.einsum("bdn,bn->bd", x, C[:, :, t].float())
        else:                                          # (D, N)
            y = torch.einsum("bdn,dn->bd", x, C.float())
        ys.append(y)
    y = torch.stack(ys, dim=2)                         # (B, D, L)

    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(dtype_in)


_SSM_BACKEND_REPORTED = False


def _discretize(u, delta, A, B, delta_bias, delta_softplus):
    """Shared fp32 preamble: softplus(delta+bias), lam = delta*A (log-decay),
    deltaB_u = (delta*B)*u. Returns (u32, lam, deltaB_u) with lam/deltaB_u (B,D,L,N)."""
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    lam = torch.einsum("bdl,dn->bdln", delta, A.float())          # log of deltaA
    if B.dim() == 3:                                              # (B, N, L) selective
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B.float(), u)
    else:                                                         # (D, N) time-invariant
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B.float(), u)
    return u, lam, deltaB_u


def selective_scan_ssd(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False, chunk=None):
    """Chunked selective scan -- the Mamba-2 SSD algorithm (Dao & Gu 2024, 'Transformers
    are SSMs') applied to our Mamba-1 recurrence, in pure PyTorch.

    Mathematically IDENTICAL to ``selective_scan_ref`` (verified to 1e-14 in fp64: it
    computes the same products exp(sum lam), just re-bracketed): within a chunk of
    ``chunk`` steps the pairwise decays exp(S_t - S_s) form one lower-triangular matrix
    applied by matmul (parallel); only the L/chunk chunk boundaries stay sequential.

    Where it wins: on GPU, the reference loop's cost is L sequential python steps (kernel
    launches that cannot overlap); SSD cuts sequential steps ~chunk-fold, at the price of
    ~chunk-fold more (cheap, batched) FLOPs. On CPU the tradeoff INVERTS -- the loop has
    no launch overhead and SSD's extra FLOPs make it comparable or slower -- which is why
    the auto backend only picks SSD for CUDA tensors. Peak extra memory is one chunk's
    decay matrix, (B, D, chunk, chunk, N) -- ~13 MB at D=768, chunk=64, fp32.
    No dependencies beyond torch; differentiable; fp32 internals like the reference.
    """
    import os
    if chunk is None:
        chunk = int(os.environ.get("POINT_MOE_SSD_CHUNK", "64"))
    dtype_in = u.dtype
    u, lam, b = _discretize(u, delta, A, B, delta_bias, delta_softplus)
    batch, dim, L = u.shape
    n = A.shape[1]
    c = max(int(chunk), 1)

    pad = (-L) % c
    if pad:                                                       # a=exp(0)=1, b=0: inert tail
        lam = F.pad(lam, (0, 0, 0, pad))
        b = F.pad(b, (0, 0, 0, pad))
    nc = lam.shape[2] // c
    lam = lam.view(batch, dim, nc, c, n)
    b = b.view(batch, dim, nc, c, n)

    tri = torch.ones(c, c, dtype=torch.bool, device=u.device).tril()
    neg_inf = float("-inf")
    C32 = C.float()
    C_sel = C32.dim() == 3
    if C_sel:                                                     # (B,N,L) -> (B,N,nc,c)
        Cp = F.pad(C32, (0, pad)).view(batch, -1, nc, c)

    def _chunk(lam_k, b_k, C_k, h_in):
        """One chunk: (y_k, h_out) from its slice + carried state. Pure in its tensor
        args, so it can be gradient-checkpointed: backward re-runs THIS chunk only."""
        S_k = torch.cumsum(lam_k, dim=2)                          # (B,D,c,N)
        # M[t,s] = exp(S_t - S_s) for t >= s else 0  == prod_{r=s+1..t} deltaA_r
        diff = S_k.unsqueeze(3) - S_k.unsqueeze(2)                # (B,D,t,s,N)
        M = torch.exp(diff.masked_fill(~tri.view(1, 1, c, c, 1), neg_inf))
        h_k = torch.einsum("bdtsn,bdsn->bdtn", M, b_k)            # within-chunk states
        h_k = h_k + torch.exp(S_k) * h_in.unsqueeze(2)            # + carried state
        if C_sel:
            y_k = torch.einsum("bdtn,bnt->bdt", h_k, C_k)
        else:
            y_k = torch.einsum("bdtn,dn->bdt", h_k, C_k)
        return y_k, h_k[:, :, -1]

    # Training memory: WITHOUT checkpointing autograd retains every chunk's (B,D,c,c,N)
    # decay matrix M for backward -- L/chunk of them, gigabytes at scale. Checkpointing
    # each chunk stores only its slim inputs and recomputes M in backward: peak extra
    # memory = ONE M (megabytes), cost ~1.3x scan compute. Disable: POINT_MOE_SSD_CKPT=0.
    needs_grad = torch.is_grad_enabled() and (lam.requires_grad or b.requires_grad
                                              or C32.requires_grad)
    use_ckpt = needs_grad and os.environ.get("POINT_MOE_SSD_CKPT", "1") != "0"
    if use_ckpt:
        from torch.utils.checkpoint import checkpoint as _ckpt

    h_in = u.new_zeros((batch, dim, n))
    ys = []
    for k in range(nc):                                           # sequential ONLY over chunks
        C_k = Cp[:, :, k] if C_sel else C32
        if use_ckpt:
            y_k, h_in = _ckpt(_chunk, lam[:, :, k], b[:, :, k], C_k, h_in,
                              use_reentrant=False)
        else:
            y_k, h_in = _chunk(lam[:, :, k], b[:, :, k], C_k, h_in)
        ys.append(y_k)
    y = torch.cat(ys, dim=2)[:, :, :L]                            # (B,D,L)

    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(dtype_in)


def selective_scan(u, delta, A, B, C, D=None, z=None,
                   delta_bias=None, delta_softplus=False, backend="auto"):
    """Backend-dispatching selective scan. See module docstring for semantics.

    auto  -> fused mamba_ssm CUDA kernel if importable, else the chunked SSD scan
    cuda  -> require the fused kernel (raises if unavailable)
    ssd   -> chunked SSD scan (Mamba-2 algorithm, pure torch; fast, no deps)
    torch / ref -> the naive per-step reference loop (slow; kept for validation)
    """
    global _SSM_BACKEND_REPORTED
    if backend not in ("auto", "cuda", "ssd", "torch", "ref"):
        raise ValueError(f"unknown ssm backend {backend!r}")
    if backend in ("auto", "cuda"):
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            out = selective_scan_fn(
                u, delta, A, B, C, D, z=z,
                delta_bias=delta_bias, delta_softplus=delta_softplus,
                return_last_state=False,
            )
            if not _SSM_BACKEND_REPORTED:
                _SSM_BACKEND_REPORTED = True
                print("[ssm] selective_scan: using the FUSED mamba_ssm CUDA kernel.", flush=True)
            return out
        except Exception as _e:
            if backend == "cuda":
                raise   # caller explicitly demanded the kernel; surface the failure
            # auto fallback: SSD on GPU (launch-overhead-bound -> SSD's ~chunk-fold fewer
            # sequential steps win); naive loop on CPU (FLOP-bound -> the loop is fine).
            picked = "ssd" if u.is_cuda else "ref"
            if not _SSM_BACKEND_REPORTED:
                _SSM_BACKEND_REPORTED = True
                if picked == "ssd":
                    print(f"[ssm] fused mamba_ssm kernel unavailable ({type(_e).__name__}) -> using the "
                          f"chunked SSD scan (Mamba-2 algorithm, pure torch). Same math (verified); "
                          f"slower than the fused kernel but ~chunk-fold fewer sequential steps than "
                          f"the naive loop. To require the kernel: --ssm-backend cuda.", flush=True)
                else:
                    print(f"[ssm] fused mamba_ssm kernel unavailable ({type(_e).__name__}); CPU device "
                          f"-> using the reference loop (SSD only helps on GPU).", flush=True)
            backend = picked
    if backend == "ssd":
        return selective_scan_ssd(
            u, delta, A, B, C, D=D, z=z,
            delta_bias=delta_bias, delta_softplus=delta_softplus,
        )
    if not _SSM_BACKEND_REPORTED:
        _SSM_BACKEND_REPORTED = True
        print("[ssm] selective_scan: using the NAIVE reference loop (validation backend; slow).",
              flush=True)
    return selective_scan_ref(
        u, delta, A, B, C, D=D, z=z,
        delta_bias=delta_bias, delta_softplus=delta_softplus,
    )
