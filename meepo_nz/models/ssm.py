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

import os

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
                       delta_bias=None, delta_softplus=False,
                       h0=None, return_last_state=False):
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

    x = u.new_zeros((batch, dim, n)) if h0 is None else h0.float()
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
    y = y.to(dtype_in)
    return (y, x) if return_last_state else y


_SSM_BACKEND_REPORTED = False


# ---- SSD hot ops, hoisted to module level so torch.compile caches them ---------------
# (compiling per-call closures would recompile on EVERY scan call). The same functions
# serve the eager path, so compiled and eager can never diverge in math.

def _ssd_intra_sel(S_g, b_g, C_g, notri):
    diff = (S_g.unsqueeze(4) - S_g.unsqueeze(3)).masked_fill(notri, float("-inf"))
    M = torch.exp(diff)                                           # (B,D,G,c,c,N)
    h = torch.einsum("bdgtsn,bdgsn->bdgtn", M, b_g)
    return torch.einsum("bdgtn,bngt->bdgt", h, C_g), h[:, :, :, -1]


def _ssd_intra_ti(S_g, b_g, C_g, notri):
    diff = (S_g.unsqueeze(4) - S_g.unsqueeze(3)).masked_fill(notri, float("-inf"))
    M = torch.exp(diff)
    h = torch.einsum("bdgtsn,bdgsn->bdgtn", M, b_g)
    return torch.einsum("bdgtn,dn->bdgt", h, C_g), h[:, :, :, -1]


def _ssd_carry_sel(S_g, C_g, hin_g):
    return torch.einsum("bdgtn,bdgn,bngt->bdgt", torch.exp(S_g), hin_g, C_g)


def _ssd_carry_ti(S_g, C_g, hin_g):
    return torch.einsum("bdgtn,bdgn,dn->bdgt", torch.exp(S_g), hin_g, C_g)


_SSD_EAGER = {"intra_sel": _ssd_intra_sel, "intra_ti": _ssd_intra_ti,
              "carry_sel": _ssd_carry_sel, "carry_ti": _ssd_carry_ti}
_SSD_COMPILE = {"tried": False, "fns": None, "failed": False}


def _ssd_fns():
    """Opt-in torch.compile of the SSD hot ops (POINT_MOE_SSD_COMPILE=1). On Windows this
    rides `pip install triton-windows` (Triton JIT-compiles GPU kernels with its bundled
    LLVM -- no MSVC, no admin). EXPERIMENTAL: any failure at compile time or on first
    execution falls back to eager for the rest of the run, with one loud message."""
    import os
    st = _SSD_COMPILE
    if not st["tried"]:
        st["tried"] = True
        if os.environ.get("POINT_MOE_SSD_COMPILE", "0") == "1":
            try:
                st["fns"] = {k: torch.compile(f) for k, f in _SSD_EAGER.items()}
                print("[ssm] POINT_MOE_SSD_COMPILE=1: torch.compile wrapped the SSD ops "
                      "(first call per shape JIT-compiles -- expect the first step to be "
                      "slow; steady-state should be faster).", flush=True)
            except Exception as e:
                st["failed"] = True
                print(f"[ssm] POINT_MOE_SSD_COMPILE=1 but torch.compile setup failed "
                      f"({type(e).__name__}: {e}) -> eager SSD ops.", flush=True)
    return st["fns"] if (st["fns"] and not st["failed"]) else None


try:                                                            # checkpoint's internal
    from torch.utils.checkpoint import _StopRecomputationError as _StopRecomp
except Exception:                                               # control-flow exception --
    class _StopRecomp(BaseException):                           # must NEVER be swallowed
        pass


def _ssd_call(key, *args):
    """Compiled op if enabled and healthy; eager otherwise. Fallback wrapping applies only
    until the first success ('committed'), after which the compiled op is called bare so
    checkpoint's control-flow exceptions (early-stop of recompute) pass through untouched.
    A pre-commit failure permanently falls back (ops are pure, so mid-run fallback is safe)."""
    fns = _ssd_fns()
    if fns is None:
        return _SSD_EAGER[key](*args)
    if _SSD_COMPILE.get("ok"):
        return fns[key](*args)
    try:
        out = fns[key](*args)
    except _StopRecomp:
        raise
    except Exception as e:
        _SSD_COMPILE["failed"] = True
        print(f"[ssm] compiled SSD op '{key}' failed ({type(e).__name__}: {e}) -> eager "
              f"for the rest of the run.", flush=True)
        return _SSD_EAGER[key](*args)
    _SSD_COMPILE["ok"] = True
    return out


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
                       delta_bias=None, delta_softplus=False, chunk=None,
                       h0=None, return_last_state=False):
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
    S = torch.cumsum(lam, dim=3)                                  # S_t = sum_{r<=t} lam_r

    tri = torch.ones(c, c, dtype=torch.bool, device=u.device).tril()
    notri = (~tri).view(1, 1, 1, c, c, 1)
    C32 = C.float()
    C_sel = C32.dim() == 3
    if C_sel:                                                     # (B,N,L) -> (B,N,nc,c)
        Cp = F.pad(C32, (0, pad)).view(batch, -1, nc, c)

    # Group size: process G chunks per batched op, sized so one group's pairwise-decay
    # tensor (B,D,G,c,c,N) stays within a memory budget (default 128 MB; env-tunable).
    budget = float(os.environ.get("POINT_MOE_SSD_GROUP_MB", "128")) * 1e6
    G = max(1, min(nc, int(budget / max(batch * dim * c * c * n * 4, 1))))

    key_i = "intra_sel" if C_sel else "intra_ti"
    key_c = "carry_sel" if C_sel else "carry_ti"

    def _intra(S_g, b_g, C_g):
        return _ssd_call(key_i, S_g, b_g, C_g, notri)

    def _carry_readout(S_g, C_g, hin_g):
        return _ssd_call(key_c, S_g, C_g, hin_g)

    needs_grad = torch.is_grad_enabled() and (lam.requires_grad or b.requires_grad
                                              or C32.requires_grad)
    use_ckpt = needs_grad and os.environ.get("POINT_MOE_SSD_CKPT", "1") != "0"
    if use_ckpt:
        from torch.utils.checkpoint import checkpoint as _ckpt
        run_intra = lambda *a: _ckpt(_intra, *a, use_reentrant=False)
        run_carry = lambda *a: _ckpt(_carry_readout, *a, use_reentrant=False)
    else:
        run_intra, run_carry = _intra, _carry_readout

    def _Cslice(g0, g1):
        return Cp[:, :, g0:g1] if C_sel else C32

    # Pass 1 (parallel, grouped): within-chunk outputs + each chunk's final intra state.
    y_intras, intra_lasts = [], []
    for g0 in range(0, nc, G):
        g1 = min(g0 + G, nc)
        y_g, last_g = run_intra(S[:, :, g0:g1], b[:, :, g0:g1], _Cslice(g0, g1))
        y_intras.append(y_g)
        intra_lasts.append(last_g)
    intra_last = torch.cat(intra_lasts, dim=2)                    # (B,D,nc,N)
    a_chunk = torch.exp(S[:, :, :, -1])                           # whole-chunk decay (B,D,nc,N)

    # Sequential part: ONLY the tiny (B,D,N) carry recurrence over nc chunks
    # (h_out[k] = a_chunk[k]*h_in[k] + intra_last[k]); ~1 fused op per chunk.
    h = u.new_zeros((batch, dim, n)) if h0 is None else h0.float()
    h_ins = []
    for k in range(nc):
        h_ins.append(h)
        h = torch.addcmul(intra_last[:, :, k], a_chunk[:, :, k], h)
    h_ins = torch.stack(h_ins, dim=2)                             # (B,D,nc,N)

    # Pass 2 (parallel, grouped): add each incoming state's contribution to the outputs.
    ys = []
    for i, g0 in enumerate(range(0, nc, G)):
        g1 = min(g0 + G, nc)
        ys.append(y_intras[i] + run_carry(S[:, :, g0:g1], _Cslice(g0, g1), h_ins[:, :, g0:g1]))
    y = torch.cat(ys, dim=2).reshape(batch, dim, nc * c)[:, :, :L]

    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    y = y.to(dtype_in)
    # h after the carry loop is the exact final state: padded tail chunks are
    # inert (decay exp(0)=1, input 0), so they pass the state through unchanged.
    return (y, h) if return_last_state else y


def selective_scan(u, delta, A, B, C, D=None, z=None,
                   delta_bias=None, delta_softplus=False, backend="auto",
                   h0=None, return_last_state=False):
    """Backend-dispatching selective scan. See module docstring for semantics.

    auto  -> fused mamba_ssm CUDA kernel if importable, else the chunked SSD scan
    cuda  -> require the fused kernel (raises if unavailable)
    ssd   -> chunked SSD scan (Mamba-2 algorithm, pure torch; fast, no deps)
    torch / ref -> the naive per-step reference loop (slow; kept for validation)
    """
    global _SSM_BACKEND_REPORTED
    if backend not in ("auto", "cuda", "ssd", "torch", "ref", "triton-ssd"):
        raise ValueError(f"unknown ssm backend {backend!r}")
    if h0 is not None or return_last_state:
        # exact state carry (sequence slicing) is served by the pure-torch backends;
        # the fused kernel path is not extended. Same math either way (verified).
        picked = "ssd" if u.is_cuda else "ref"
        fn = selective_scan_ssd if picked == "ssd" else selective_scan_ref
        return fn(u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias,
                  delta_softplus=delta_softplus, h0=h0,
                  return_last_state=return_last_state)
    if backend == "triton-ssd":
        # EXPERIMENTAL kernel backend: meepo's scan (d_state=1, per-channel decay,
        # shared B/C) maps exactly onto Mamba-2 SSD with nheads=d_inner, headdim=1,
        # ngroups=1, N=1. Pure Triton (works with triton-windows). Accuracy-
        # equivalent, NOT bit-exact to the eager scan; gate with
        # scripts/check_triton_ssd.py on the GPU box before training.
        try:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        except Exception as e:
            raise RuntimeError(f"--ssm-backend triton-ssd needs mamba_ssm's Triton ops "
                               f"(pip install --no-deps -e third_party/mamba-main; triton required): {e}")
        Bt, Dd, L = u_.shape if False else (u.shape[0], u.shape[1], u.shape[2])
        x = u.transpose(1, 2).unsqueeze(-1)                      # (B,L,H=D,P=1)
        dtb = delta.transpose(1, 2)                              # (B,L,H) raw (bias/softplus in-kernel)
        Bm = B.transpose(1, 2).unsqueeze(2) if B.dim() == 3 else B   # (B,L,G=1,N)
        Cm = C.transpose(1, 2).unsqueeze(2) if C.dim() == 3 else C
        y = mamba_chunk_scan_combined(
            x, dtb, A.squeeze(-1), Bm, Cm, chunk_size=int(os.environ.get("POINT_MOE_SSD_CHUNK", "128") or 128),
            D=D, z=None, dt_bias=delta_bias, dt_softplus=delta_softplus)
        return y.squeeze(-1).transpose(1, 2)                     # (B,D,L)
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
        except torch.cuda.OutOfMemoryError:
            raise   # NEVER fall back on OOM: the SSD path allocates far MORE memory
                    # (intra-chunk tensors scale with d_state) -- falling back turns a
                    # transient OOM into a guaranteed one. Lower the batch instead.
        except Exception as _e:
            if backend == "cuda":
                raise   # caller explicitly demanded the kernel; surface the failure
            # auto fallback: SSD on GPU (launch-overhead-bound -> SSD's ~chunk-fold fewer
            # sequential steps win); naive loop on CPU (FLOP-bound -> the loop is fine).
            picked = "ssd" if u.is_cuda else "ref"
            # Report EVERY DISTINCT failure reason once (not just the first ever): a
            # later call with a different shape/dtype profile silently routing to SSD
            # after the 'FUSED' banner is how the 2026-07-09 96GB OOM happened.
            global _SSM_FALLBACK_SEEN
            try:
                _SSM_FALLBACK_SEEN
            except NameError:
                _SSM_FALLBACK_SEEN = set()
            _key = (type(_e).__name__, str(_e)[:160])
            if _key not in _SSM_FALLBACK_SEEN:
                _SSM_FALLBACK_SEEN.add(_key)
                print(f"[ssm] WARNING fused kernel REJECTED this call profile "
                      f"(u{tuple(u.shape)} {u.dtype}, B{tuple(B.shape)} {B.dtype}, "
                      f"N={A.shape[1]}): {type(_e).__name__}: {str(_e)[:160]} "
                      f"-> falling back to '{picked}'. At large d_state the SSD "
                      f"fallback is MUCH heavier -- use --ssm-backend cuda to fail "
                      f"loudly instead.", flush=True)
            if not _SSM_BACKEND_REPORTED:
                _SSM_BACKEND_REPORTED = True
                if picked == "ssd":
                    print(f"[ssm] fused mamba_ssm kernel unavailable ({type(_e).__name__}) -> using the "
                          f"chunked SSD scan (Mamba-2 algorithm, pure torch). Same math (verified); "
                          f"(experimental extra speed: pip install triton-windows + POINT_MOE_SSD_COMPILE=1) "
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
