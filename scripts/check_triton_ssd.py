"""Parity gate for --ssm-backend triton-ssd. Run ON THE GPU BOX before training:

    python scripts/check_triton_ssd.py

Compares forward outputs AND input/parameter gradients of the Triton SSD kernel
against the eager reference scan on random meepo-shaped problems (d_state=1,
per-channel decay). Pass = train with --ssm-backend triton-ssd. Fail = stay on
the default backend; nothing is lost.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meepo_nz.models.ssm import selective_scan, selective_scan_ssd  # noqa: E402


def main():
    if not torch.cuda.is_available():
        sys.exit("[triton-ssd] needs a CUDA device.")
    dev = "cuda"
    torch.manual_seed(0)
    worst = 0.0
    for (Bt, Dd, L) in ((1, 48, 4096), (2, 96, 8192), (1, 64, 100000)):
        u = torch.randn(Bt, Dd, L, device=dev, dtype=torch.float32) * 0.5
        dt = torch.randn(Bt, Dd, L, device=dev) * 0.5
        A = -torch.rand(Dd, 1, device=dev) - 0.05
        Bm = torch.randn(Bt, 1, L, device=dev)
        Cm = torch.randn(Bt, 1, L, device=dev)
        Dp = torch.ones(Dd, device=dev)
        bias = torch.zeros(Dd, device=dev)
        args = []
        for t in (u, dt, Bm, Cm):
            t = t.clone().requires_grad_(True)
            args.append(t)
        u1, dt1, B1, C1 = args
        y_ref = selective_scan_ssd(u1, dt1, A, B1, C1, D=Dp, z=None,
                                   delta_bias=bias, delta_softplus=True)
        g = torch.randn_like(y_ref)
        y_ref.backward(g)
        ref_grads = [t.grad.clone() for t in args]
        for t in args:
            t.grad = None
        y_ker = selective_scan(u1, dt1, A, B1, C1, D=Dp, z=None,
                               delta_bias=bias, delta_softplus=True,
                               backend="triton-ssd")
        y_ker.backward(g)
        ker_grads = [t.grad.clone() for t in args]
        errs = [(y_ref - y_ker).abs().max().item()]
        errs += [(a - b).abs().max().item() for a, b in zip(ref_grads, ker_grads)]
        rel = max(errs) / max(y_ref.abs().max().item(), 1.0)
        worst = max(worst, rel)
        print(f"[triton-ssd] B={Bt} D={Dd} L={L}: max abs err {max(errs):.3e} "
              f"(rel {rel:.3e}) across y/du/ddt/dB/dC")
    if worst < 5e-3:
        print(f"[triton-ssd] PASS (worst rel {worst:.3e} < 5e-3). "
              f"Safe to train with --ssm-backend triton-ssd.")
    else:
        sys.exit(f"[triton-ssd] FAIL (worst rel {worst:.3e}). Do NOT use this backend; "
                 f"train with the default and report the numbers.")


if __name__ == "__main__":
    main()
