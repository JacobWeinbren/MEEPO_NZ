"""Fast differentiable row-gather.

Autograd's default backward for advanced indexing (``src[idx]`` with a long
index tensor) is ``_index_put_impl_(accumulate=True)``, whose generic
``indexing_backward_kernel`` is pathologically slow on some CUDA builds --
notably torch 2.12 / CUDA 13 on Blackwell (sm_120), where a 16k-point conv
gather backward measured ~76 ms/call (96% of total step time) even with
deterministic algorithms DISABLED. The matmul path is unaffected (cuBLAS ships
tuned sm_120 kernels); only this hand-written scatter regresses.

``index_add_`` (and the ``scatter_*`` family) use a different, fast kernel on
the same stack -- the model's pooling already relies on ``scatter_reduce_``
without showing up in the profile. We therefore route the gather's backward
through ``index_add_``. The forward is ``index_select`` (identical values to
``src[idx]``); the gradient is the same accumulate-by-index, so this is a pure
kernel swap -- numerically equivalent up to float add-order, no change to model
semantics or to the smoke-test outputs.
"""

import torch


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx):           # src: (M, C); idx: (P,) long -> (P, C)
        ctx.save_for_backward(idx)
        ctx.num_rows = src.shape[0]
        return src.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad_out):          # grad_out: (P, C) -> grad_src: (M, C)
        (idx,) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_src = grad_out.new_zeros(ctx.num_rows, grad_out.shape[-1])
        grad_src.index_add_(0, idx, grad_out)
        return grad_src, None


def gather_rows(src, idx):
    """``src[idx]`` along dim 0, differentiable wrt ``src`` via a fast
    ``index_add_`` backward.

    src : (M, C)
    idx : (...,) long tensor of row ids; output is (*idx.shape, C).
    Values are identical to ``src[idx]``; only the backward kernel differs.
    """
    flat = idx.reshape(-1)
    out = _GatherRows.apply(src, flat)
    return out.reshape(*idx.shape, src.shape[-1])
