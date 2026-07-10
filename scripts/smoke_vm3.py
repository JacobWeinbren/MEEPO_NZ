"""VM3 smoke test (CPU, no data, no GPU) -- the gate before GPU time.

Checks, in order:
  [1] flip-index is an involution over packed varlen sequences;
  [2] VoxelMamba3 forward on a synthetic 3-cloud batch: finite output,
      correct shape, per-point features;
  [3] backward: loss.backward() produces finite grads on every parameter
      that ought to receive one (incl. both DSB mixers, IWE, B/C biases,
      dt_bias, angles path);
  [4] PACKED-VARLEN CONSISTENCY: running clouds packed together equals
      running each cloud alone (torch-ref mixer is per-sequence exact, so
      this isolates the packing / serialization / flip / pool / Up plumbing);
  [5] decay banding: dt_bias spans [dt_min, dt_max] log-spaced per head;
  [6] end-to-end MeepoSeg(--backbone vm3) forward: logits (N, 2).

Run:  PYTHONPATH=. python3 scripts/smoke_vm3.py
"""
from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("POINT_MOE_DISABLE_SPCONV", "1")  # CPU-safe conv on GPU boxes

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from meepo_nz.models.point_structure import Point                     # noqa: E402
from meepo_nz.models.vm3 import (VoxelMamba3, PackedMamba3,           # noqa: E402
                                 _flip_index, _cu_from_batch)

torch.manual_seed(0)
np.random.seed(0)

TINY = dict(order=("hilbert", "hilbert-trans", "z", "z-trans"),
            stride=(2, 2), enc_depths=(1, 1, 1), enc_channels=(32, 64, 64),
            dec_depths=(1, 1), dec_channels=(32, 64),
            d_state=8, headdim=16, expand=2, mlp_ratio=2.0, drop_path=0.0,
            dsb_down=(1, 2, 2), iwe_window=8, use_cpe=True,
            decay_bands=True, chunk_size=16, shuffle_orders=False,
            stem_kernel_size=3, ssm_backend="torch",
            grad_checkpointing=False)


def _cloud(n, span=20.0, dl=0.1, feat_dim=6, seed=0):
    rng = np.random.default_rng(seed)
    coord = rng.uniform(0, span, size=(n, 3)).astype(np.float32)
    coord[:, 2] = rng.uniform(0, 8.0, size=n).astype(np.float32)
    grid = np.floor(coord / dl).astype(np.int32)
    grid[0] = (255, 255, 63)  # pin the serialization depth across clouds/batches
    feat = np.concatenate([coord / span, rng.normal(size=(n, feat_dim - 3))], axis=1)
    return coord, grid, feat.astype(np.float32)


def _point(sizes, feat_dim=6):
    coords, grids, feats = [], [], []
    for i, n in enumerate(sizes):
        c, g, f = _cloud(n, seed=10 + i)
        coords.append(c); grids.append(g); feats.append(f)
    offset = torch.tensor(np.cumsum(sizes), dtype=torch.long)
    return Point(coord=torch.from_numpy(np.concatenate(coords)),
                 grid_coord=torch.from_numpy(np.concatenate(grids)).int(),
                 feat=torch.from_numpy(np.concatenate(feats)),
                 offset=offset)


def main():
    ok = lambda i, msg: print(f"[smoke-vm3 {i}] OK  {msg}")

    # [1] flip involution ---------------------------------------------------
    cu = torch.tensor([0, 5, 5, 12, 20], dtype=torch.int32)
    rev = _flip_index(cu, torch.device("cpu"))
    idx = torch.arange(20)
    assert torch.equal(idx[rev][rev], idx), "flip must be an involution"
    assert torch.equal(rev[:5], torch.arange(4, -1, -1)), "first sequence reversed"
    ok(1, "flip-index involution over packed varlen (incl. empty sequence)")

    # [2] forward ------------------------------------------------------------
    sizes = [700, 500, 300]
    pt = _point(sizes)
    net = VoxelMamba3(in_channels=6, **TINY)
    impl = net.mixer_impl()
    out = net(Point(pt))
    assert out.feat.shape == (sum(sizes), net.out_channels)
    assert torch.isfinite(out.feat).all()
    ok(2, f"forward: ({sum(sizes)}, {net.out_channels}) finite, mixer={impl}")

    # [3] backward -----------------------------------------------------------
    out2 = net(Point(_point(sizes)))
    loss = out2.feat.square().mean()
    loss.backward()
    missing, nonfinite = [], []
    for name, p in net.named_parameters():
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all():
            nonfinite.append(name)
    assert not nonfinite, f"non-finite grads: {nonfinite[:5]}"
    # drop-path 0 + all branches active -> everything trainable should get grad
    hard_missing = [n for n in missing if not n.endswith("D")]  # D can be zero-grad if x tiny
    assert len(hard_missing) == 0, f"params with no grad: {hard_missing[:8]}"
    ok(3, f"backward: finite grads on {sum(1 for _ in net.parameters())} params")

    # [4] packed-varlen == per-cloud (plumbing exactness on the ref path) ----
    net.eval()
    with torch.no_grad():
        joint = net(Point(_point(sizes))).feat
        parts = []
        for i, n in enumerate(sizes):
            c, g, f = _cloud(n, seed=10 + i)
            single = Point(coord=torch.from_numpy(c),
                           grid_coord=torch.from_numpy(g).int(),
                           feat=torch.from_numpy(f),
                           offset=torch.tensor([n], dtype=torch.long))
            parts.append(net(single).feat)
        solo = torch.cat(parts, dim=0)
    err = (joint - solo).abs().max().item()
    assert err < 5e-4, f"packed vs per-cloud mismatch: max abs err {err:.3e}"
    ok(4, f"packed-varlen == per-cloud (max abs err {err:.2e})")

    # [5] decay bands ---------------------------------------------------------
    pm = PackedMamba3(32, d_state=8, headdim=16, expand=2, backend="torch",
                      dt_min=1e-3, dt_max=1e-1, decay_bands=True)
    dt = F.softplus(pm.mixer.dt_bias.detach().float())
    assert abs(dt.min().item() - 1e-3) < 1e-4 and abs(dt.max().item() - 1e-1) < 1e-2
    assert torch.all(dt[1:] >= dt[:-1]), "bands must be monotone"
    lg = torch.log(dt)
    gaps = lg[1:] - lg[:-1]
    assert (gaps.max() - gaps.min()).item() < 1e-3, "bands must be log-spaced"
    ok(5, f"decay bands: dt in [{dt.min():.4f}, {dt.max():.4f}] log-spaced over {dt.numel()} heads")

    # [6] end-to-end MeepoSeg(vm3) --------------------------------------------
    from meepo_nz.models.segmentation_model import MeepoSeg
    from meepo_nz.utils.config import Config
    cfg = Config()
    cfg.backbone = "vm3"
    cfg.use_dtm_raster = False        # backbone smoke: point features only
    cfg.scene_mode = True
    cfg.ssm_backend = "torch"
    cfg.shuffle_orders = False
    cfg.stem_kernel_size = 3
    for k, v in dict(vm3_order=TINY["order"], vm3_stride=(2, 2),
                     vm3_enc_depths=(1, 1, 1), vm3_enc_channels=(32, 64, 64),
                     vm3_dec_depths=(1, 1), vm3_dec_channels=(32, 64),
                     vm3_state=8, vm3_headdim=16, vm3_expand=2,
                     vm3_mlp_ratio=2.0, vm3_drop_path=0.0,
                     vm3_dsb_down=(1, 2, 2), vm3_iwe_window=8,
                     vm3_chunk_size=16).items():
        setattr(cfg, k, v)
    model = MeepoSeg(cfg, in_features_dim=6)
    pt = _point([400, 300])
    batch = {"feat": pt.feat, "coord": pt.coord, "grid_coord": pt.grid_coord,
             "offset": pt.offset}
    logits = model(batch)
    assert logits.shape == (700, 2) and torch.isfinite(logits).all()
    n_par = model.num_parameters()
    ok(6, f"MeepoSeg(vm3) end-to-end: logits (700, 2), params={n_par:,}")

    print("[smoke-vm3] ALL OK — safe to spend GPU time. "
          "(On the GPU box the mixer should report 'mamba3-official'; "
          "this CPU run used the torch reference.)")


if __name__ == "__main__":
    main()
