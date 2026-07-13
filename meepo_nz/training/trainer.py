"""
Training loop for MEEPO (PTv3 + sparse MoE).

Implements every per-epoch requirement:

  * frequent, varying logging (running loss, throughput, elapsed, ETA);
  * a checkpoint saved **every epoch**;
  * error images in the paper's style saved **every epoch** (``n_vis_tiles``);
  * classified ``.laz`` files written **every epoch** for the same tiles;
  * a training dashboard PNG (loss / metrics / epoch-time) refreshed **every
    epoch**.

Mixed precision (AMP) is enabled on CUDA (good for Blackwell), with gradient
clipping and multiplicative LR decay per epoch, mirroring the KPConv training
recipe.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import ConfusionAccumulator, RMSEAccumulator
from .losses import SegLoss, inverse_frequency_weights
from .visualize import (render_error_image, update_training_charts, render_epoch_panels,
                        render_review_panel, render_scene_report,
                        update_refine_charts, render_spag_dc_panel)
from ..data.batch import move_batch
from ..utils.laz_io import write_classified, IGNORE_LABEL


def _fmt_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _per_cloud(packed: torch.Tensor, lengths) -> List[np.ndarray]:
    out, i0 = [], 0
    for L in lengths:
        L = int(L)
        out.append(packed[i0:i0 + L].detach().cpu().numpy())
        i0 += L
    return out


def _vis_feature_columns(cfg):
    """Indices + labels of the *interpretable* input feature channels, for the
    per-epoch panels. Skips the raw xyz coordinates, the constant channel and the
    six higher-order moments (x^2..yz, not visually meaningful); keeps mean
    elevation, curvature, the number-of-returns count and intensity.
    Indices follow the assemble_features column order exactly."""
    if getattr(cfg, "use_rgb", False):
        return []
    cols, i = [], 0
    if getattr(cfg, "use_xyz_in_features", False): i += 3
    if getattr(cfg, "use_constant_feature", False): i += 1
    if cfg.use_mean_elevation: cols.append((i, "Mean elevation")); i += 1
    if cfg.use_curvature:      cols.append((i, "Surface curvature")); i += 1
    if cfg.use_higher_moments: i += 6
    if getattr(cfg, "use_return_features", False):
        cols.append((i, "Number of returns")); i += 1
    if getattr(cfg, "use_return_ratio", False):
        cols.append((i, "Return ratio")); i += 1
    if getattr(cfg, "use_intensity", False):
        cols.append((i, "Intensity")); i += 1
    if getattr(cfg, "use_prev_dtm", False): cols.append((i, "Height above prev DTM")); i += 1
    return cols


def load_pretrained(model, path, verbose=True):
    """Weights-only transfer init: load a checkpoint's model weights into ``model``
    WITHOUT optimizer/scheduler/epoch state (that is what distinguishes transfer
    fine-tuning from resuming). Accepts trainer checkpoints ({'model_state': ...}) or
    raw state_dicts; strips 'module.' / '_orig_mod.' wrapper prefixes.

    Shape-mismatched tensors are a hard error with cause guidance (they mean the
    architectures genuinely differ -- e.g. tiles built with vs without a prior change
    in_features_dim). Missing/unexpected KEYS are tolerated and reported (e.g. a
    pretrain without --spag-rl fine-tuned with it: the RL head initialises fresh)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    clean = {}
    for k, v in sd.items():
        for pre in ("_orig_mod.", "module."):
            while k.startswith(pre):
                k = k[len(pre):]
        clean[k] = v
    own = model.state_dict()
    ok = {k: v for k, v in clean.items() if k in own and own[k].shape == v.shape}
    bad = [(k, tuple(clean[k].shape), tuple(own[k].shape))
           for k in clean if k in own and own[k].shape != clean[k].shape]
    missing = [k for k in own if k not in clean]
    unexpected = [k for k in clean if k not in own]
    if bad:
        lines = "\n".join(f"    {k}: checkpoint{s_} vs model{m}" for k, s_, m in bad[:8])
        raise SystemExit(
            f"[init-from] SHAPE MISMATCH on {len(bad)} tensor(s):\n{lines}\n"
            f"[init-from] The architectures differ. Usual causes: different backbone dims, "
            f"num_classes, or architecture flags between the pretrain and this run. "
            f"Pretrain and fine-tune tile sets must share dl and feature layout.")
    frac = len(ok) / max(len(own), 1)
    if frac < 0.5:
        raise SystemExit(f"[init-from] only {100*frac:.0f}% of model tensors found in "
                         f"{path!r} -- wrong checkpoint?")
    model.load_state_dict(ok, strict=False)
    if verbose:
        print(f"[init-from] loaded {len(ok)}/{len(own)} tensors from {path}"
              + (f"; fresh-init (missing in ckpt): {len(missing)}" if missing else "")
              + (f"; ignored (not in model): {len(unexpected)}" if unexpected else ""),
              flush=True)
        if missing:
            heads = sorted({m.split(".")[0] for m in missing})
            print(f"[init-from]   fresh-init modules: {', '.join(heads[:8])}", flush=True)
    return model


class Trainer:

    def _mem_debug(self):
        """POINT_MOE_MEM_DEBUG=K: every K optimizer steps, census LIVE CUDA tensors
        (count + MB grouped by dtype/shape, top 8) and the delta vs the last census.
        Live total FLAT while reserved/shared grows => allocator/WDDM pool growth
        (mitigate: POINT_MOE_EMPTY_CACHE_EVERY=1). Live total RISING => a real leak;
        the fastest-growing shape below is the culprit's fingerprint."""
        import gc as _gc
        import os as _os
        k = int(_os.environ.get("POINT_MOE_MEM_DEBUG", "0") or 0)
        if not k or not torch.cuda.is_available():
            return
        if getattr(self, "_opt_steps_done", 0) % k:
            return
        by = {}
        total = 0
        for o in _gc.get_objects():
            try:
                if torch.is_tensor(o) and o.is_cuda:
                    mb = o.numel() * o.element_size() / 2**20
                    key = (str(o.dtype).replace("torch.", ""), tuple(o.shape))
                    c, m = by.get(key, (0, 0.0))
                    by[key] = (c + 1, m + mb)
                    total += mb
            except Exception:
                continue
        prev = getattr(self, "_mem_debug_prev", None)
        self._mem_debug_prev = total
        top = sorted(by.items(), key=lambda kv: -kv[1][1])[:8]
        d = f" (delta {total - prev:+.0f}MB)" if prev is not None else ""
        print(f"  [mem-debug @opt {getattr(self, '_opt_steps_done', 0)}] live CUDA tensors: "
              f"{total:.0f}MB{d}; alloc={torch.cuda.memory_allocated()/2**30:.2f}G "
              f"resv={torch.cuda.memory_reserved()/2**30:.2f}G", flush=True)
        for (dt, shp), (c, m) in top:
            print(f"      {m:8.0f}MB  x{c:<4d} {dt} {shp}", flush=True)
    def _maybe_empty_cache(self):
        # Windows lacks expandable_segments; long runs fragment the caching allocator.
        # POINT_MOE_EMPTY_CACHE_EVERY=K releases cached blocks every K optimizer steps
        # (0/unset = off). Costs a sync; saves reserved-memory creep on 16 GB cards.
        import os as _os
        k = int(_os.environ.get("POINT_MOE_EMPTY_CACHE_EVERY", "0") or 0)
        self._opt_steps_done = getattr(self, "_opt_steps_done", 0) + 1
        if k and torch.cuda.is_available() and self._opt_steps_done % k == 0:
            torch.cuda.empty_cache()
        self._mem_debug()

    def __init__(self, model, cfg, train_set, val_set, collate, vis_set=None,
                 device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_set = train_set
        self.val_set = val_set
        self.vis_set = vis_set if vis_set is not None else val_set
        self.collate = collate
        # Mix3D fires on TRAINING batches only; validation/test use a zero-mix copy so the
        # per-scene metrics are computed on un-mixed scenes.
        import copy as _copy
        self._val_collate = _copy.copy(collate)
        try:
            self._val_collate.mix_prob = 0.0
        except Exception:
            self._val_collate = collate

        self.out_dir = os.path.join(cfg.out_dir, cfg.name)
        os.makedirs(self.out_dir, exist_ok=True)
        cfg_path = os.path.join(self.out_dir, "config.yaml")
        try:
            cfg.save(cfg_path)
        except Exception:
            pass

        # Loss weighting. Default: unweighted CE -- the MEEPO criteria term
        # (CrossEntropyLoss, loss_weight=1); plain nn.CrossEntropyLoss, mean reduction.
        # Inverse-frequency weighting biases toward the minority class (here it
        # collapses ground), so it is opt-in via loss_class_balance="inverse".
        balance = str(getattr(cfg, "loss_class_balance", "none")).lower()
        if cfg.class_weights is not None:
            weights = torch.tensor(cfg.class_weights, dtype=torch.float32).to(self.device)
            print(f"[loss] explicit class weights {cfg.class_weights}")
        elif balance == "inverse":
            weights = self._estimate_class_weights().to(self.device)
            print(f"[loss] inverse-frequency class weights {weights.tolist()}")
        else:
            weights = None
            print("[loss] cross-entropy, unweighted  [MEEPO criteria: CrossEntropyLoss loss_weight=1, ignore_index=2]")
        lovasz_w = float(getattr(cfg, "lovasz_weight", 1.0)) if bool(getattr(cfg, "loss_lovasz", False)) else 0.0
        self.use_gdreg = bool(getattr(cfg, "use_groundiff_regression", False))
        self.spag_learned = bool(getattr(cfg, "spag_learned", False))
        self.spag_regime_weight = float(getattr(cfg, "spag_regime_weight", 0.1))
        # Co-train the regime head on DTM-RMSE (self-critical REINFORCE) jointly with the
        # backbone, instead of the oracle smooth-L1. Strided + subsampled so the CPU corrector
        # adds little to the hot loop; the head's update rides the main backward/optimizer.
        self.spag_rl = bool(getattr(cfg, "spag_rl", False)) and self.spag_learned
        self.spag_rl_every = max(1, int(getattr(cfg, "spag_rl_every", 10)))
        self.spag_rl_weight = float(getattr(cfg, "spag_rl_weight", 1.0))
        self.spag_rl_sigma = float(getattr(cfg, "spag_rl_sigma", 0.5))
        self.spag_rl_max_points = int(getattr(cfg, "spag_rl_max_points", 30000))  # 0 = use ALL (slow)
        self.spag_rl_res = float(getattr(cfg, "spag_rl_res", getattr(cfg, "dtm_rmse_res", 1.0)))
        self.spag_rl_reward = str(getattr(cfg, "spag_rl_reward", "p95")).lower()  # rmse|p95|p99|max
        self.spag_rl_eval_every = int(getattr(cfg, "spag_rl_eval_every", 200))    # opt-steps; 0 = off
        self.spag_rl_eval_tiles = int(getattr(cfg, "spag_rl_eval_tiles", 12))
        self.spag_rl_eval_max_points = int(getattr(cfg, "spag_rl_eval_max_points", 80000))  # cap per held-out cloud
        self._spag_step = 0
        self._spag_opt_count = 0
        self._spag_last = {}
        self._spag_holdout_idx = None
        self._spag_holdout = []
        self._spag_holdout_last = {}
        self._spag_rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 12345)
        if self.spag_rl:
            print(f"[spag-rl] regime head trained by DTM '{self.spag_rl_reward}' REINFORCE every "
                  f"{self.spag_rl_every} steps (sigma={self.spag_rl_sigma}, weight={self.spag_rl_weight}, "
                  f"max_points={'ALL' if self.spag_rl_max_points <= 0 else self.spag_rl_max_points}, "
                  f"res={self.spag_rl_res}m); backbone detached from RL; oracle smooth-L1 OFF; "
                  f"held-out cliff eval every {self.spag_rl_eval_every} opt-steps on {self.spag_rl_eval_tiles} tiles")
        elif self.spag_learned:
            print(f"[spag] learned regime head via oracle smooth-L1 (weight {self.spag_regime_weight})")
        if self.use_gdreg:
            from .losses import GrounDiffLoss
            l1w = float(getattr(cfg, "groundiff_l1_weight", 1.0))
            l2w = float(getattr(cfg, "groundiff_l2_weight", 1.0))
            clsw = float(getattr(cfg, "groundiff_cls_weight", 1.0))
            scale = float(getattr(cfg, "ndsm_scale", 10.0))
            self.criterion = GrounDiffLoss(
                class_weights=weights, lovasz_weight=lovasz_w,
                l1_weight=l1w, l2_weight=l2w, cls_weight=clsw, ndsm_scale=scale).to(self.device)
            print(f"[loss]  + GrounDiff nDSM regression L1+L2 (lambda1={l1w:g}, lambda2={l2w:g}, "
                  f"cls_weight={clsw:g}, nDSM/{scale:g}m)  [Dhaouadi et al. 2025, Eqs. 11-12]")
            print(f"[loss]    -> dense per-point height target removes the predict-all-ground "
                  f"collapse of a classification-only loss")
        else:
            self.criterion = SegLoss(class_weights=weights, lovasz_weight=lovasz_w)
        if lovasz_w > 0:
            print(f"[loss]  + Lovasz-softmax (weight {lovasz_w:g})  [MEEPO criteria: CE + Lovasz, both weight 1]")
        else:
            print("[loss]  (Lovasz off - SparseGF Table 6: Lovasz raises DTM RMSE)")

        opt_name = str(getattr(cfg, "optimizer", "sgd")).lower()
        sched_name = str(getattr(cfg, "lr_schedule", "onecycle")).lower()
        onecycle_kpx = sched_name in ("onecycle_kpx", "kpx")
        onecycle_pmoe = sched_name in ("onecycle", "1cycle", "pmoe", "onecycle_pmoe")
        self._onecycle_pmoe = False               # set True below; controls per-step LR stepping
        if opt_name == "adamw":
            # KPConvX-L 1-cycle starts at cyc_lr0 and the scheduler ramps to the peak;
            # MEEPO OneCycle / cosine / exp use adamw_lr as the head base (OneCycle's
            # max_lr is taken as [adamw_lr, adamw_lr*block_lr_scale]).
            base_lr = float(getattr(cfg, "kpx_lr_start", 1e-4)) if onecycle_kpx \
                else float(getattr(cfg, "adamw_lr", 2e-3))
            betas = tuple(getattr(cfg, "adamw_betas", (0.9, 0.999)))
            # fused AdamW fuses the per-parameter update into one CUDA kernel
            # (a measurable step-time win on Blackwell); harmless no-op on CPU.
            _fused = self.device.type == "cuda"
            wd = float(getattr(cfg, "adamw_weight_decay", 0.005))
            # PTv3/LitePT "block lr rate": the backbone trains at block_lr_scale x the
            # head/raster-encoder LR (a discriminative LR; Tab.12/13). param groups carry
            # the scale via their own base_lr, and the LR scheduler multiplier applies to
            # all groups proportionally so the 0.1x ratio is preserved throughout.
            blk_scale = float(getattr(cfg, "block_lr_scale", 0.1))
            # MEEPO "block lr" (semseg-meepo.py: param_dicts keyword="block", lr=0.0006 =
            # 0.1x the 0.006 base): only the enc/dec Mamba *blocks* train at blk_scale x the
            # base LR; the embedding, pooling/unpooling, raster encoder and task heads train
            # at the full base LR. The OneCycle multiplier scales all groups proportionally,
            # preserving the ratio throughout.
            blk_kw = str(getattr(cfg, "block_lr_keyword", "block"))
            # Honor `_no_weight_decay` flags (official Mamba-3 sets them on dt_bias
            # and D; decaying dt_bias erodes the VM3 local/global decay bands over
            # long schedules). Split each LR group into decay / no-decay halves;
            # AdamW's per-group weight_decay overrides the constructor default.
            def _nd(p):
                return bool(getattr(p, "_no_weight_decay", False))
            named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            if blk_scale != 1.0:
                spec = [
                    ([p for n, p in named if blk_kw not in n and not _nd(p)], base_lr, wd),
                    ([p for n, p in named if blk_kw not in n and _nd(p)], base_lr, 0.0),
                    ([p for n, p in named if blk_kw in n and not _nd(p)], base_lr * blk_scale, wd),
                    ([p for n, p in named if blk_kw in n and _nd(p)], base_lr * blk_scale, 0.0),
                ]
            else:
                spec = [
                    ([p for n, p in named if not _nd(p)], base_lr, wd),
                    ([p for n, p in named if _nd(p)], base_lr, 0.0),
                ]
            spec = [(ps, lr, w) for ps, lr, w in spec if len(ps) > 0]
            param_groups = [{"params": ps, "lr": lr, "weight_decay": w} for ps, lr, w in spec]
            self._group_lrs = [lr for _, lr, _ in spec]
            _n_nd = sum(len(ps) for ps, _, w in spec if w == 0.0)
            self.optimizer = torch.optim.AdamW(
                param_groups, lr=base_lr, betas=betas,
                eps=float(getattr(cfg, "adamw_eps", 1e-8)),
                weight_decay=wd, fused=_fused,
            )
            print(f"[opt] AdamW lr={base_lr:.1e} betas={betas} wd={wd:.2e}"
                  f"{' fused' if _fused else ''}"
                  f"{f' | {blk_kw!r}-param lr x{blk_scale:g}' if blk_scale != 1.0 else ''}"
                  f" | no-decay params: {_n_nd}")
        else:
            base_lr = float(cfg.learning_rate)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=base_lr,
                momentum=cfg.momentum, weight_decay=cfg.weight_decay,
            )
        if onecycle_kpx:
            # KPConvX-L 1-cycle (train_S3DIS.py cyc_*): exp ramp lr0->lr1 over warmup
            # epochs, hold for plateau epochs, then exp decay (/10 every decay10 epochs).
            lr0 = float(getattr(cfg, "kpx_lr_start", 1e-4))
            lr1 = float(getattr(cfg, "kpx_lr_max", 5e-3))
            warm = int(getattr(cfg, "kpx_lr_warmup_epochs", 30))
            plat = int(getattr(cfg, "kpx_lr_plateau_epochs", 5))
            dec10 = int(getattr(cfg, "kpx_lr_decay10_epochs", 120))
            raise_rate = (lr1 / lr0) ** (1.0 / max(warm, 1))
            decrease_rate = 0.1 ** (1.0 / max(dec10, 1))

            def _kpx_onecycle(epoch, _wr=warm, _pl=plat, _rr=raise_rate, _dr=decrease_rate):
                # multiplier relative to lr0 (the optimizer base lr)
                if epoch <= _wr:
                    return _rr ** epoch
                if epoch <= _wr + _pl:
                    return _rr ** _wr
                return (_rr ** _wr) * (_dr ** (epoch - _wr - _pl))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_kpx_onecycle)
            print(f"[opt] KPConvX-L 1-cycle LR {lr0:.1e} ->(x10 over {warm}ep) {lr1:.1e} "
                  f"| plateau {plat}ep | /10 every {dec10}ep")
        elif onecycle_pmoe and opt_name == "adamw":
            # MEEPO OneCycleLR (configs/point_moe/indoor.py): max_lr=[head, block] peaks,
            # pct_start ramp-up then cosine anneal. torch OneCycleLR steps PER optimiser step,
            # so it needs steps/epoch -> built lazily at the start of train(). max_lr peaks are
            # [adamw_lr, adamw_lr*block_lr_scale] = [0.002, 0.0006], reproducing the reference.
            self.scheduler = None
            self._onecycle_pmoe = True
            blk = float(getattr(cfg, "block_lr_scale", 0.3))
            self._oc_max_lr = getattr(self, "_group_lrs", None) \
                or ([base_lr, base_lr * blk] if blk != 1.0 else base_lr)
            print(f"[opt] MEEPO OneCycleLR max_lr={self._oc_max_lr} "
                  f"pct_start={float(getattr(cfg,'onecycle_pct_start',0.05)):g} "
                  f"div={float(getattr(cfg,'onecycle_div_factor',10.0)):g} "
                  f"final_div={float(getattr(cfg,'onecycle_final_div_factor',1000.0)):g} "
                  f"(per-step; total steps set at train start)")
        elif sched_name == "cosine":
            # PTv3 recipe: linear LR warmup for warmup_epochs, then cosine decay to a
            # small floor over the remaining epochs. LambdaLR multiplier applies equally
            # to every param group (so block_lr_scale's 0.1x ratio is preserved).
            import math as _math
            total = max(int(cfg.epochs), 1)
            warm = max(int(getattr(cfg, "warmup_epochs", 0)), 0)
            floor = 1e-4                                  # eta_min / base_lr

            def _cos_warmup(epoch, _w=warm, _T=total, _f=floor):
                if _w > 0 and epoch < _w:
                    return (epoch + 1) / _w               # linear warmup 0 -> 1
                t = (epoch - _w) / max(1, _T - _w)
                return _f + (1.0 - _f) * 0.5 * (1.0 + _math.cos(_math.pi * min(t, 1.0)))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_cos_warmup)
            print(f"[opt] cosine LR {base_lr:.1e} -> {base_lr*floor:.1e} over {total} epochs"
                  f"{f' (warmup {warm}ep)' if warm > 0 else ''}")
        else:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.lr_decay)
        self.use_amp = bool(cfg.amp) and self.device.type == "cuda"
        # bfloat16 on CUDA has the same exponent range as fp32, so geometric
        # features / activations cannot overflow the way fp16 (max ~65504) does
        # -> avoids mid-epoch NaNs. bf16 needs no loss scaling, so the GradScaler
        # is enabled only for fp16. (AMP dtype is an implementation detail; the
        # paper specifies no mixed precision.)
        amp_name = str(getattr(cfg, "amp_dtype", "bf16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_name in ("bf16", "bfloat16") else torch.float16
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.use_amp and self.amp_dtype == torch.float16)

        self.history: List[Dict] = []
        self.log_path = os.path.join(self.out_dir, "train_log.jsonl")

    # ----------------------------------------------------------------- helpers
    def _estimate_class_weights(self) -> torch.Tensor:
        counts = np.zeros(self.cfg.num_classes, dtype=np.float64)
        n = min(len(self.train_set), 64)
        for i in range(n):
            labs = self.train_set[i]["labels"]
            for c in range(self.cfg.num_classes):
                counts[c] += int(np.sum(labs == c))
        if counts.sum() == 0:
            return torch.ones(self.cfg.num_classes)
        return inverse_frequency_weights(counts, self.cfg.num_classes)

    def _data_diagnostics(self, nsamp):
        """Startup sanity printed before the first step: label balance per split, the
        epoch-repetition factor, and feature-channel means. Catches the two silent
        killers -- a label convention that IGNOREs most points (loss never sees them ->
        model degenerates to all-ground), and a fixed-step epoch recycling a tiny corpus
        at full LR (memorisation) -- BEFORE ~100 GPU-hours get spent. Never raises."""
        import glob
        import json
        import numpy as np
        try:
            n_tr = len(self.train_set)
            td = getattr(self.train_set, "tile_dir", None) or getattr(self.cfg, "tile_dir", None)
            if td and os.path.isdir(td):
                fr = {}
                for split in ("train", "val"):
                    cnt = np.zeros(3, dtype=np.int64)
                    used = 0
                    for p in sorted(glob.glob(os.path.join(td, "*.npz"))):
                        if used >= 16:
                            break
                        try:
                            with np.load(p, allow_pickle=True) as d:
                                s = str(d["split"]) if "split" in d.files else "train"
                            if s != split:
                                continue
                            lab = np.load(p[:-4] + ".labels.npy", mmap_mode="r")
                            step = max(1, lab.shape[0] // 200000)
                            cnt += np.bincount(np.asarray(lab[::step]).ravel().astype(np.int64),
                                               minlength=3)[:3]
                            used += 1
                        except Exception:
                            continue
                    tot = max(int(cnt.sum()), 1)
                    fr[split] = (cnt[1] / tot, cnt[0] / tot, cnt[2] / tot, used)
                for split, (g, ng, ig, used) in fr.items():
                    print(f"[diag] {split} labels ({used} tiles sampled): ground {100*g:.1f}%  "
                          f"non-ground {100*ng:.1f}%  IGNORE {100*ig:.1f}%", flush=True)
                g, ng, ig, _u = fr.get("train", (0.0, 0.0, 0.0, 0))
                if ig > 0.30:
                    print(f"[diag] *** WARNING: {100*ig:.0f}% of train points are IGNORE -- excluded "
                          f"from the loss AND the metrics. If this dataset marks non-ground as ASPRS "
                          f"class 1 (common in British EA products: only ground is classified), the "
                          f"loss supervises ~only ground and the model degenerates to predicting "
                          f"ground everywhere. Fix: re-run stage 04 with --unclassified-classes 0 "
                          f"(labels are baked into tiles). ***", flush=True)
                elif ng < 0.05:
                    print(f"[diag] *** WARNING: only {100*ng:.1f}% of train points supervise as "
                          f"NON-GROUND -- expect collapse toward all-ground predictions. ***",
                          flush=True)
            if nsamp > 0 and n_tr > 0:
                rep = nsamp / float(n_tr)
                if rep > 8:
                    print(f"[diag] *** WARNING: fixed-step epoch draws {nsamp} samples from only "
                          f"{n_tr} train tiles = ~{rep:.0f} views/tile PER EPOCH at full LR -- a "
                          f"memorisation regime (epoch size was tuned for a much larger corpus). "
                          f"For small corpora use --epoch-steps 0 (= one full pass per epoch) with "
                          f"more --epochs; wall-time per epoch shrinks proportionally. ***",
                          flush=True)
                else:
                    print(f"[diag] epoch draws {nsamp} samples over {n_tr} train tiles "
                          f"(~{rep:.1f} views/tile)", flush=True)
            if td:
                ns = os.path.join(td, "norm_stats.json")
                if os.path.exists(ns):
                    with open(ns) as fh:
                        st = json.load(fh)
                    mean = st.get("mean")
                    if mean:
                        mm = ", ".join(f"{float(m):+.2f}" for m in mean)
                        print(f"[diag] feature channel means: [{mm}]", flush=True)
                        big = [i for i, m in enumerate(mean) if abs(float(m)) > 10.0]
                        if big:
                            print(f"[diag] *** WARNING: channel(s) {big} have |mean| > 10 -- if one "
                                  f"of these is the z-minus-previous-DTM channel, the prior DTM and "
                                  f"the LAS heights disagree (vertical datum / units) and the prior "
                                  f"misleads rather than helps. Check the prior rasters' datum. ***",
                                  flush=True)
        except Exception as e:
            print(f"[diag] (startup diagnostics skipped: {type(e).__name__}: {e})", flush=True)

    def _loaders(self):
        import numpy as np
        from torch.utils.data import RandomSampler, SubsetRandomSampler
        from ..data.dataset import make_region_balanced_sampler
        bs = int(self.cfg.batch_num)
        accum = max(1, int(getattr(self.cfg, "grad_accum_steps", 1)))
        # nsamp is in MICRO-batches: epoch_steps optimiser steps x accum micro-batches each.
        nsamp = int(getattr(self.cfg, "epoch_steps", 0)) * bs * accum  # KPConv fixed-step epoch
        # Sphere sampling is UNIFORM random by default -> draw candidate spheres
        # (with replacement to fill the fixed-step epoch). Only build the weighted skew
        # sampler when explicitly enabled.
        if bool(getattr(self.cfg, "use_region_balanced_sampler", False)):
            try:
                sampler = make_region_balanced_sampler(self.train_set, self.cfg)  # draws nsamp
            except Exception:
                sampler = (RandomSampler(self.train_set, replacement=True, num_samples=nsamp)
                           if nsamp > 0 else None)
        else:
            sampler = (RandomSampler(self.train_set, replacement=True, num_samples=nsamp)
                       if nsamp > 0 else None)
        _rb = bool(getattr(self.cfg, "use_region_balanced_sampler", False))
        print(f"[setup] sphere sampling: {'REGION-BALANCED (equal mass per source cloud)' if _rb else 'UNIFORM random over all spheres'}",
              flush=True)
        nw = int(getattr(self.cfg, "num_workers", 0))
        NW_CAP = 24
        if nw > NW_CAP:
            print(f"[setup] num_workers={nw} -> capped to {NW_CAP}. Beyond this the large "
                  f"multiscale batches (with their neighbour-index arrays) shipped over IPC, "
                  f"plus single-threaded pin_memory, make it slower AND the prefetch buffer "
                  f"can exhaust RAM (an OOM kill). Pass --num-workers <= {NW_CAP}.", flush=True)
            nw = NW_CAP
        # POINT_MOE_PIN_MEMORY=0 disables DataLoader pinning. At ~512k-point scenes
        # each prefetched multiscale batch is GB-scale; pinned staging is GPU-mapped
        # under WDDM and can fill the ENTIRE shared-GPU budget (observed 30/31.6 GB),
        # thrashing the machine before step 1. Unpinned = pageable H2D (slower copy,
        # but copies are a tiny fraction of these step times).
        import os as _os
        pin = self.device.type == "cuda" and _os.environ.get("POINT_MOE_PIN_MEMORY", "1") != "0"
        common = dict(collate_fn=self.collate, num_workers=nw, pin_memory=pin)
        if nw > 0:
            # persistent_workers OFF by default: workers are torn down each epoch, which
            # RESETS the gradual copy-on-write RAM creep of the forked workers so it can't
            # compound into an OOM over a long run. Re-fork cost is a couple of seconds
            # per epoch (the dataset is already built), negligible next to epoch time.
            common.update(
                persistent_workers=bool(getattr(self.cfg, "dataloader_persistent", False)),
                prefetch_factor=int(getattr(self.cfg, "dataloader_prefetch", 2)),
            )

        # ---- KPConv variable batch size (total-points budget; paper-faithful) ----
        if False:  # variable batching removed with sphere mode
            return self._variable_loaders(bs, nsamp, sampler, common)

        train_loader = DataLoader(
            self.train_set, batch_size=bs,
            sampler=sampler, shuffle=sampler is None, drop_last=True, **common,
        )

        # bound validation to validation_size steps (a fixed seeded subset) so it
        # doesn't dominate the now-shorter epoch; full pass if validation_size<=0
        vsteps = int(getattr(self.cfg, "validation_size", 0))
        n_val = len(self.val_set)
        common_val = dict(common); common_val["collate_fn"] = self._val_collate   # Mix3D off for val
        if vsteps > 0 and n_val > vsteps * bs:
            idx = np.random.default_rng(int(getattr(self.cfg, "seed", 0))).choice(
                n_val, size=vsteps * bs, replace=False)
            val_loader = DataLoader(self.val_set, batch_size=bs,
                                    sampler=SubsetRandomSampler(idx.tolist()), **common_val)
        else:
            val_loader = DataLoader(self.val_set, batch_size=bs, shuffle=False, **common_val)

        if nw > 0:
            print(f"[setup] DataLoader workers={nw} (parallel multiscale-batch build), "
                  f"pin_memory={pin}")
        self._data_diagnostics(nsamp)
        steps = nsamp // bs if nsamp > 0 else (len(self.train_set) // bs)
        vshown = min(vsteps, n_val // bs) if vsteps > 0 else (n_val // bs)
        kind = "KPConv fixed-step epoch" if nsamp > 0 else "full pass over all tiles"
        if accum > 1:
            osteps = max(steps // accum, 1)
            print(f"[setup] epoch = {osteps} optimiser steps x effective batch {bs * accum} "
                  f"({bs} blocks/forward x {accum} grad-accum) "
                  f"({kind}); validation = {vshown} steps")
        else:
            print(f"[setup] epoch = {steps} steps x {bs} = {steps*bs} train samples "
                  f"({kind}); validation = {vshown} steps")
        return train_loader, val_loader

    def _variable_loaders(self, bs, nsamp, sampler, common):
        """KPConv-style variable batching: pack spheres into a total-points budget
        (calibrated so a batch averages ``bs`` spheres) instead of a fixed count, so
        the per-batch point count - hence VRAM and step time - stays ~constant as
        point density varies. ``sampler`` is the skew/weighted index sampler (or None)."""
        import numpy as np
        from torch.utils.data import RandomSampler, SubsetRandomSampler, WeightedRandomSampler
        from ..data.dataset import calibrate_batch_limit, PointBudgetBatchSampler

        if sampler is not None:
            base = sampler
        else:
            base = (RandomSampler(self.train_set, replacement=True, num_samples=nsamp)
                    if nsamp > 0 else RandomSampler(self.train_set))
        try:
            n_idx = len(base)
        except TypeError:
            n_idx = len(self.train_set)
        weights = (base.weights.cpu().numpy()
                   if isinstance(base, WeightedRandomSampler) else None)
        blim = int(getattr(self.cfg, "batch_limit", 0)) or \
            calibrate_batch_limit(self.train_set, self.cfg, weights)
        self.batch_limit = int(blim)
        n_batches = max(1, round(n_idx / max(bs, 1)))
        train_bs = PointBudgetBatchSampler(
            base, self.train_set.candidate_point_counts(), blim, num_batches=n_batches)
        train_loader = DataLoader(self.train_set, batch_sampler=train_bs, **common)

        vsteps = int(getattr(self.cfg, "validation_size", 0))
        n_val = len(self.val_set)
        vcounts = self.val_set.candidate_point_counts()
        if vsteps > 0 and n_val > vsteps * bs:
            vidx = np.random.default_rng(int(getattr(self.cfg, "seed", 0))).choice(
                n_val, size=min(vsteps * bs, n_val), replace=False).tolist()
            vbase, vnb = SubsetRandomSampler(vidx), vsteps
        else:
            vbase, vnb = list(range(n_val)), max(1, round(n_val / max(bs, 1)))
        val_bs = PointBudgetBatchSampler(vbase, vcounts, blim, num_batches=vnb)
        common_val = dict(common); common_val["collate_fn"] = self._val_collate   # Mix3D off for val
        val_loader = DataLoader(self.val_set, batch_sampler=val_bs, **common_val)

        print(f"[setup] variable batch (KPConv): batch_limit={blim:,} pts/batch "
              f"(~{bs} spheres/batch avg) | epoch={len(train_bs)} steps; "
              f"validation={len(val_bs)} steps")
        return train_loader, val_loader

    # ------------------------------------------------------- spag-rl held-out eval
    def _spag_holdout_select(self):
        """Pick a FIXED set of the most cliff-heavy val tiles ONCE (by plane-detrended
        GT-ground P95 |residual|: high = steep local relief = cliffs), reused every epoch
        as a stable instrument for the regime head. Pure-numpy, bounded sample for cost."""
        if self._spag_holdout_idx is not None:
            return
        n = len(self.val_set)
        cand = (sorted(set(int(i) for i in np.linspace(0, n - 1, 96).round().astype(int)))
                if n > 96 else list(range(n)))
        scores = []
        for i in cand:
            try:
                c = self.val_set._load(int(i))
                pts = np.asarray(c["local"], dtype=np.float64)
                lab = np.asarray(c["labels"]).reshape(-1)
                g = lab == 1
                if int(g.sum()) < 64:
                    continue
                P = pts[g]
                A = np.column_stack([P[:, 0], P[:, 1], np.ones(P.shape[0])])
                coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
                resid = np.abs(P[:, 2] - A.dot(coef))
                scores.append((float(np.percentile(resid, 95)), int(i)))
            except Exception:
                continue
        scores.sort(reverse=True)
        k = max(1, int(self.spag_rl_eval_tiles))
        self._spag_holdout_idx = [i for _, i in scores[:k]]
        if self._spag_holdout_idx:
            sc = [round(s, 2) for s, _ in scores[:k]]
            print(f"[spag-rl] held-out cliff set: {len(self._spag_holdout_idx)} val tiles "
                  f"(plane-detrended P95 residual {min(sc)}-{max(sc)} m)", flush=True)

    def _spag_holdout_build(self):
        """Cache the held-out scenes from the CURRENT model (fresh predictions): one
        point-budget block forward per tile -> (pooled, pred_stats, xyz, raw_pred, gt).
        Refreshed each epoch so predictions track the backbone; the head is then varied
        between rebuilds, isolating head quality."""
        self._spag_holdout_select()
        if not self._spag_holdout_idx:
            self._spag_holdout = []
            return
        was_training = self.model.training
        self.model.eval()
        cache = []
        with torch.no_grad():
            for i in self._spag_holdout_idx:
                try:
                    batch = move_batch(self._val_collate([self.val_set[int(i)]]), self.device, self.cfg)
                    logits = self.model(batch)
                    if getattr(self.model, "_regime_pooled", None) is None:
                        continue
                    pooled = self.model._regime_pooled.detach().cpu()
                    pstats = self.model._regime_pred_stats.detach().cpu()
                    pred = logits.argmax(dim=1).cpu().numpy().reshape(-1)
                    coord = batch["coord"].detach().cpu().numpy()
                    labels = batch["labels"].detach().cpu().numpy().reshape(-1)
                    offs = batch["offset"].detach().cpu().numpy().reshape(-1)
                    c0 = batch.get("cloud_lengths_0", None)
                    c0 = (c0.detach().cpu().numpy().reshape(-1) if torch.is_tensor(c0)
                          else (np.asarray(c0).reshape(-1) if c0 is not None
                                else np.diff(np.concatenate([[0], offs]))))
                    starts = np.concatenate([[0], np.cumsum(c0)[:-1]]).astype(np.int64)
                    cap = int(self.spag_rl_eval_max_points)
                    for b in range(pooled.shape[0]):
                        s = int(starts[b]); e = s + int(c0[b])
                        if e - s < 64:
                            continue
                        sel = np.arange(s, e)
                        if cap > 0 and sel.size > cap:                      # 1 m DEM is resolution-limited;
                            sel = self._spag_rng.choice(sel, size=cap, replace=False)  # subsample is accuracy-safe
                        cache.append({"pooled": pooled[b:b + 1], "pred_stats": pstats[b:b + 1],
                                      "xyz": coord[sel].astype(np.float64),
                                      "pred": pred[sel].astype(np.int64),
                                      "gt": labels[sel].astype(np.int64)})
                except Exception:
                    continue
        if was_training:
            self.model.train()
        self._spag_holdout = cache

    def _spag_holdout_eval(self):
        """Greedy-head DTM-RMSE on the cached held-out cliff set vs no-refine. Cheap: only
        the (norm-free) head forward + the corrector on cached scenes. The greedy params are
        what inference deploys, so this is the real 'is the head improving?' signal."""
        cache = self._spag_holdout
        if not cache:
            return None
        from ..inference.spag_rl import _scene_errs, _agg, _refine_errs
        was_training = self.model.training
        self.model.eval()
        # greedy (deterministic) globals per scene on the main thread (torch), then the
        # corrector + DEM-RMSE per tile. Held-out clouds are subsampled (eval_max_points) so
        # this stays cheap even on the extreme-relief tiles.
        jobs, en_list = [], []
        with torch.no_grad():
            for sc in cache:
                pooled = sc["pooled"].to(self.device); pstats = sc["pred_stats"].to(self.device)
                g = self.model.squash_globals(
                    self.model.regime_logits(pooled, pstats)).cpu().numpy().reshape(-1)
                jobs.append((sc["xyz"], sc["pred"], sc["gt"], self.cfg, g, self.spag_rl_res))
                en_list.append(_scene_errs(sc["xyz"], sc["pred"], sc["gt"], self.spag_rl_res))  # no-refine
        out = [_refine_errs(j) for j in jobs]
        if was_training:
            self.model.train()
        rg, rn, rec, mg = [], [], [], []
        for (eg, info), en in zip(out, en_list):
            if eg.size:
                rg.append(_agg(eg, "rmse")); mg.append(_agg(eg, self.spag_rl_reward))
            if en.size:
                rn.append(_agg(en, "rmse"))
            ng = max(int(info.get("n_ground", 0)), 1)
            rec.append(int(info.get("n_reclassified", 0)) / ng)
        if not rg:
            return None
        return {"rmse_greedy": float(np.mean(rg)),
                "rmse_noref": (float(np.mean(rn)) if rn else float("nan")),
                "score_greedy": (float(np.mean(mg)) if mg else float("nan")),
                "reclass": float(np.mean(rec))}

    # ------------------------------------------------------------------- train
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # drop eval-shape pool blocks (staircase fix)

    def train(self, epochs: Optional[int] = None):
        epochs = epochs or self.cfg.epochs
        train_loader, val_loader = self._loaders()
        n_iter = max(len(train_loader), 1)
        # MEEPO OneCycleLR steps per OPTIMISER step -> build it now that steps/epoch is known.
        self._oc_step = 0
        if getattr(self, "_onecycle_pmoe", False) and self.scheduler is None:
            import math as _math
            accum = max(1, int(getattr(self.cfg, "grad_accum_steps", 1)))
            steps_per_epoch = max(1, _math.ceil(n_iter / accum))
            total_steps = epochs * steps_per_epoch
            self._oc_total = total_steps
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self._oc_max_lr,
                total_steps=total_steps + 4,                       # small slack: never over-step
                pct_start=float(getattr(self.cfg, "onecycle_pct_start", 0.05)),
                anneal_strategy="cos",
                div_factor=float(getattr(self.cfg, "onecycle_div_factor", 10.0)),
                final_div_factor=float(getattr(self.cfg, "onecycle_final_div_factor", 1000.0)),
            )
            print(f"[opt] OneCycleLR built: {steps_per_epoch} opt-steps/epoch x {epochs} ep "
                  f"= {total_steps} steps (warmup {int(round(0.05*total_steps))} steps)")
        global_start = time.time()
        self.best_miou = -1.0
        self.best_epoch = 0
        print(f"[setup] device={self.device} amp={self.use_amp} "
              f"params={self.model.num_parameters():,} "
              f"train_tiles={len(self.train_set)} val_tiles={len(self.val_set)}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            if self.spag_rl and self.spag_rl_eval_every > 0:
                self._spag_holdout_build()          # refresh held-out cliff scenes with current predictions
            ep_start = time.time()
            running = 0.0
            seen_pts = 0
            nonfinite = 0
            conf = ConfusionAccumulator()

            accum = max(1, int(getattr(self.cfg, "grad_accum_steps", 1)))
            self.optimizer.zero_grad(set_to_none=True)
            win_bwd = False            # did any micro-batch in this accumulation window backward?
            for it, batch in enumerate(train_loader):
                batch = move_batch(batch, self.device, self.cfg)
                labels = batch["labels"]
                win_end = ((it + 1) % accum == 0) or (it == n_iter - 1)

                with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    import contextlib as _ctx
                    import os as _os
                    _off = (torch.autograd.graph.save_on_cpu(
                                pin_memory=_os.environ.get("POINT_MOE_OFFLOAD_PIN", "1") != "0")
                            if getattr(self.cfg, "offload_activations", False)
                            and torch.cuda.is_available() else _ctx.nullcontext())
                    with _off:
                        logits = self.model(batch)
                    if getattr(self, "use_gdreg", False):
                        # GrounDiff: aux = continuous nDSM regression (pred, target)
                        aux_pred = getattr(self.model, "_reg_pred", None)
                        aux_tgt = batch.get("ndsm")
                    else:
                        aux_pred = None     # plain SegLoss (CE + Lovasz); no auxiliary
                        aux_tgt = None
                    loss = self.criterion(logits, labels, aux_pred, aux_tgt)
                    if self.spag_learned and self.spag_rl:
                        # LEARNED SPAG-DC (RMSE objective): strided self-critical REINFORCE on
                        # the regime head, scored by the REAL corrector's DTM-RMSE-vs-GT. The
                        # head's gradient rides this micro-batch's backward + the main optimizer.
                        self._spag_step += 1
                        if (self._spag_step % self.spag_rl_every) == 0:
                            from ..inference.spag_rl import reinforce_loss_term
                            c0 = batch.get("cloud_lengths_0", None)
                            if c0 is None:
                                _off = batch["offset"]
                                c0 = torch.diff(torch.cat([_off.new_zeros(1), _off]))
                            rl_loss, rlm = reinforce_loss_term(
                                self.model, batch["coord"], logits.argmax(dim=1), labels, c0,
                                self.cfg, sigma=self.spag_rl_sigma, res=self.spag_rl_res,
                                max_points=self.spag_rl_max_points, rng=self._spag_rng,
                                metric=self.spag_rl_reward)
                            if rl_loss is not None and torch.isfinite(rl_loss):
                                loss = loss + self.spag_rl_weight * rl_loss.to(loss.dtype)
                                self._spag_last = rlm
                    elif self.spag_learned:
                        # LEARNED SPAG-DC: per-scene smooth-L1 on the regime globals,
                        # normalised by each global's [lo,hi] span (train-only auxiliary).
                        rp = getattr(self.model, "_regime_pred", None)
                        rt = batch.get("regime")
                        if rp is not None and rt is not None:
                            span = (self.model._spag_hi - self.model._spag_lo).to(rp.dtype)
                            rloss = torch.nn.functional.smooth_l1_loss(
                                rp / span, rt.to(rp.device, rp.dtype) / span)
                            if torch.isfinite(rloss):
                                loss = loss + self.spag_regime_weight * rloss

                # defensive: never let a single non-finite micro-batch corrupt the weights; skip
                # its backward but still close the accumulation window normally. loss/accum makes
                # the accumulated gradient the MEAN over the window (= the paper's cross-GPU mean).
                if torch.isfinite(loss):
                    self.scaler.scale(loss / accum).backward()
                    win_bwd = True
                    running += float(loss.item())
                    seen_pts += int(labels.shape[0])
                    with torch.no_grad():
                        pred = logits.argmax(dim=1)
                        conf.update(pred.cpu().numpy(), labels.cpu().numpy())
                else:
                    nonfinite += 1

                if win_end:
                    if win_bwd:
                        if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            total_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.cfg.grad_clip_norm)
                            if torch.isfinite(total_norm):
                                self.scaler.step(self.optimizer)
                            else:
                                nonfinite += 1            # non-finite grads -> skip the step
                            self.scaler.update()
                            self._maybe_empty_cache()
                        else:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self._maybe_empty_cache()
                    if win_bwd and getattr(self, "_onecycle_pmoe", False) and \
                       self.scheduler is not None and self._oc_step < self._oc_total:
                        self.scheduler.step()             # OneCycle steps per optimiser step
                        self._oc_step += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    win_bwd = False
                    if self.spag_rl and self.spag_rl_eval_every > 0:
                        self._spag_opt_count += 1
                        if (self._spag_opt_count % self.spag_rl_eval_every) == 0:
                            _he = self._spag_holdout_eval()
                            if _he is not None:
                                self._spag_holdout_last = _he
                                print(f"  [spag-rl eval @opt {self._spag_opt_count}] cliff-set "
                                      f"greedy_RMSE={_he['rmse_greedy']:.3f}m  no-refine={_he['rmse_noref']:.3f}m  "
                                      f"(d{_he['rmse_greedy'] - _he['rmse_noref']:+.3f})  "
                                      f"{self.spag_rl_reward}={_he['score_greedy']:.3f}m  "
                                      f"reclass={_he['reclass'] * 100:.0f}%", flush=True)

                _le = int(getattr(self.cfg, "log_every_steps", 0) or 0)
                _due = (it % _le == 0) if _le > 0 else (it % max(n_iter // 10, 1) == 0)
                if _due or (it == n_iter - 1):
                    elapsed = time.time() - ep_start
                    pps = seen_pts / max(elapsed, 1e-6)
                    frac = (it + 1) / n_iter
                    eta_ep = elapsed / max(frac, 1e-6) * (1 - frac)
                    _ostep = f"[opt {(it + 1) // accum}/{max(n_iter // accum, 1)}]" if accum > 1 else ""
                    _rl = ""
                    if self.spag_rl and self._spag_last.get("n", 0):
                        _sl = self._spag_last
                        _rl = (f" | spag-rl[{_sl.get('metric', 'rmse')}] g={_sl['score_base']:.2f} "
                               f"s={_sl['score_sample']:.2f} adv={_sl['advantage']:+.3f} "
                               f"reclass={_sl['reclass_frac']*100:.0f}% rmse={_sl['rmse_base']:.2f}m")
                    _mem = (f" mem[a={torch.cuda.memory_allocated()/2**30:.1f} "
                            f"r={torch.cuda.memory_reserved()/2**30:.1f} "
                            f"pk={torch.cuda.max_memory_allocated()/2**30:.1f}G]"
                            if torch.cuda.is_available() else "")
                    print(f"  e{epoch:03d} [{it + 1:4d}/{n_iter}]{_ostep} "
                          f"loss={running / max(it + 1 - nonfinite, 1):.4f} "
                          f"lr={self.optimizer.param_groups[0]['lr']:.2e} "
                          f"{pps / 1e3:.1f}k pts/s "
                          f"elapsed={_fmt_time(elapsed)} eta={_fmt_time(eta_ep)}{_mem}{_rl}",
                          flush=True)

            if not getattr(self, "_onecycle_pmoe", False):
                self.scheduler.step()                     # per-epoch schedulers (cosine/exp/kpx)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()          # epoch boundary: drop inference-shape pool blocks
            train_metrics = conf.compute()
            train_loss = running / max(n_iter - nonfinite, 1)
            if nonfinite:
                print(f"  [warn] {nonfinite}/{n_iter} batches had non-finite loss and "
                      f"were skipped this epoch.", flush=True)

            # ---- validation ----
            val_loss, val_metrics = self.evaluate(
                val_loader, max_tiles=int(getattr(self.cfg, "scene_val_tiles", 32)))

            ep_time = time.time() - ep_start
            done = epoch
            avg_ep = (time.time() - global_start) / done
            eta_all = avg_ep * (epochs - done)

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time_s": ep_time,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{k: val_metrics[k] for k in ("IoU1", "IoU2", "OA", "Kappa", "MCC", "mIoU")},
                "RMSE": val_metrics.get("RMSE", float("nan")),
                "IoU1_raw": val_metrics.get("IoU1_raw", float("nan")),
                "IoU2_raw": val_metrics.get("IoU2_raw", float("nan")),
                "OA_raw": val_metrics.get("OA_raw", float("nan")),
                "mIoU_raw": val_metrics.get("mIoU_raw", float("nan")),
                "RMSE_raw": val_metrics.get("RMSE_raw", float("nan")),
                "n_reclassified": val_metrics.get("n_reclassified", 0),
                "reclassified_frac": val_metrics.get("reclassified_frac", 0.0),
                "train_OA": train_metrics["OA"],
            }
            self.history.append(record)
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")

            print(f"[epoch {epoch:03d}/{epochs}] "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                  f"IoU1={val_metrics['IoU1']:.2f} IoU2={val_metrics['IoU2']:.2f} "
                  f"OA={val_metrics['OA']:.2f} Kappa={val_metrics['Kappa']:.2f} "
                  f"MCC={val_metrics['MCC']:.2f} | "
                  f"mIoU={val_metrics['mIoU']:.2f} RMSE={val_metrics.get('RMSE', float('nan')):.3f}m | "
                  f"time={_fmt_time(ep_time)} ETA(all)={_fmt_time(eta_all)}",
                  flush=True)
            # SPAG-DC before/after: show the refinement's impact on the deployed metrics.
            if str(getattr(self.cfg, "refine_method", "spag_dc")).lower() not in ("off", "none", ""):
                _i1r = val_metrics.get("IoU1_raw", float("nan"))
                _rmr = val_metrics.get("RMSE_raw", float("nan"))
                _nrc = val_metrics.get("n_reclassified", 0)
                _frc = 100.0 * val_metrics.get("reclassified_frac", 0.0)
                print(f"           SPAG-DC refine: IoU1 {_i1r:.2f} -> {val_metrics['IoU1']:.2f} "
                      f"(d{val_metrics['IoU1'] - _i1r:+.2f}) | "
                      f"RMSE {_rmr:.3f} -> {val_metrics.get('RMSE', float('nan')):.3f}m "
                      f"(d{val_metrics.get('RMSE', float('nan')) - _rmr:+.3f}) | "
                      f"reclassified {_nrc} spikes ({_frc:.2f}% of ground)", flush=True)

            # ---- per-epoch artefacts ----
            ep_dir = os.path.join(self.out_dir, f"epoch_{epoch:03d}")
            os.makedirs(ep_dir, exist_ok=True)
            if self.cfg.save_checkpoint_every_epoch:
                self._save_checkpoint(os.path.join(ep_dir, "model.pt"), epoch, val_metrics)
            # keep the best model by validation mIoU (validation metrics can wobble
            # epoch-to-epoch early on, so the last epoch is not always the best)
            cur = float(val_metrics.get("mIoU", float("nan")))
            if np.isfinite(cur) and cur > self.best_miou:
                self.best_miou = cur
                self.best_epoch = epoch
                self._save_checkpoint(os.path.join(self.out_dir, "model_best.pt"),
                                      epoch, val_metrics)
                print(f"  [best] new best val mIoU={cur:.2f} (epoch {epoch}) "
                      f"-> model_best.pt", flush=True)
            update_training_charts(self.history,
                                   os.path.join(self.out_dir, "training_dashboard.png"),
                                   eta_text=f"ETA (all): {_fmt_time(eta_all)}")
            if str(getattr(self.cfg, "refine_method", "spag_dc")).lower() not in ("off", "none", ""):
                try:
                    update_refine_charts(self.history, self.out_dir)
                except Exception as _e:
                    print(f"  [warn] refine chart skipped: {type(_e).__name__}: {_e}", flush=True)
            if self.cfg.render_errors_every_epoch or self.cfg.write_laz_every_epoch:
                self._per_epoch_vis(ep_dir, epoch)

        print(f"[done] total time {_fmt_time(time.time() - global_start)} | "
              f"best val mIoU={self.best_miou:.2f} at epoch {self.best_epoch} "
              f"(model_best.pt)")
        self._save_checkpoint(os.path.join(self.out_dir, "model_final.pt"), epochs, val_metrics)

    # ---------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(self, val_loader, max_tiles=None):
        # Scene mode: measure the val metric with the SAME full-resolution fragment
        # inference deployed at test time and shown in the per-epoch viz (predict_scene),
        # then accumulate the metric on FULL-RES points. This mirrors Pointcept's
        # SemSegTester (GridSample mode="test" fragments -> softmax-merged to full res)
        # and matches LitePT / MEEPO evaluation. The legacy path below instead scored
        # ONE random point-budget crop at voxel resolution -- faithful to neither the
        # reference val (whole grid-sampled scene, no crop) nor the reference test
        # (fragment voting) -- which made the aggregate metric collapse to the majority
        # class even though per-scene predict_scene inference was 93-99% OA. The scene
        # branch falls back to the single-forward path on any error so it can never
        # silently kill a run.
        if bool(getattr(self.cfg, "scene_mode", False)):
            try:
                return self._evaluate_scene(max_tiles=max_tiles)
            except Exception as e:
                print(f"  [warn] scene-faithful eval failed ({type(e).__name__}: {e}); "
                      f"falling back to single-forward val")
        self.model.eval()
        conf = ConfusionAccumulator()
        rmse_acc = RMSEAccumulator(res=float(getattr(self.cfg, "dtm_rmse_res", 1.0)))
        total_loss, n = 0.0, 0
        for batch in val_loader:
            batch = move_batch(batch, self.device, self.cfg)
            labels = batch["labels"]
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(batch)
                # Report loss on clamped logits. BatchNorm in eval() mode uses running
                # stats; before they converge it can rescale logits to huge magnitudes
                # (argmax/mIoU unaffected -> metrics stay valid, but raw CE explodes).
                # Clamping to a softmax-saturating range makes val_loss a stable, epoch-
                # comparable number. Does NOT touch training loss or predictions.
                loss = self.criterion(logits.clamp(-30.0, 30.0), labels)
            if torch.isfinite(loss):
                total_loss += float(loss.item()); n += 1
            pred = logits.argmax(dim=1)
            pred_np = pred.cpu().numpy(); lab_np = labels.cpu().numpy()
            conf.update(pred_np, lab_np)
            # ---- DTM-RMSE per cloud (OpenGF/SparseGF surface metric) ----
            try:
                pts = batch["points"][0].detach().cpu().numpy()       # per-voxel centred xyz
                lens = batch["cloud_lengths_0"].detach().cpu().numpy()
                o = 0
                for L in lens:
                    L = int(L)
                    if L > 0 and (o + L) <= pts.shape[0]:
                        sl = slice(o, o + L)
                        rmse_acc.update(pts[sl], pred_np[sl], lab_np[sl])
                    o += L
            except Exception:
                pass
        self.model.train()
        metrics = conf.compute()
        metrics["RMSE"] = rmse_acc.compute()
        return (total_loss / max(n, 1)), metrics

    @torch.no_grad()
    def _evaluate_scene(self, max_tiles=None):
        """Validation for whole-scene mode, following KPConv's large-scene recipe.

        KPConv never feeds a whole cloud to the network: the input is a fixed-size region
        (an ``in_radius`` sphere -- S3DIS uses 2 m), and per-epoch validation scores a
        BOUNDED number of such regions (``validation_size``), NOT the whole scene -- full
        coverage is reserved for the final test. We mirror that. Per epoch we score one
        point-budget region per sampled tile -- the ``scene_max_points`` points nearest
        the tile centre, at full density: IDENTICAL to the training eval crop and to one
        predict_scene block (Pointcept SphereCrop(point_max)). It is scored via the
        deployed full-res inference (predict_scene: softmax-merge blocks -> argmax -> NN-
        project) and accumulated on FULL-RESOLUTION points. ``max_tiles`` caps how many
        tiles are scored (fixed evenly-spaced subset -> stable epoch-to-epoch number).
        ``max_tiles=None`` scores the WHOLE split at FULL coverage (whole tiles) -- the
        final test, our analogue of KPConv's vote-until-covered pass.
        """
        from ..inference.voting import predict_scene
        self.model.eval()
        cfg = self.cfg
        conf = ConfusionAccumulator()
        rmse_acc = RMSEAccumulator(res=float(getattr(cfg, "dtm_rmse_res", 1.0)))
        conf_raw = ConfusionAccumulator()                     # SPAG-DC: before refinement
        rmse_raw = RMSEAccumulator(res=float(getattr(cfg, "dtm_rmse_res", 1.0)))
        n_reclass_total, n_ground_total = 0, 0
        nl = int(getattr(cfg, "neighbor_limit", 50) or 50)
        mean = getattr(self.val_set, "mean", None)
        std = getattr(self.val_set, "std", None)
        n_tiles = len(self.val_set)
        full_cover = (max_tiles is None or int(max_tiles) <= 0)   # final test: whole tiles
        cap = n_tiles if full_cover else min(int(max_tiles), n_tiles)
        if cap >= n_tiles:
            order = list(range(n_tiles))
        else:
            # evenly-spaced, deterministic subset -> a stable, comparable number each epoch
            order = sorted(set(int(i) for i in np.linspace(0, n_tiles - 1, cap).round().astype(int)))
        win_pts = int(getattr(cfg, "scene_max_points", 102400))   # per-epoch region = one point-budget block
        total_loss, n_used, n_pts = 0.0, 0, 0
        t0 = time.time()
        step = max(1, len(order) // 6)
        for k, fi in enumerate(order):
            try:
                c = self.val_set._load(int(fi))
                pts = np.asarray(c["local"], dtype=np.float32)
                if pts.shape[0] == 0:
                    continue
                labels = np.asarray(c["labels"]).reshape(-1)
                ret = np.asarray(c["returns"])
                inten = c.get("intensity"); rr = c.get("ret_ratio")
                if not full_cover and pts.shape[0] > win_pts:
                    # KPConv/Pointcept bounded region: the max_points points NEAREST the
                    # tile centre, at full density -- IDENTICAL to the training eval crop
                    # (_crop_block) and to ONE predict_scene block -> one forward per tile.
                    xy = pts[:, :2]; ctr = 0.5 * (xy.min(0) + xy.max(0))
                    d2 = (xy[:, 0] - ctr[0]) ** 2 + (xy[:, 1] - ctr[1]) ** 2
                    sel = np.argpartition(d2, win_pts - 1)[:win_pts]
                    pts = pts[sel]; labels = labels[sel]; ret = ret[sel]
                    inten = None if inten is None else np.asarray(inten)[sel]
                    rr = None if rr is None else np.asarray(rr)[sel]
                out = predict_scene(
                    pts, ret[:, 0], ret[:, 1], cfg, self.model, self.device,
                    mean=mean, std=std, prev_dtm=c.get("prior", c.get("dtm")), intensity=inten, ret_ratio=rr,
                    return_proba=True, return_precleanup=True,
                    tta=bool(getattr(cfg, "tta", False)) and full_cover)
                # (pred=refined, proba, pred_raw=pre-refine argmax)
                if isinstance(out, tuple) and len(out) == 3:
                    pred, proba, pred_raw = out
                elif isinstance(out, tuple):
                    pred, proba = out; pred_raw = pred
                else:
                    pred, proba, pred_raw = out, None, out
                pred = np.asarray(pred).reshape(-1)
                pred_raw = np.asarray(pred_raw).reshape(-1)
                conf.update(pred, labels)                      # refined (deployed)
                conf_raw.update(pred_raw, labels)              # raw (pre-SPAG-DC)
                n_ground_total += int((pred_raw == 1).sum())                 # ground == 1
                n_reclass_total += int(((pred_raw == 1) & (pred == 0)).sum())  # ground->non-ground spikes
                n_pts += int(pred.shape[0])
                try:
                    rmse_acc.update(pts, pred, labels)
                    rmse_raw.update(pts, pred_raw, labels)
                except Exception:
                    pass
                # val_loss = mean CE of the merged full-res softmax over non-ignore points
                # (a stable, deployed-inference loss; NOT the train CE+Lovasz, so not
                # directly comparable to the old single-forward val_loss). NaN-safe: a few
                # points can be left uncovered by the block merge (proba/0 -> NaN); skip them.
                if proba is not None:
                    pr = np.asarray(proba, dtype=np.float64)
                    valid = labels < int(IGNORE_LABEL)
                    if valid.any() and pr.ndim == 2 and pr.shape[1] >= 2:
                        p = pr[valid, labels[valid].astype(np.int64)]
                        p = p[np.isfinite(p)]
                        if p.size:
                            ce = float(-np.log(np.clip(p, 1e-7, 1.0)).mean())
                            if np.isfinite(ce):
                                total_loss += ce; n_used += 1
            except Exception as e:
                print(f"  [warn] val tile {fi} skipped: {type(e).__name__}: {e}", flush=True)
            if (k + 1) % step == 0 or (k + 1) == len(order):
                print(f"    val {k + 1}/{len(order)} tiles  pts={n_pts}  {time.time() - t0:.0f}s", flush=True)
        self.model.train()
        metrics = conf.compute()                               # refined = deployed metric
        metrics["RMSE"] = rmse_acc.compute()
        raw_m = conf_raw.compute()                             # pre-refinement, for the before/after delta
        for kk in ("IoU1", "IoU2", "OA", "mIoU"):
            metrics[f"{kk}_raw"] = raw_m.get(kk, float("nan"))
        metrics["RMSE_raw"] = rmse_raw.compute()
        metrics["n_reclassified"] = int(n_reclass_total)
        metrics["reclassified_frac"] = (n_reclass_total / max(n_ground_total, 1))
        return (total_loss / max(n_used, 1)), metrics

    @torch.no_grad()
    def evaluate_split(self, dataset):
        """Score an arbitrary split (e.g. the held-out TEST set) with the SAME
        loader and metric path as validation, so the number is directly comparable
        to the reported val mIoU. Builds the loaders with ``val_set`` temporarily
        pointed at ``dataset``; this is called once after training, so the one-off
        rebuild of the (train) sampler is negligible."""
        saved = self.val_set
        self.val_set = dataset
        try:
            _, loader = self._loaders()
        finally:
            self.val_set = saved
        return self.evaluate(loader, max_tiles=None)

    # ------------------------------------------------------- per-epoch outputs
    @staticmethod
    def _sample_prior_dtm(prior, world_xy):
        """Sample the previous-year prior DTM at scene points, for the inputs panel.
        Accepts the 5-channel ``MultiRaster`` (channel 0 = DTM) or a legacy
        single-channel ``Raster``. Returns ``(N,)`` heights, or None if unavailable."""
        if prior is None:
            return None
        try:
            from ..data.dtm import Raster, sample_dtm
            data = np.asarray(getattr(prior, "data"))
            dtm = data[0] if data.ndim == 3 else data
            ras = Raster(np.asarray(dtm, dtype=np.float32),
                         float(prior.x_min), float(prior.y_min), float(prior.res))
            v = np.asarray(sample_dtm(ras, np.asarray(world_xy, dtype=np.float64)), dtype=np.float32)
            if not np.isfinite(v).any():
                return None
            med = float(np.nanmedian(v[np.isfinite(v)]))
            return np.where(np.isfinite(v), v, med).astype(np.float32)
        except Exception:
            return None

    @torch.no_grad()
    def _per_epoch_vis(self, ep_dir: str, epoch: int):
        """Per-epoch gallery. By default renders N voted full AREAS (sphere-voting
        over a ``vis_area_size`` window around each scene-type centre) - far more
        legible at scene scale than a single 16 m input sphere. Set
        ``vis_full_area=False`` to fall back to the per-sphere renderer. Each item
        is wrapped so a failure skips that panel without affecting training."""
        from ..inference.voting import predict_scene
        self.model.eval()
        cfg = self.cfg
        n_vis = min(cfg.n_vis_tiles, len(self.vis_set))
        half = float(getattr(cfg, "vis_area_size", 100.0)) / 2.0
        nl = int(getattr(cfg, "neighbor_limit", 50) or 50)
        mean = getattr(self.vis_set, "mean", None)
        std = getattr(self.vis_set, "std", None)
        # Gallery voting can use a coarser sphere spacing than training (the relief is
        # rendered at ~1 m/px), cutting the per-epoch vis cost ~(spacing ratio)^2 with no
        # (sphere-mode spacing override removed with sphere mode. No
        # effect in scene mode, which renders via PTv3-native one-forward-per-block.)
        rendered = 0
        for i in range(len(self.vis_set)):
            if rendered >= n_vis:
                break
            try:
                fi, ck, center = self.vis_set.cands[i]
                c = self.vis_set._load(int(fi))
                pts = c["local"]; origin = np.asarray(c["origin"])
                center = np.asarray(center, dtype=np.float64)
                d = pts[:, :2].astype(np.float64) - center[:2]
                m = (np.abs(d[:, 0]) <= half) & (np.abs(d[:, 1]) <= half)
                if int(m.sum()) < int(getattr(cfg, 'scene_min_points', 100)):
                    continue
                wp = pts[m].astype(np.float32)
                ret = c["returns"][m]
                pred, pred_raw = predict_scene(
                    wp, ret[:, 0], ret[:, 1], cfg, self.model, self.device,
                    mean=mean, std=std, prev_dtm=c.get("prior", c.get("dtm")),
                    intensity=c["intensity"][m], ret_ratio=c["ret_ratio"][m],
                    return_precleanup=True)
                world = wp.astype(np.float64) + origin[None, :]
                y_true = np.asarray(c["labels"])[m]

                base = os.path.splitext(os.path.basename(self.vis_set.files[int(fi)]))[0]
                title = f"MEEPO  (epoch {epoch})  {base}  [{int(cfg.vis_area_size)} m]"
                # ---- assemble the model's per-point INPUTS for the combined panel ----
                inten = np.asarray(c["intensity"])[m]
                rr = np.asarray(c["ret_ratio"])[m]
                feats = {"return_count": np.asarray(ret[:, 0], dtype=np.float32),
                         "return_ratio": np.asarray(rr, dtype=np.float32),
                         "intensity": np.asarray(inten, dtype=np.float32)}
                prior_dtm = self._sample_prior_dtm(c.get("prior", c.get("dtm")), world[:, :2])
                if prior_dtm is not None:
                    feats["prior_dtm"] = prior_dtm
                if cfg.render_errors_every_epoch:
                    # ONE combined image per scene: inputs + gap-free TIN DEMs
                    # (DSM / true DTM / predicted DTM) + classification + profiles.
                    try:
                        render_scene_report(world, y_true, pred,
                                             os.path.join(ep_dir, f"scene_{base}.png"),
                                             feats=feats, title=title)
                    except Exception as e:
                        print(f"  [warn] scene report render skipped for {base}: {e}")
                    # SPAG-DC before/after: spikes removed from the DTM surface + RMSE delta.
                    if str(getattr(cfg, "refine_method", "spag_dc")).lower() not in ("off", "none", ""):
                        try:
                            render_spag_dc_panel(
                                world, pred_raw, pred,
                                os.path.join(ep_dir, f"refine_{base}.png"),
                                y_true=y_true,
                                title=f"SPAG-DC refine  (epoch {epoch})  {base}")
                        except Exception as e:
                            print(f"  [warn] SPAG-DC panel skipped for {base}: {e}")
                if cfg.write_laz_every_epoch:
                    try:
                        # one CLEAN LAZ per scene (standard LAS fields only; QGIS-safe)
                        write_classified(
                            os.path.join(ep_dir, f"scene_{base}.laz"),
                            world, pred,
                            num_returns=ret[:, 0], return_number=ret[:, 1],
                            intensity=inten)
                    except Exception as e:
                        print(f"  [warn] LAZ write skipped for {base}: {e}")
                rendered += 1
            except Exception as e:
                print(f"  [warn] area gallery item {i} skipped: {e}")
        self.model.train()


    def _save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        torch.save({
            "epoch": epoch,
            "model_state": getattr(self.model, "_orig_mod", self.model).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": {k: getattr(self.cfg, k) for k in vars(self.cfg)}
            if hasattr(self.cfg, "__dict__") else {},
            "metrics": metrics,
            "in_features_dim": self.cfg.in_features_dim,
        }, path)
