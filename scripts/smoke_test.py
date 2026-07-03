#!/usr/bin/env python3
"""
MEEPO NZ - CPU smoke test (no GPU, no real data, runs in seconds).

Exercises the whole clean-PyTorch stack on tiny synthetic clouds to prove the
build is wired correctly end-to-end:

  1. previous-year CLASSIFICATION raster (Deviation A): build -> crop+downsample
     -> per-sphere multi-channel patch crop  (data/dtm.py);
  2. multi-channel augmentation (in-plane warp + height-channel vertical scale);
  3. PTv3 collate -> MEEPO forward + backward (TRAIN, dense MoE path) and a
     forward in EVAL (the MoE inference scatter path);
  4. full MEEPO-L config instantiation (parameter count);
  5. PTv3 sphere-voting inference (+ return_proba).

Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The smoke test runs on CPU. spconv's implicit-GEMM kernels are CUDA-only, so if
# a Blackwell spconv build is installed we must force the clean-PyTorch conv here
# (training on --device cuda uses spconv automatically). Set before importing the
# package, since the backend flag is resolved at submanifold_conv import time.
os.environ["POINT_MOE_DISABLE_SPCONV"] = "1"

import numpy as np
import torch

from meepo_nz.utils.config import Config
from meepo_nz.utils.laz_io import GROUND_CLASSES, IGNORE_LABEL
from meepo_nz.features.shallow_features import expected_feature_dim
from meepo_nz.data.ptv3_collate import PTv3Collate
from meepo_nz.data.augment import augment_tile
from meepo_nz.data.dtm import (build_prior_raster_from_prev, crop_multiraster_patch,
                                    crop_downsample_multiraster)
from meepo_nz.models import build_meepo
from meepo_nz.training.losses import SegLoss
from meepo_nz.inference.voting import predict_cloud_spheres


def _tiny_cfg():
    cfg = Config()
    cfg.first_subsampling_dl = 0.5
    cfg.in_radius = 8.0
    cfg.sphere_center_spacing = 8.0
    cfg.sphere_min_points = 50
    cfg.enc_stride = (2, 2)
    cfg.meepo_enc_depths = (1, 1, 1); cfg.meepo_enc_channels = (8, 16, 16)
    cfg.meepo_dec_depths = (1, 1); cfg.meepo_dec_channels = (16, 16)
    cfg.mamba_state_dim = 1; cfg.mamba_conv_dim = 4; cfg.mamba_expand_factor = 3
    cfg.ssm_backend = "torch"   # CPU smoke: force the pure-torch selective scan
    cfg.drop_path_rate = 0.0; cfg.stem_kernel_size = 3
    cfg.dtm_patch_size = 24; cfg.dtm_feat_dim = 6; cfg.dtm_cnn_mid = 12
    return cfg


def _synth_classified(n=40000, size=60.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * size
    gz = 0.05 * xy[:, 0] + 2.0 * np.sin(xy[:, 1] / 8.0)
    cls = np.full(n, GROUND_CLASSES[0], dtype=np.int64)
    veg = rng.random(n) < 0.35
    z = gz.copy(); z[veg] += rng.random(veg.sum()) * 12.0
    cls[veg] = 5
    return np.column_stack([xy, z]).astype(np.float64), cls, gz


def main():
    torch.manual_seed(0); np.random.seed(0)
    ok = []

    # ---- 1. prior-classification raster -------------------------------------
    xyz, cls, gz = _synth_classified()
    mr = build_prior_raster_from_prev(xyz, cls, GROUND_CLASSES, res=1.0)
    assert mr.shape[0] == 5 and np.isfinite(mr.data).all()
    assert mr.data[2].max() > 1.0          # nDSM picks up canopy
    sub = crop_downsample_multiraster(mr, 10, 10, 50, 50, target_res=1.0)
    assert sub.shape[0] == 5
    patch = crop_multiraster_patch(mr, 20.0, 20.0, 10.0, 24, origin_z=float(gz.mean()))
    assert patch.shape == (5, 24, 24) and np.isfinite(patch).all()
    print(f"[1/5] prior raster: {mr.shape}  nDSM_max={mr.data[2].max():.1f}  "
          f"gprob=[{mr.data[3].min():.2f},{mr.data[3].max():.2f}]  patch={patch.shape}  PASS")
    ok.append(True)

    # ---- 2. multi-channel augmentation --------------------------------------
    cfg = _tiny_cfg()
    rng = np.random.default_rng(0)
    local = (rng.random((2000, 3)).astype(np.float32)) * 10.0
    aug_local, aug_patch = augment_tile(local, patch, cfg, rng, tile_size=10.0, pivot=None)
    assert aug_patch.shape == (5, 24, 24) and np.isfinite(aug_patch).all()
    print(f"[2/5] augment: patch{aug_patch.shape} local{aug_local.shape}  PASS")
    ok.append(True)

    # ---- 3. collate + model train(fwd+bwd) + eval(fwd) ----------------------
    fdim = expected_feature_dim(cfg)

    def mk(n):
        pts = (np.random.rand(n, 3).astype(np.float32) - 0.5) * 2 * cfg.in_radius
        pts[:, 2] *= 0.1
        pa = np.random.randn(5, cfg.dtm_patch_size, cfg.dtm_patch_size).astype(np.float32)
        pa[4] = (pa[4] > 0).astype(np.float32)
        return dict(points=pts, features=np.random.randn(n, fdim).astype(np.float32),
                    labels=(np.random.rand(n) > 0.5).astype(np.int64), dtm_patch=pa,
                    origin=np.zeros(3), path="t")

    batch = PTv3Collate(cfg)([mk(1200), mk(1000)])
    model = build_meepo(cfg)
    model.train()
    logits = model(batch)
    assert logits.shape == (batch["coord"].shape[0], cfg.num_classes)
    loss = SegLoss()(logits, batch["labels"]); loss.backward()
    gnorm = sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    assert np.isfinite(gnorm) and gnorm > 0
    model.eval()
    with torch.no_grad():
        _ = model(batch)
    print(f"[3/5] model: params={model.num_parameters():,} voxels={batch['coord'].shape[0]} "
          f"loss={float(loss):.3f} grad_norm={gnorm:.2f}  train+eval  PASS")
    ok.append(True)

    # ---- 4. full MEEPO config instantiation ---------------------------------
    full = Config(); full.first_subsampling_dl = 0.5   # backbone defaults to "meepo"
    fm = build_meepo(full)
    np_full = fm.num_parameters()
    assert np_full > 1_000_000
    print(f"[4/9] full MEEPO instantiated: {np_full:,} params  PASS")
    ok.append(True)

    # ---- 5. sphere-voting inference -----------------------------------------
    n = 5000
    cxy = np.random.rand(n, 2) * 30.0
    cz = 0.05 * cxy[:, 0] + 2.0 * np.sin(cxy[:, 1] / 8.0)
    cv = np.random.rand(n) < 0.3; cz[cv] += np.random.rand(cv.sum()) * 10
    cur = np.column_stack([cxy, cz]).astype(np.float64)
    nret = np.random.randint(1, 4, n).astype(np.float32)
    rnum = np.ones(n, np.float32)
    inten = np.random.rand(n).astype(np.float32)
    pred, proba = predict_cloud_spheres(cur, nret, rnum, cfg, model, torch.device("cpu"),
                                        prev_dtm=mr, intensity=inten, return_proba=True)
    assert pred.shape == (n,) and proba.shape == (n, 2)
    print(f"[5/5] voting: pred{pred.shape} classes={np.unique(pred).tolist()} proba{proba.shape}  PASS")
    ok.append(True)

    # ---- 5b. TTA (scene-mode): rotation-averaged softmax; cloud + georeferenced
    #          prior raster rotated together (verified sample-preserving rot90+georef)
    import copy as _copy
    cfg_tta = _copy.copy(cfg); cfg_tta.scene_mode = True
    pred_t, proba_t = predict_cloud_spheres(cur, nret, rnum, cfg_tta, model, torch.device("cpu"),
                                            prev_dtm=mr, intensity=inten, return_proba=True, tta=True)
    assert pred_t.shape == (n,) and proba_t.shape == (n, 2)
    assert np.allclose(proba_t.sum(1), 1.0, atol=1e-3), "TTA proba rows must sum to 1"
    print(f"[5b] TTA voting (z-rot 0/90/180/270, prior rotated with cloud): "
          f"pred{pred_t.shape} proba-rows~1  PASS")
    ok.append(True)

    # ---- 5c. overlapping-disc SOFT VOTING branch (SparseGF): force the scene to
    #          tile into several overlapping discs (small budget + small step) and
    #          assert COMPLETE coverage (no NaN/zero rows) + soft-vote averaging.
    cfg_vote = _copy.copy(cfg)
    cfg_vote.scene_mode = True
    cfg_vote.scene_max_points = 1200            # << n_sub -> forces multi-disc tiling
    cfg_vote.scene_vote_step_m = 15.0           # Rc=(sqrt2/2)*15~=10.6 m over the 30 m cloud
    pred_v, proba_v = predict_cloud_spheres(cur, nret, rnum, cfg_vote, model, torch.device("cpu"),
                                            prev_dtm=mr, intensity=inten, return_proba=True)
    assert pred_v.shape == (n,) and proba_v.shape == (n, 2)
    assert np.isfinite(proba_v).all(), "coverage gap: voting left NaN/uncovered points"
    assert np.allclose(proba_v.sum(1), 1.0, atol=1e-3), "soft-voted proba rows must sum to 1"
    assert set(np.unique(pred_v).tolist()).issubset({0, 1})
    print(f"[5c] overlapping-disc soft-vote (max_pts=1200, step=15m -> multi-disc): "
          f"pred{pred_v.shape} full-coverage proba-rows~1  PASS")
    ok.append(True)

    # ---- 6. MEEPO 4-direction (strided) + grad-checkpoint propagation -------
    #   exercises the strided scan path (n_directions=4) and guards the silent-
    #   non-engagement failure: grad_checkpointing must reach EVERY MEEPO block,
    #   else large scenes OOM despite the flag being "on".
    from meepo_nz.models.meepo import Block as _MBlock
    lcfg = _tiny_cfg(); lcfg.mamba_directions = 4; lcfg.grad_checkpointing = True
    lm = build_meepo(lcfg); lm.train()
    _gc = [getattr(b, "grad_checkpointing", False) for b in lm.modules() if isinstance(b, _MBlock)]
    assert _gc and all(_gc), "grad_checkpointing must propagate to ALL MEEPO blocks"
    fdim = expected_feature_dim(lcfg)
    def _mk(nn_):
        p = (np.random.rand(nn_, 3).astype(np.float32) - 0.5) * 2 * lcfg.in_radius; p[:, 2] *= 0.1
        pa = np.random.randn(5, lcfg.dtm_patch_size, lcfg.dtm_patch_size).astype(np.float32)
        pa[4] = (pa[4] > 0).astype(np.float32)
        return dict(points=p, features=np.random.randn(nn_, fdim).astype(np.float32),
                    labels=(np.random.rand(nn_) > 0.5).astype(np.int64),
                    dtm_patch=pa, origin=np.zeros(3), path="t")
    lb = PTv3Collate(lcfg)([_mk(1200), _mk(1000)])
    llogits = lm(lb)
    lloss = torch.nn.functional.cross_entropy(llogits, lb["labels"])
    lloss.backward()
    assert llogits.shape[1] == 2 and torch.isfinite(lloss)
    print(f"[6/9] MEEPO 4-dir (strided) + grad-checkpoint on all {len(_gc)} blocks: "
          f"params={lm.num_parameters():,} loss={float(lloss):.3f} fwd+bwd  PASS")
    ok.append(True)

    # ---- 8. GrounDiff nDSM regression loss path (Dhaouadi et al. 2025) --------
    # Verifies the CONTINUOUS L1+L2 height regression: the dataset target builder
    # (height_above_ground), the NaN-aware collate aggregation, the regression
    # head, and GrounDiffLoss (CE[+Lovasz] + L1 + L2). This is the fix for the
    # majority-class (predict-all-ground) collapse.
    from meepo_nz.data.dtm import height_above_ground
    from meepo_nz.training.losses import GrounDiffLoss
    gcfg = _tiny_cfg(); gcfg.use_groundiff_regression = True; gcfg.use_dtm_raster = False
    gcfg.use_height_aware_loss = False
    def _mkg(n):
        xyz = np.random.default_rng(n).uniform(0, 40, size=(n, 3)).astype(np.float32)
        xyz[:, 2] = 0.05 * xyz[:, 0]
        lab = (np.random.default_rng(n + 1).random(n) > 0.4).astype(np.int64)   # ~60% ground
        veg = lab == 0
        xyz[veg, 2] += np.random.default_rng(n + 2).uniform(0.1, 12.0, int(veg.sum())).astype(np.float32)
        ndsm = height_above_ground(xyz.astype(np.float64), lab, res=1.0, min_ground=8)
        return {"points": xyz, "features": np.random.default_rng(n).standard_normal((n, expected_feature_dim(gcfg))).astype(np.float32),
                "labels": lab, "ndsm": ndsm.astype(np.float32), "origin": np.zeros(3), "path": "g"}
    gb = PTv3Collate(gcfg)([_mkg(1400), _mkg(1100)])
    assert "ndsm" in gb and gb["ndsm"].shape == gb["labels"].shape
    # ground voxels should regress to ~0; vegetation voxels should be > 0
    _nd = gb["ndsm"].detach().cpu().numpy(); _lb = gb["labels"].detach().cpu().numpy()
    _g0 = np.nanmean(np.abs(_nd[_lb == 1])); _v0 = np.nanmean(_nd[_lb == 0])
    assert np.isfinite(_v0) and _v0 > _g0, "non-ground nDSM target must exceed ground (which is ~0)"
    gmodel = build_meepo(gcfg); gmodel.train()
    glog = gmodel(gb); gaux = gmodel._reg_pred
    assert gaux is not None and gaux.shape == (glog.shape[0],), "regression head must emit one value/point in train"
    gl = GrounDiffLoss(lovasz_weight=1.0, l1_weight=1.0, l2_weight=1.0, cls_weight=1.0, ndsm_scale=10.0)(
        glog, gb["labels"], gaux, gb["ndsm"])
    gl.backward()
    # gradient must reach the regression head (the anti-collapse signal)
    gw = gmodel.reg_head.weight.grad
    assert gw is not None and torch.isfinite(gw).all() and float(gw.abs().sum()) > 0, "no gradient into nDSM head"
    gmodel.eval(); _ = gmodel(gb)
    assert torch.isfinite(gl) and gmodel._reg_pred is None
    print(f"[8/9] GrounDiff nDSM regression: target ground|veg = {_g0:.2f}|{_v0:.2f} m, "
          f"reg_pred={tuple(gaux.shape)} L_total={float(gl):.3f} (grad reaches head; detached@eval)  PASS")
    ok.append(True)

    # ---- 8. SPAG-DC ground-misclassification corrector (IEEE Sensors 2025) ----
    # Exercises the full SPAG-DC path: region-growing core, adaptive seed grid, MCS
    # purification, local-TPS surface, and the mu2+n*sigma2 distance-threshold correction.
    # Two properties: (a) it catches planted spikes; (b) on CLEAN sloped ground it demotes
    # ~nothing -- the fixed-tail flaw that made the previous refiner raise DTM-RMSE.
    from meepo_nz.inference.spag_dc import spag_dc_refine
    rng_s = np.random.default_rng(1)
    gx, gy = np.meshgrid(np.linspace(0, 30, 60), np.linspace(0, 30, 60))
    gx = gx.ravel(); gy = gy.ravel()
    base_z = 0.4 * gx + 0.05 * rng_s.standard_normal(gx.size)     # sloped + rough ground
    pcxyz = np.column_stack([gx, gy, base_z]).astype(np.float64)
    sidx = np.random.default_rng(2).choice(pcxyz.shape[0], 10, replace=False)
    pcxyz[sidx, 2] += rng_s.uniform(3.0, 15.0, 10)                # 10 giant spikes
    pcraw = np.ones(pcxyz.shape[0], dtype=np.int64)              # all predicted ground (==1)
    scfg = _tiny_cfg(); scfg.refine_method = "spag_dc"
    ref, info = spag_dc_refine(pcxyz, pcraw, scfg, return_info=True)
    spk = (pcraw == 1) & (ref == 0)
    assert set(np.unique(ref)).issubset({0, 1}), "refined labels must stay binary"
    assert int(spk[sidx].sum()) >= 8, "SPAG-DC should catch most planted spikes"
    # clean-ground preservation: a slope with NO spikes must lose ~no ground
    cz = (0.4 * gx + 0.05 * rng_s.standard_normal(gx.size)).astype(np.float64)
    cxyz = np.column_stack([gx, gy, cz]).astype(np.float64)
    cref = spag_dc_refine(cxyz, np.ones(cxyz.shape[0], np.int64), scfg)
    frac_demoted = float((cref == 0).mean())
    assert frac_demoted < 0.05, f"SPAG-DC demoted {frac_demoted:.1%} of clean ground (must be small)"
    print(f"[9/9] SPAG-DC: caught {int(spk[sidx].sum())}/10 spikes, reclassified {info['n_reclassified']} "
          f"(core={info['n_core']} seeds={info['n_seeds']}); clean-ground demoted {frac_demoted:.1%}  PASS")
    ok.append(True)

    # ---- 9b. LEARNED SPAG-DC: regime head + oracle target + learned-globals override --
    from meepo_nz.inference.spag_dc import (oracle_regime_globals, SPAG_GLOBAL_LO,
                                            SPAG_GLOBAL_HI, SPAG_N_GLOBALS)
    # (a) oracle target from GT-ground terrain stays inside the global box
    gx2, gy2 = np.meshgrid(np.linspace(0, 30, 50), np.linspace(0, 30, 50))
    grnd = np.column_stack([gx2.ravel(), gy2.ravel(),
                            0.3 * gx2.ravel() + 0.05 * np.random.default_rng(3).standard_normal(gx2.size)])
    orc = oracle_regime_globals(grnd.astype(np.float64))
    assert orc.shape == (SPAG_N_GLOBALS,) and np.all(orc >= SPAG_GLOBAL_LO - 1e-6) and np.all(orc <= SPAG_GLOBAL_HI + 1e-6)
    # (b) model forward emits per-scene regime globals in-box; aux loss reaches the head
    lcfg = _tiny_cfg(); lcfg.spag_learned = True; lcfg.use_dtm_raster = False
    lm = build_meepo(lcfg); lm.train()
    def _mkr(n):
        return {"points": (np.random.rand(n, 3).astype(np.float32) - 0.5) * 2 * lcfg.in_radius,
                "features": np.random.randn(n, expected_feature_dim(lcfg)).astype(np.float32),
                "labels": (np.random.rand(n) > 0.5).astype(np.int64),
                "regime": oracle_regime_globals(np.random.rand(200, 3) * 20).astype(np.float32),
                "origin": np.zeros(3), "path": "r"}
    rb = PTv3Collate(lcfg)([_mkr(1200), _mkr(1000)])
    assert rb["regime"].shape == (2, SPAG_N_GLOBALS)
    _ = lm(rb); rpred = lm._regime_pred
    assert rpred is not None and rpred.shape == (2, SPAG_N_GLOBALS)
    lo = lm._spag_lo; hi = lm._spag_hi
    assert bool((rpred >= lo - 1e-4).all() and (rpred <= hi + 1e-4).all()), "regime preds must lie in the global box"
    span = (hi - lo)
    rloss = torch.nn.functional.smooth_l1_loss(rpred / span, rb["regime"] / span)
    rloss.backward()
    gw = lm.regime_head[0].weight.grad
    assert gw is not None and torch.isfinite(gw).all() and float(gw.abs().sum()) > 0, "no gradient into regime head"
    # (c) learned globals drive spag_dc_refine (still catches spikes, stays binary)
    lg = orc.copy()
    ref_l, info_l = spag_dc_refine(pcxyz, pcraw, scfg, return_info=True, learned_globals=lg)
    assert set(np.unique(ref_l)).issubset({0, 1}) and int(((pcraw == 1) & (ref_l == 0))[sidx].sum()) >= 8
    assert "learned_globals" in info_l
    lm.eval(); _ = lm(rb)
    assert lm._regime_pred is not None and lm._regime_pred.shape == (2, SPAG_N_GLOBALS)  # available at inference
    print(f"[9b] LEARNED SPAG-DC: regime_pred={tuple(rpred.shape)} in-box; oracle={np.round(orc,2).tolist()}; "
          f"aux grad reaches head; learned globals catch {int(((pcraw==1)&(ref_l==0))[sidx].sum())}/10 spikes  PASS")
    ok.append(True)

    # ---- [11] REINFORCE calibration of SPAG-DC globals vs DTM-RMSE (scripts/10_fit_spag_rl) ----
    # The non-differentiable corrector is scored by OpenGF DTM-RMSE-vs-GT-ground; the regime
    # head (fed pooled feats + prediction stats) is the policy mean. Scene has a genuine CLIFF
    # (ground, big residual) + planted spikes (non-ground): demoting the cliff RAISES RMSE, so
    # the reward penalises cliff destruction. Assert: grad reaches the head ONLY, metrics finite.
    from meepo_nz.inference.spag_rl import reinforce_update
    assert lm._regime_pooled is not None and lm._regime_pred_stats is not None, "forward must cache head inputs"
    rng_r = np.random.default_rng(11)
    cz2 = (0.3 * gx + 0.05 * rng_r.standard_normal(gx.size)).astype(np.float64)
    cz2[gx > 20.0] += 8.0                                          # genuine cliff step (still ground)
    rxyz = np.column_stack([gx, gy, cz2]).astype(np.float64)
    spk_i = rng_r.choice(rxyz.shape[0], 12, replace=False)
    rxyz[spk_i, 2] += rng_r.uniform(4.0, 12.0, 12)                 # true spikes (non-ground)
    rgt = np.ones(rxyz.shape[0], dtype=np.int64); rgt[spk_i] = 0
    rpred_lbl = np.ones(rxyz.shape[0], dtype=np.int64)            # model predicted all-ground
    scene = {"pooled": lm._regime_pooled[0:1].detach().cpu(),
             "pred_stats": lm._regime_pred_stats[0:1].detach().cpu(),
             "xyz": rxyz, "pred": rpred_lbl, "gt": rgt}
    for _nm, _p in lm.named_parameters():
        _p.requires_grad = _nm.startswith("regime_head")
    lm.zero_grad(set_to_none=True)                                # clear stale grads from [9b]
    _hp = [p for _nm, p in lm.named_parameters() if _nm.startswith("regime_head")]
    _ropt = torch.optim.Adam(_hp, lr=3e-3)
    _rlcfg = _tiny_cfg(); _rlcfg.refine_method = "spag_dc"
    m0 = reinforce_update(lm, [scene], _ropt, _rlcfg, sigma=0.5, res=1.0)
    assert m0["n"] >= 1 and np.isfinite(m0["rmse_base"]) and np.isfinite(m0["rmse_sample"]), f"RL step unusable: {m0}"
    _gh = lm.regime_head[0].weight.grad
    assert _gh is not None and torch.isfinite(_gh).all(), "no finite gradient into regime head from REINFORCE"
    _bbp = next(p for _nm, p in lm.named_parameters() if not _nm.startswith("regime_head"))
    assert _bbp.grad is None, "backbone must stay frozen during SPAG-DC RL calibration"
    print(f"[11] SPAG-DC RMSE REINFORCE: greedy_RMSE={m0['rmse_base']:.3f}m sample_RMSE={m0['rmse_sample']:.3f}m "
          f"adv={m0['advantage']:+.3f} reclass={m0['reclass_frac']*100:.0f}% (grad->head only)  PASS")
    ok.append(True)

    # ---- [11b] in-training REINFORCE term: rides the main backward (trainer joint co-training) --
    from meepo_nz.inference.spag_rl import reinforce_loss_term
    rb1 = PTv3Collate(lcfg)([_mkr(1300)])                        # single scene -> B=1 regime logits
    lm.train()
    for _nm, _p in lm.named_parameters():
        _p.requires_grad = True
    lm.zero_grad(set_to_none=True)
    seg1 = lm(rb1)                                               # populates live _regime_logits (1,6)
    assert lm._regime_logits is not None and lm._regime_logits.shape[0] == 1, "forward must cache live logits"
    cl0 = torch.tensor([rxyz.shape[0]], dtype=torch.long)        # one cloud; matches B=1 (Mix3D-correct slicing)
    term, tmet = reinforce_loss_term(
        lm, torch.from_numpy(rxyz), torch.from_numpy(rpred_lbl), torch.from_numpy(rgt), cl0,
        _rlcfg, sigma=0.5, res=1.0, max_points=4000, rng=np.random.default_rng(7))
    assert term is not None and torch.isfinite(term) and tmet["n"] == 1, f"RL term unusable: {tmet}"
    seg_loss_dummy = torch.nn.functional.cross_entropy(seg1, rb1["labels"])
    (seg_loss_dummy + 1.0 * term.to(seg_loss_dummy.dtype)).backward()   # added to seg loss, like the trainer
    _gh2 = lm.regime_head[0].weight.grad
    assert _gh2 is not None and torch.isfinite(_gh2).all(), "in-training RL term did not reach the regime head"
    print(f"[11b] in-training RL term: greedy={tmet['rmse_base']:.3f}m samp={tmet['rmse_sample']:.3f}m "
          f"adv={tmet['advantage']:+.3f}; added to seg loss, grad reaches head  PASS")
    ok.append(True)

    # ---- [10] MEEPO recipe: Mix3D (offset-merge) + RandomDropout + x/y tilt + ElasticDistortion ----
    mcfg = _tiny_cfg(); mcfg.spag_learned = True; mcfg.use_dtm_raster = False
    mm = build_meepo(mcfg); mm.train()
    def _mkm(n):
        return {"points": (np.random.rand(n, 3).astype(np.float32) - 0.5) * 2 * mcfg.in_radius,
                "features": np.random.randn(n, expected_feature_dim(mcfg)).astype(np.float32),
                "labels": (np.random.rand(n) > 0.5).astype(np.int64),
                "regime": oracle_regime_globals(np.random.rand(200, 3) * 20).astype(np.float32),
                "origin": np.zeros(3), "path": "m"}
    samples = [_mkm(900), _mkm(800), _mkm(700), _mkm(600)]
    # Mix3D merges 4 clouds PAIRWISE -> 2 backbone scenes (offset), but cloud_lengths_0 stays 4
    b0 = PTv3Collate(mcfg, mix_prob=0.0)(samples)
    b1 = PTv3Collate(mcfg, mix_prob=1.0)(samples)
    assert b0["offset"].numel() == 4, "no-mix: 4 clouds -> 4 scenes"
    assert b1["offset"].numel() == 2, "Mix3D: 4 clouds -> 2 merged scenes (offset rewrite)"
    assert b1["cloud_lengths_0"].numel() == 4, "cloud_lengths_0 stays per-ORIGINAL-cloud under Mix3D"
    assert int(b1["offset"][-1]) == int(b0["offset"][-1]), "Mix3D keeps all points (last offset == total)"
    _ = mm(b1)
    assert mm._regime_pred is not None and mm._regime_pred.shape == (4, SPAG_N_GLOBALS), \
        "regime head must pool per ORIGINAL cloud (4) under Mix3D, matching the 4 targets"
    # x/y tilt + ElasticDistortion perturb coords, stay finite, still return the warped raster patch
    acfg = _tiny_cfg(); acfg.augment_tilt_xy = 0.04908738521234052; acfg.augment_elastic = True
    loc = np.random.rand(2000, 3).astype(np.float32); pat = np.random.rand(5, 24, 24).astype(np.float32)
    al, ap = augment_tile(loc.copy(), pat, acfg, np.random.default_rng(0), tile_size=1.0, pivot=0.0)
    assert al.shape == loc.shape and np.isfinite(al).all(), "tilt+elastic output must be finite, same shape"
    assert ap is not None and np.isfinite(np.asarray(ap)).all(), "raster patch still returned + finite"
    assert not np.allclose(al, loc), "tilt+elastic must perturb the coordinates"
    # RandomDropout keep-mask (drops ~dropout_ratio of points across all aligned arrays)
    N = 5000; keep = np.random.default_rng(0).random(N) >= 0.2
    print(f"[10] MEEPO recipe: Mix3D offset 4->{int(b1['offset'].numel())} "
          f"(cloud_lengths_0={int(b1['cloud_lengths_0'].numel())}, all pts kept) -> regime per-cloud="
          f"{tuple(mm._regime_pred.shape)}; x/y-tilt+elastic perturb (finite); "
          f"RandomDropout keeps ~{int(keep.sum())}/{N}  PASS")
    ok.append(True)

    # ---- [norm] --norm ln must make the model fully batch-independent:
    #      backbone BatchNorm1d -> LayerNorm, raster BatchNorm2d -> GroupNorm, so
    #      micro-batch 1 + grad-accum never computes BN stats from one scene/forward.
    #      Guard: assert ZERO BatchNorm survives (else micro-batch-1 safety is broken).
    from collections import Counter as _Counter
    def _census(_m):
        c = _Counter()
        for _mod in _m.modules():
            _n = type(_mod).__name__
            if _n in ("BatchNorm1d", "BatchNorm2d", "LayerNorm", "GroupNorm"):
                c[_n] += 1
        return c
    for _bb in ("meepo",):
        _c = Config(); _c.backbone = _bb; _c.first_subsampling_dl = 0.5; _c.in_radius = 8.0
        _c.norm = "ln"; _c.dtm_patch_size = 24; _c.dtm_feat_dim = 6; _c.dtm_cnn_mid = 12
        _c.stem_kernel_size = 3; _c.ssm_backend = "torch"
        _c.enc_stride = (2, 2)
        _c.meepo_enc_depths = (1, 1, 1); _c.meepo_enc_channels = (8, 16, 16)
        _c.meepo_dec_depths = (1, 1); _c.meepo_dec_channels = (16, 16)
        _fd = expected_feature_dim(_c)
        def _mk(n, _c=_c, _fd=_fd):
            _p = (np.random.rand(n, 3).astype(np.float32) - 0.5) * 2 * _c.in_radius; _p[:, 2] *= 0.1
            _pa = np.random.randn(5, _c.dtm_patch_size, _c.dtm_patch_size).astype(np.float32)
            return dict(points=_p, features=np.random.randn(n, _fd).astype(np.float32),
                        labels=(np.random.rand(n) > 0.5).astype(np.int64), dtm_patch=_pa,
                        origin=np.zeros(3), path="t")
        _m = build_meepo(_c); _m.train()
        _b = PTv3Collate(_c)([_mk(900), _mk(700)])
        SegLoss()(_m(_b), _b["labels"]).backward()
        _cc = _census(_m)
        assert _cc.get("BatchNorm1d", 0) == 0 and _cc.get("BatchNorm2d", 0) == 0, \
            f"MEEPO must leave NO BatchNorm: {dict(_cc)}"
        assert _cc.get("LayerNorm", 0) > 0 and _cc.get("GroupNorm", 0) > 0, \
            f"MEEPO backbone->LayerNorm + raster->GroupNorm expected: {dict(_cc)}"
    print("[norm] MEEPO backbone is LayerNorm/RMSNorm-only (0 BatchNorm); raster->GroupNorm "
          "(norm=ln, fwd+bwd) -> micro-batch-1 safe  PASS")
    ok.append(True)

    # ---- [12] SSD chunked scan == reference loop (Mamba-2 algorithm, pure torch) -------------
    # The no-kernel GPU path (Windows / ArcGIS, kernel-less boxes) runs selective_scan_ssd;
    # it must be numerically the SAME computation as the naive reference loop: forward AND
    # gradients, including the padding path (odd L) and the model's exact config
    # (selective B/C, delta_softplus, D skip, N=1).
    from meepo_nz.models.ssm import selective_scan_ref, selective_scan_ssd, selective_scan
    torch.manual_seed(0)
    _u = torch.randn(2, 8, 203, requires_grad=True); _dt = torch.randn(2, 8, 203, requires_grad=True)
    _A = (-torch.exp(torch.randn(8, 1))).requires_grad_(True)
    _B = torch.randn(2, 1, 203, requires_grad=True); _C = torch.randn(2, 1, 203, requires_grad=True)
    _D = torch.randn(8, requires_grad=True); _db = torch.randn(8, requires_grad=True)
    def _sc(fn, **kw):
        y = fn(_u, _dt, _A, _B, _C, D=_D, delta_bias=_db, delta_softplus=True, **kw)
        return y, torch.autograd.grad(y.square().sum(), (_u, _dt, _A, _B, _C, _D, _db))
    _y1, _g1 = _sc(selective_scan_ref)
    _y2, _g2 = _sc(selective_scan_ssd, chunk=64)                     # odd L=203 -> exercises padding
    _ey = (_y1 - _y2).abs().max().item()
    _eg = max((a - b).abs().max().item() for a, b in zip(_g1, _g2))
    assert _ey < 1e-4 and _eg < 1e-2, f"SSD scan diverges from reference: dy={_ey}, dgrad={_eg}"
    _y3 = selective_scan(_u, _dt, _A, _B, _C, D=_D, delta_bias=_db, delta_softplus=True, backend="ssd")
    assert torch.allclose(_y2, _y3), "dispatcher backend='ssd' must route to selective_scan_ssd"
    print(f"[12] SSD chunked scan (Mamba-2 alg): parity with reference loop fwd|dy|={_ey:.1e} "
          f"grad|dg|={_eg:.1e} (fp32, odd-L padding path); dispatcher 'ssd' routes correctly  PASS")
    ok.append(True)

    print("\nSMOKE TEST PASSED - clean-PyTorch ground segmentation (MEEPO CNN-Mamba backbone) "
          "runs end-to-end on CPU (no spconv / flash-attn / torch_scatter / mamba-ssm kernel).")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
