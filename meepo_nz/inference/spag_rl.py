"""Black-box (REINFORCE) calibration of the learned SPAG-DC globals against DTM-RMSE.

SPAG-DC is non-differentiable (region growing + adaptive quadtree + MCS + TPS +
residual threshold), so it cannot be trained end-to-end. Instead we treat the
regime head's six pre-squash outputs as the mean of a diagonal-Gaussian policy
over the SPAG-DC globals, sample parameters, run the *real* corrector, and score
the result with the OpenGF **DTM-RMSE against GT ground** -- the same metric the
validator reports. Crucially that metric is computed over a DEM grid spanning the
whole tile, so demoting genuine cliff-ground (the failure you saw) corrupts the
predicted DEM at the cliff pixels and *raises* RMSE: optimising it directly
discourages cliff destruction, with no hand-coded slope rule.

Self-critical estimator (low variance): baseline = the corrector run with the
deterministic (mean) globals, which is exactly what inference uses. The advantage
``reward_sample - reward_mean`` pushes the mean toward parameter regions that beat
the current greedy choice. Only the regime head receives gradient.

The head is fed both the data (pooled backbone features) and the model's own
outputs (per-scene prediction stats); see SegmentationModel._pred_stats.
"""

from __future__ import annotations

import math
from typing import List, Dict

import numpy as np
import torch

from .spag_dc import spag_dc_refine, SPAG_N_GLOBALS
from ..training.metrics import dtm_rmse_components, dtm_error_pixels


def _refine_errs(args):
    """Run the real corrector with given globals and return (per-pixel |error|, info).
    Per-cloud cost scales ~linearly with point count (e.g. ~0.7 s @20k, ~3 s @80k), so the
    reward subsample (max_points) is the dominant speed lever; it is accuracy-safe because the
    1 m DEM is resolution-limited and the advantage is a same-points difference (bias cancels)."""
    xyz, pred, gt, cfg, globals_, res = args
    ref, info = spag_dc_refine(xyz, pred, cfg, return_info=True, learned_globals=globals_)
    return _scene_errs(xyz, ref, gt, res), info


def _scene_errs(xyz: np.ndarray, pred: np.ndarray, gt: np.ndarray, res: float) -> np.ndarray:
    """Per-pixel |DEM error| (m) for one cloud; empty array if undefined."""
    try:
        return dtm_error_pixels(xyz, pred.astype(np.int64), gt.astype(np.int64), res)
    except Exception:
        return np.empty(0, dtype=np.float64)


def _agg(errs: np.ndarray, metric: str) -> float:
    """Aggregate per-pixel errors. 'rmse' = sqrt(mean d^2); 'p95'/'p99' = high percentile of
    |d|; 'max' = worst cell. Tail metrics weight cliffs/edges far more than mean RMSE."""
    if errs.size == 0:
        return float("inf")
    m = (metric or "rmse").lower()
    if m == "p95":
        return float(np.percentile(errs, 95))
    if m == "p99":
        return float(np.percentile(errs, 99))
    if m == "max":
        return float(errs.max())
    return float(np.sqrt(np.mean(errs * errs)))   # rmse (default)


def _scene_rmse(xyz: np.ndarray, pred: np.ndarray, gt: np.ndarray, res: float) -> float:
    """OpenGF DTM-RMSE (m) for one cloud: sqrt(mean d^2). inf if undefined."""
    return _agg(_scene_errs(xyz, pred, gt, res), "rmse")


def reinforce_update(model, scenes: List[Dict], optimizer, cfg,
                     sigma: float = 0.5, res: float = 1.0, metric: str = "p95") -> Dict[str, float]:
    """One self-critical REINFORCE step over a minibatch of cached scenes.

    Each scene dict holds: ``pooled`` (1, C) and ``pred_stats`` (1, S) detached
    tensors on the model's device; ``xyz`` (Np, 3) float; ``pred`` (Np,) raw
    predicted labels (1=ground); ``gt`` (Np,) GT labels.

    ``metric`` selects the reward aggregation over per-pixel DEM errors ('rmse',
    'p95', 'p99', 'max'); tail metrics target cliffs. Returns metrics including the
    plain DTM-RMSE (headline) and the optimised-metric values; the baseline is the
    deterministic (mean) params -- what inference uses.
    """
    device = model._spag_lo.device
    rows = []                                                          # (logp, g_mean, g_samp, xyz, pred, gt)
    for sc in scenes:
        pooled = sc["pooled"].to(device)
        pred_stats = sc["pred_stats"].to(device)
        xyz = np.asarray(sc["xyz"], dtype=np.float64)
        pred = np.asarray(sc["pred"], dtype=np.int64)
        gt = np.asarray(sc["gt"], dtype=np.int64)
        mean = model.regime_logits(pooled, pred_stats).float()          # (1, 6), grad -> head only
        dist = torch.distributions.Normal(mean, float(sigma))
        sample = dist.sample()
        logp = dist.log_prob(sample).sum()
        g_mean = model.squash_globals(mean.detach()).cpu().numpy().reshape(-1)
        g_samp = model.squash_globals(sample).cpu().numpy().reshape(-1)
        rows.append((logp, g_mean, g_samp, xyz, pred, gt))

    jobs = []
    for (_, g_mean, g_samp, xyz, pred, gt) in rows:
        jobs.append((xyz, pred, gt, cfg, g_mean, res))
        jobs.append((xyz, pred, gt, cfg, g_samp, res))
    out = [_refine_errs(j) for j in jobs]

    losses = []
    base_rmses, samp_rmses, base_scores, samp_scores, advs, reclass = [], [], [], [], [], []
    for i, (logp, _gm, _gs, _xyz, _pred, _gt) in enumerate(rows):
        errs_m, info_m = out[2 * i]
        errs_s, _info_s = out[2 * i + 1]
        score_m = _agg(errs_m, metric); score_s = _agg(errs_s, metric)
        if not (np.isfinite(score_m) and np.isfinite(score_s)):
            continue
        advantage = score_m - score_s                                  # >0 when the sample beats greedy
        losses.append(-(advantage) * logp)
        base_scores.append(score_m); samp_scores.append(score_s); advs.append(advantage)
        base_rmses.append(_agg(errs_m, "rmse")); samp_rmses.append(_agg(errs_s, "rmse"))
        ng = max(int(info_m.get("n_ground", 0)), 1)
        reclass.append(int(info_m.get("n_reclassified", 0)) / ng)

    if not losses:
        return {"loss": float("nan"), "rmse_base": float("nan"), "rmse_sample": float("nan"),
                "score_base": float("nan"), "score_sample": float("nan"), "metric": metric,
                "advantage": float("nan"), "reclass_frac": float("nan"), "n": 0}

    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach().cpu()),
            "rmse_base": float(np.mean(base_rmses)),
            "rmse_sample": float(np.mean(samp_rmses)),
            "score_base": float(np.mean(base_scores)),
            "score_sample": float(np.mean(samp_scores)),
            "metric": metric,
            "advantage": float(np.mean(advs)),
            "reclass_frac": float(np.mean(reclass)),
            "n": len(losses)}


def reinforce_loss_term(model, coord, pred, gt, cloud_lengths_0, cfg,
                        sigma: float = 0.5, res: float = 1.0,
                        max_points: int = 0, rng=None, metric: str = "p95"):
    """Differentiable REINFORCE surrogate for the regime head, to ADD to the training loss
    so the head co-trains with the backbone (no separate pass / optimizer).

    Trains ONLY the head: the policy mean is recomputed from the DETACHED cached pooled
    features + prediction stats (``model._regime_pooled`` / ``_regime_pred_stats``), so the
    RL gradient never flows into the backbone (the seg loss owns the backbone). Samples
    SPAG-DC globals, runs the real corrector, and scores sampled vs mean by ``metric`` over
    the per-pixel DEM error (self-critical baseline = deterministic/mean params). Points are
    sliced PER ORIGINAL CLOUD via ``cloud_lengths_0`` (correct under Mix3D) and subsampled to
    ``max_points`` (<=0 -> use all points). Returns (loss, metrics); loss is None when no
    cloud yielded a usable reward (caller then skips the term).
    """
    rng = rng if rng is not None else np.random.default_rng()
    pooled = getattr(model, "_regime_pooled", None)
    pstats = getattr(model, "_regime_pred_stats", None)
    if pooled is None or pstats is None:
        return None, {"n": 0}
    B = int(pooled.shape[0])
    if torch.is_tensor(cloud_lengths_0):
        c0 = cloud_lengths_0.detach().cpu().numpy().reshape(-1).astype(np.int64)
    else:
        c0 = np.asarray(cloud_lengths_0, dtype=np.int64).reshape(-1)
    if c0.size != B:
        return None, {"n": 0}
    starts = np.concatenate([[0], np.cumsum(c0)[:-1]]).astype(np.int64)
    coord = coord.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy().reshape(-1)
    gt = gt.detach().cpu().numpy().reshape(-1)

    logits_all = model.regime_logits(pooled, pstats).float()           # (B, 6), grad -> HEAD ONLY
    rows = []                                                          # (logp, g_mean, g_samp, xyz, pred, gt)
    for b in range(B):
        s = int(starts[b]); e = s + int(c0[b])
        if e - s < 64:
            continue
        idx = np.arange(s, e)
        if max_points and max_points > 0 and idx.size > max_points:
            idx = idx[rng.choice(idx.size, size=max_points, replace=False)]
        xyz_b = coord[idx].astype(np.float64)
        pred_b = pred[idx].astype(np.int64)
        gt_b = gt[idx].astype(np.int64)
        mean = logits_all[b:b + 1]
        dist = torch.distributions.Normal(mean, float(sigma))
        sample = dist.sample()
        logp = dist.log_prob(sample).sum()
        g_mean = model.squash_globals(mean.detach()).cpu().numpy().reshape(-1)
        g_samp = model.squash_globals(sample).cpu().numpy().reshape(-1)
        rows.append((logp, g_mean, g_samp, xyz_b, pred_b, gt_b))
    if not rows:
        return None, {"n": 0}

    # run all 2*len(rows) corrector evaluations concurrently (mean baseline + sample per cloud)
    jobs = []
    for (_, g_mean, g_samp, xyz_b, pred_b, gt_b) in rows:
        jobs.append((xyz_b, pred_b, gt_b, cfg, g_mean, res))
        jobs.append((xyz_b, pred_b, gt_b, cfg, g_samp, res))
    out = [_refine_errs(j) for j in jobs]

    terms, base, samp, bscore, sscore, advs, rec = [], [], [], [], [], [], []
    for i, (logp, _g_mean, _g_samp, _xyz, _pred, _gt) in enumerate(rows):
        errs_m, info_m = out[2 * i]
        errs_s, _info_s = out[2 * i + 1]
        score_m = _agg(errs_m, metric); score_s = _agg(errs_s, metric)
        if not (np.isfinite(score_m) and np.isfinite(score_s)):
            continue
        adv = score_m - score_s                                        # >0 when the sample beats greedy
        terms.append(-(adv) * logp)
        bscore.append(score_m); sscore.append(score_s); advs.append(adv)
        base.append(_agg(errs_m, "rmse")); samp.append(_agg(errs_s, "rmse"))
        ng = max(int(info_m.get("n_ground", 0)), 1)
        rec.append(int(info_m.get("n_reclassified", 0)) / ng)

    if not terms:
        return None, {"n": 0}
    loss = torch.stack(terms).mean()
    return loss, {"rmse_base": float(np.mean(base)), "rmse_sample": float(np.mean(samp)),
                  "score_base": float(np.mean(bscore)), "score_sample": float(np.mean(sscore)),
                  "metric": metric, "advantage": float(np.mean(advs)),
                  "reclass_frac": float(np.mean(rec)), "n": len(terms)}
