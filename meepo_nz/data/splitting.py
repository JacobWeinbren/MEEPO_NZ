"""Runtime train/val/test re-splitting (no re-preprocessing needed).

The split stored in each tile ``.npz`` was assigned once at stage-04 time. When
a persistent train/val composition gap emerges (e.g. the seeded permutation
landed an unrepresentative val set -- visible as diverging ground-fraction
"[diag]" lines), ``--resplit-seed N`` overrides the stored split AT LOAD TIME
with a deterministic per-cloud hash:

    u = md5(f"{seed}:{cloud_stem}")  ->  uniform in [0, 1)
    u < test_frac                -> test
    u < test_frac + val_frac     -> val
    else                         -> train

Properties:
  * order-independent (no dependence on directory listing or filename sort --
    OS-grid names sort geographically, a permutation over a sorted list does
    not, but a hash removes the question entirely);
  * stable across runs, resumes, and machines for the same seed;
  * per-CLOUD granularity (the tile stem is the cloud stem), preserving the
    stage-04 guarantee that overlapping tiles of one cloud never straddle
    splits;
  * changing the seed changes val AND test -- pick a seed, check the [diag]
    class-balance lines roughly agree between train and val, then FREEZE it
    for all subsequent runs of that dataset.
"""
from __future__ import annotations

import hashlib
import os


def hash_split(stem: str, seed: int, val_frac: float = 0.1,
               test_frac: float = 0.1) -> str:
    h = hashlib.md5(f"{int(seed)}:{stem}".encode("utf-8")).hexdigest()
    u = int(h[:12], 16) / float(1 << 48)
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def effective_split(path: str, stored_split: str, cfg) -> str:
    """Stored split, unless cfg.resplit_seed is set -> deterministic hash split."""
    seed = getattr(cfg, "resplit_seed", None)
    if seed is None:
        return stored_split
    stem = os.path.splitext(os.path.basename(path))[0]
    return hash_split(stem, int(seed),
                      val_frac=float(getattr(cfg, "resplit_val_frac", 0.1)),
                      test_frac=float(getattr(cfg, "resplit_test_frac", 0.1)))
