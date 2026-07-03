"""Tile storage shared by preprocessing (write) and the dataset (read).

The large per-point / per-candidate arrays are stored as **uncompressed ``.npy``
sidecars** next to a small uncompressed ``.npz``. Two reasons:

* ``.npy`` files can be **memory-mapped** (``np.load(..., mmap_mode='r')``); an
  ``.npz`` cannot. With mmap, all DataLoader workers share ONE OS-page-cache copy
  of a tile instead of each worker decompressing its own heap copy - so tile RAM
  no longer scales with ``num_workers``.
* Uncompressed means a "reload" is a page-cache hit (no zlib decompress), which is
  what was starving the GPU once the per-sphere feature compute was removed.

Slicing a memmap (``arr[idx]``) materialises only the touched rows, so the cached
entry stays tiny (just the mmap handles) while ``__getitem__`` still gets real
arrays. Legacy compressed ``.npz`` tiles (no sidecars) still load - just without
the mmap/RAM benefit - so the format is backward compatible.
"""
from __future__ import annotations

import os
import numpy as np

# Big arrays -> mmap-able .npy sidecars. Everything else (dtm raster, geo, origin,
# centers, membership, cand_off, split flags) stays small and lives in the .npz.
MMAP_KEYS = (
    "local", "labels", "returns", "intensity", "ret_ratio",
    "cand_idx", "feat_mean_elev", "feat_curvature", "dtm_height",
)


def _base(npz_path: str) -> str:
    return npz_path[:-4] if npz_path.endswith(".npz") else npz_path


def _sidecar(base: str, key: str) -> str:
    return f"{base}.{key}.npy"


def save_tile(out_path: str, arrays: dict) -> None:
    """Write a tile: big arrays -> uncompressed ``.npy`` sidecars (mmap-able), the
    remaining small arrays/metadata -> an uncompressed ``.npz`` at ``out_path``."""
    base = _base(out_path)
    small = {}
    for k, v in arrays.items():
        if k in MMAP_KEYS and v is not None:
            np.save(_sidecar(base, k), np.ascontiguousarray(v))
        else:
            small[k] = v
    np.savez(out_path, **small)                      # uncompressed: cheap, page-cache friendly


def load_tile(npz_path: str, mmap: bool = True) -> dict:
    """Load a tile into a plain dict. Big arrays come from ``.npy`` sidecars,
    memory-mapped when ``mmap`` is True; if a sidecar is missing (legacy tile) the
    array is taken from the ``.npz`` instead. Small arrays always come from the
    ``.npz``. Callers slice (and cast) per access so memmaps stay unmaterialised."""
    d = np.load(npz_path, allow_pickle=True)
    base = _base(npz_path)
    out = {k: d[k] for k in d.files}                 # small arrays (and legacy big ones)
    mode = "r" if mmap else None
    for k in MMAP_KEYS:
        side = _sidecar(base, k)
        if os.path.exists(side):
            out[k] = np.load(side, mmap_mode=mode)   # memmap: shared page cache, no decompress
    return out
