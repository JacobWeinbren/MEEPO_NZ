"""Tiny progress helper: use tqdm when installed, else a periodic stderr bar.

Drop-in for ``for x in progress(seq, total=n, desc="..."): ...`` so callers do
not gain a hard dependency on tqdm.
"""
from __future__ import annotations

import sys


def progress(iterable, total=None, desc="", every=0.04):
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    except Exception:
        return _Fallback(iterable, total=total, desc=desc, every=every)


class _Fallback:
    def __init__(self, iterable, total=None, desc="", every=0.04):
        self.it = iter(iterable)
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)
        self.total = total
        self.desc = desc
        self.n = 0
        self._mark = 0
        self._step = max(1, int((total or 0) * every)) if total else 200

    def __iter__(self):
        return self

    def __next__(self):
        try:
            v = next(self.it)
        except StopIteration:
            if self.total:
                sys.stderr.write(f"\r{self.desc} {self.total}/{self.total} (100%)\n")
                sys.stderr.flush()
            raise
        self.n += 1
        if self.n >= self._mark:
            self._mark += self._step
            if self.total:
                pct = 100.0 * self.n / self.total
                sys.stderr.write(f"\r{self.desc} {self.n}/{self.total} ({pct:4.1f}%)")
            else:
                sys.stderr.write(f"\r{self.desc} {self.n}")
            sys.stderr.flush()
        return v
