"""Point-cloud container and module plumbing for the clean-PyTorch MEEPO.

This is a dependency-light re-implementation of Pointcept's ``Point`` /
``PointModule`` / ``PointSequential`` (``pointcept/models/utils/structure.py``
and ``modules.py``).  The upstream version stores a ``spconv.SparseConvTensor``
on every ``Point`` and routes spconv modules specially inside
``PointSequential``.  We drop spconv entirely: the only sparse-conv operation
PTv3 uses (the ``SubMConv3d`` conditional-positional-encoding and the embedding
stem) is provided here by a pure-PyTorch submanifold convolution
(``submanifold_conv.py``) that reads ``grid_coord``/``batch``/``feat`` directly,
so no ``SparseConvTensor`` is needed.  Likewise we replace ``addict.Dict`` with a
tiny attribute dict so the package has no addict dependency.

The serialization logic (``Point.serialization``) is faithful to PTv3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from collections import OrderedDict

from .serialization import encode


# --------------------------------------------------------------------------- #
# offset <-> batch helpers (verbatim from pointcept/models/utils/misc.py)
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class AttrDict(dict):
    """Minimal stand-in for ``addict.Dict``: attribute access over a dict."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:  # pragma: no cover - mirrors addict semantics
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(key) from e


class Point(AttrDict):
    """A batched point cloud as an attribute dict.

    Expected keys (subset used by the clean MEEPO):

      - ``coord``       : (N, 3) float, original (tile-local) coordinates;
      - ``grid_coord``  : (N, 3) int,  voxel coordinates (from GridSampling);
      - ``feat``        : (N, C) float, per-point features (model input/state);
      - ``offset``      : (B,)  long,  cumulative point counts (PTv3 convention);
      - ``batch``       : (N,)  long,  per-point batch id (derived from offset).

    After ``serialization()`` it also holds ``serialized_code/order/inverse`` and
    ``serialized_depth``; after ``SerializedPooling`` it holds ``pooling_parent``
    and ``pooling_inverse`` for the matching unpooling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    @torch.compiler.disable   # data-dependent int ops (bit_length, 1<<depth) are not Dynamo-traceable; run eager
    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """Compute space-filling-curve order(s) over the voxel grid.

        Faithful to PTv3: build one sort code per requested ``order`` (e.g. z,
        z-trans, hilbert, hilbert-trans), argsort to get the linear order, and
        the scatter-based inverse permutation.  ``shuffle_orders`` randomly
        permutes which order each attention block uses (PTv3 default True).
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order_ = torch.argsort(code)
        inverse = torch.zeros_like(order_).scatter_(
            dim=1,
            index=order_,
            src=torch.arange(0, code.shape[1], device=order_.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order_ = order_[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order_
        self["serialized_inverse"] = inverse


# --------------------------------------------------------------------------- #
# Module containers
# --------------------------------------------------------------------------- #
class PointModule(nn.Module):
    """Base class: subclasses consume and return a ``Point`` inside ``PointSequential``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    """Sequential container that threads a ``Point`` (or a plain tensor) through
    its children, transparently feeding ``point.feat`` to plain ``nn.Module``s
    and the whole ``Point`` to ``PointModule``s.  No spconv special-casing."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for _ in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for _, module in self._modules.items():
            if isinstance(module, PointModule):
                input = module(input)
            else:
                # plain nn.Module: operate on the feature tensor when given a Point
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                else:
                    input = module(input)
        return input
