"""
platform.io_pairs -- matched read/write pairs.

Read and write are exact inverses, so the data kind and shape stay semantically
symmetric. I/O lives here; processing never touches these formats directly.
"""
from __future__ import annotations
import io
from .backends import has_numpy

DEPENDENCIES = {
    "python": ">=3.8",
    "requires": {"numpy": ">=1.20"},
    "optional": {"cos_comparison": "~=0.5"},
    "backends": ["formal", "numpy"],
    "status": "platform",
    "targets": "matched read/write pairs",
}


def save_field(field) -> bytes:
    import numpy as np
    buf = io.BytesIO()
    np.save(buf, np.asarray(field, dtype=float))
    return buf.getvalue()


def load_field(blob: bytes):
    import numpy as np
    return np.load(io.BytesIO(blob), allow_pickle=False)


class _FirstSeenMap:
    def __init__(self):
        self.table = {}
    def add(self, symbols):
        for s in symbols:
            if s not in self.table:
                self.table[s] = float(len(self.table))
    def decode(self, ids):
        inv = {v: k for k, v in self.table.items()}
        return [inv[int(i)] for i in ids]


def _fit_unitmap(symbols):
    from cos_comparison.interface.tools.math_tool.unit_map import UnitMap
    um = UnitMap(); um.add(list(symbols))
    return um


def fit_symbols(symbols):
    try:
        um = _fit_unitmap(symbols)
        return [int(um.table[s]) for s in symbols], um
    except Exception:
        um = _FirstSeenMap(); um.add(list(symbols))
        return [int(um.table[s]) for s in symbols], um


def decode_symbols(ids, mapper):
    return mapper.decode(list(ids))
