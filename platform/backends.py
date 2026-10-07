"""
platform.backends -- open-integration measurement backends.

The platform selects, in order:
  1. the installed cos_comparison C-accelerated passive comparison (formal),
  2. a NumPy reference implementation,
  3. a pure-Python reference.
"""
from __future__ import annotations

DEPENDENCIES = {
    "python": ">=3.8",
    "requires": {},
    "optional": {"numpy": ">=1.20", "cos_comparison": "~=0.5"},
    "backends": ["formal", "numpy", "python"],
    "status": "platform",
    "targets": "version-independent local measurement",
}

_NP = None
_FORMAL = None


def has_numpy() -> bool:
    global _NP
    if _NP is None:
        try:
            import numpy  # noqa: F401
            _NP = True
        except Exception:
            _NP = False
    return _NP


def formal_core():
    global _FORMAL
    if _FORMAL is None:
        try:
            from cos_comparison import core  # type: ignore
            _FORMAL = core
        except Exception:
            _FORMAL = False
    return _FORMAL or None


def _formal_change(x, w: int):
    import numpy as np
    core = formal_core()
    g = np.asarray(x, dtype=float).tolist()
    fh = core.cos_comparison_passive_2d(g, (w, w), d=(0, 1))
    fv = core.cos_comparison_passive_2d(g, (w, w), d=(1, 0))
    n = len(g); m = n - w
    out = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            out[i, j] = ((1 - core.get_item(fh, (i, j))) +
                         (1 - core.get_item(fv, (i, j)))) / 2.0
    return out


def _numpy_change(x, w: int):
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    a = np.asarray(x, dtype=float)
    win = sliding_window_view(a, (w, w))
    def cos_pairs(p, q):
        pp = p.reshape(*p.shape[:2], -1).astype(float)
        qq = q.reshape(*q.shape[:2], -1).astype(float)
        dot = (pp * qq).sum(-1)
        np_ = np.sqrt((pp * pp).sum(-1) * (qq * qq).sum(-1))
        cos = np.ones_like(dot)
        ok = np_ > 1e-12
        cos[ok] = dot[ok] / np_[ok]
        both_zero = ((pp * pp).sum(-1) == 0) & ((qq * qq).sum(-1) == 0)
        cos[both_zero] = 1.0
        return cos
    ch_h = 1.0 - cos_pairs(win[:-1, :], win[1:, :])
    ch_w = 1.0 - cos_pairs(win[:, :-1], win[:, 1:])
    mh, mw = ch_h.shape[0], ch_w.shape[1]
    m = min(mh, mw)
    return (ch_h[:m, :m] + ch_w[:m, :m]) / 2.0


def _py_change(x, w: int):
    a = [list(map(float, row)) for row in x]
    n = len(a); m = n - w
    def patch(i, j):
        return [a[i + u][j + v] for u in range(w) for v in range(w)]
    def cos(p, q):
        dot = sum(u * v for u, v in zip(p, q))
        bp = sum(u * u for u in p); bq = sum(v * v for v in q)
        if bp == 0 and bq == 0:
            return 1.0
        if bp * bq <= 0:
            return 0.0
        return dot / ((bp * bq) ** 0.5)
    out = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            cv = cos(patch(i, j), patch(i + 1, j)) if i + 1 < n - w + 1 else 1.0
            ch = cos(patch(i, j), patch(i, j + 1)) if j + 1 < n - w + 1 else 1.0
            out[i][j] = ((1 - cv) + (1 - ch)) / 2.0
    if has_numpy():
        import numpy as np
        return np.asarray(out)
    return out


def change_field(x, w: int = 3, backend: str | None = None):
    if backend == "formal":
        return _formal_change(x, w)
    if backend == "numpy":
        return _numpy_change(x, w)
    if backend == "python":
        return _py_change(x, w)
    if formal_core() is not None:
        try:
            return _formal_change(x, w)
        except Exception:
            pass
    if has_numpy():
        return _numpy_change(x, w)
    return _py_change(x, w)


def backend_name() -> str:
    if formal_core() is not None:
        return "formal(cos_comparison C)"
    if has_numpy():
        return "numpy"
    return "python"
