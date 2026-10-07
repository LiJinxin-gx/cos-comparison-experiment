"""
platform.relation_core -- the universal extraction/generation processing.

Processing is decoupled from I/O: functions receive tensors / symbol sequences
and return generated data with a closure verdict.
"""
from __future__ import annotations
from .backends import change_field, has_numpy
from .io_pairs import fit_symbols, decode_symbols

DEPENDENCIES = {
    "python": ">=3.8",
    "requires": {"numpy": ">=1.20"},
    "optional": {"cos_comparison": "~=0.5"},
    "backends": ["formal", "numpy"],
    "status": "platform",
    "targets": "extraction/generation inverse with closure",
}


def box_matrix(n: int, p: int = 3):
    import numpy as np
    r = n - p + 1
    F = np.zeros((r * r, n * n), dtype=float)
    for cy in range(r):
        for cx in range(r):
            row = cy * r + cx
            for a in range(p):
                for b in range(p):
                    F[row, (cy + a) * n + (cx + b)] = 1.0 / (p * p)
    return F


def generate_continuous(field, iters: int = 80, p: int = 3):
    import numpy as np
    x0 = np.asarray(field, dtype=float); n = x0.shape[0]
    F = box_matrix(n, p)
    x = np.clip(x0.reshape(-1), 1e-6, None)
    target = F @ x
    Ft_r = F.T @ target
    for _ in range(iters):
        have = F @ x
        den = F.T @ have
        x = x * np.where(den > 1e-12, Ft_r / np.where(den > 1e-12, den, 1.0), 1.0)
    return x.reshape(n, n)


def run_continuous(field, window: int = 3, iters: int = 80):
    import numpy as np
    x = np.asarray(field, dtype=float)
    relation = np.asarray(change_field(x, window), dtype=float)
    generated = generate_continuous(x, iters=iters)
    re_relation = np.asarray(change_field(generated, window), dtype=float)
    m = min(relation.shape[0], re_relation.shape[0])
    residual = np.abs(relation[:m, :m] - re_relation[:m, :m])
    return {
        "field": x, "generated": generated, "relation": relation,
        "re_relation": re_relation, "residual": residual,
        "mean_residual": float(residual.mean()),
        "max_residual": float(residual.max()),
        "pass": bool(residual.mean() < 1e-3),
    }


def relation_graph(symbols):
    ids, mapper = fit_symbols(symbols)
    return {"ids": ids, "mapper": mapper,
            "connectors": set(zip(ids, ids[1:]))}


def generate_discrete(symbols):
    ids, _ = fit_symbols(symbols)
    positions = {}
    for i, s in enumerate(ids[:-1]):
        positions.setdefault(s, []).append(i)
    pivot = next((s for s, ps in positions.items()
                  if len(ps) >= 2 and len({ids[i + 1] for i in ps}) >= 2), None)
    if pivot is None:
        return list(symbols), False
    p, q = positions[pivot][0], positions[pivot][-1]
    generated = list(symbols[:p + 1]) + list(symbols[q + 1:])
    return generated, "|".join(generated) != "|".join(symbols)


def run_discrete(symbols):
    graph = relation_graph(symbols)
    generated, novel = generate_discrete(symbols)
    g_ids, _ = fit_symbols(generated)
    edges = list(zip(g_ids, g_ids[1:]))
    support = [1.0 if e in graph["connectors"] else 0.0 for e in edges]
    frac = sum(support) / max(1, len(support))
    return {
        "symbols": list(symbols), "generated": generated,
        "edge_support": support, "grounded_fraction": frac,
        "novel": novel, "pass": bool(frac == 1.0 and novel),
    }
