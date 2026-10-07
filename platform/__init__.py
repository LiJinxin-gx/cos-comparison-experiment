"""
platform -- dedicated open-integration exploration platform for cos_comparison.

Core identity: extraction and generation are the forward / reverse of one local
comparison relation; closure is the universal criterion. Runs standalone
(``python -m platform``) or imports as a module. I/O (io_pairs) is decoupled
from processing (relation_core); measurement backends auto-select
(cos_comparison C -> NumPy -> pure Python).
"""
from .backends import change_field, backend_name, has_numpy, formal_core
from .io_pairs import save_field, load_field, fit_symbols, decode_symbols
from .relation_core import (run_continuous, run_discrete,
                            generate_continuous, generate_discrete,
                            relation_graph, box_matrix)
from .heatmap import save_png

__all__ = [
    "change_field", "backend_name", "has_numpy", "formal_core",
    "save_field", "load_field", "fit_symbols", "decode_symbols",
    "run_continuous", "run_discrete", "generate_continuous",
    "generate_discrete", "relation_graph", "box_matrix", "save_png",
]


def _selftest():
    import numpy as np
    print("platform backend:", backend_name())
    n = 20; c0 = (n - 1) / 2.0
    yy, xx = np.indices((n, n))
    d = np.hypot(xx - c0, yy - c0)
    field = ((d >= 4.0) & (d <= 7.0)).astype(float)
    rc = run_continuous(field)
    print("continuous: mean|dE|=%.6f max=%.6f -> %s"
          % (rc["mean_residual"], rc["max_residual"],
             "PASS" if rc["pass"] else "CHECK"))
    symbols = "the cat and the dog and the cat sat and the dog ran".split()
    rd = run_discrete(symbols)
    print("discrete: grounded=%.1f%% novel=%s -> %s"
          % (100 * rd["grounded_fraction"], rd["novel"],
             "PASS" if rd["pass"] else "CHECK"))


if __name__ == "__main__":
    _selftest()
