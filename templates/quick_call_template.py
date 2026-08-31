"""
Quick Call Template for cos_comparison.

Minimal, copy-paste ready template for direct API calls.
Edit the CONFIG section and run.

This is the simplest entry point: no class hierarchy, no pipeline,
just direct function calls with configurable parameters.
"""

import os
import sys
import json
from typing import Any, Dict, List, Optional, Tuple

# =====================================================================
# CONFIG (edit these values)
# =====================================================================
CONFIG: Dict[str, Any] = {
    # Backend: "cos_comparison_pydll" | "cos_comparison_c" | "cos_comparison"
    "backend": "cos_comparison_c",

    # Data: can be file paths or inline tensors
    "tensor_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # 1D example
    "tensor_b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "tensor_2d": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],

    # Passive mode
    "window_size": (2, 2),
    "step": (1, 1),
    "d": (1, 0),

    # Active mode
    "kernel": [[1.0, 0.0], [0.0, 1.0]],

    # Algorithm: "cos" | "mod" | "cosmod"
    "algorithm": "cos",

    # Output
    "output_file": "quick_result.json",
}


# =====================================================================
# SETUP
# =====================================================================
def setup(backend: str = "cos_comparison_c") -> None:
    """Initialize backend and import core APIs."""
    from cos_comparison.core import set_mode, get_mode
    try:
        set_mode([backend, "cos_comparison"])
    except Exception:
        set_mode("cos_comparison")
    print(f"[Setup] Backend: {get_mode()}")


def get_algo(name: str):
    from cos_comparison.core import cos_comparison as _cc
    return {"cos": _cc._cos, "mod": _cc._mod, "cosmod": _cc._cosmod}.get(name, _cc._cos)


# =====================================================================
# DIRECT CALLS
# =====================================================================
def demo_cosine() -> None:
    """Demo 1: Full tensor cosine similarity."""
    from cos_comparison.core import cos
    algo = get_algo(CONFIG["algorithm"])
    result = cos(CONFIG["tensor_a"], CONFIG["tensor_b"], algorithm=algo)
    print(f"[cos] similarity = {result:.6f}")


def demo_passive() -> None:
    """Demo 2: Passive mode (sliding window self-similarity)."""
    from cos_comparison.core import cos_comparison_passive
    algo = get_algo(CONFIG["algorithm"])
    result = cos_comparison_passive(
        CONFIG["tensor_2d"],
        window_size=CONFIG["window_size"],
        step=CONFIG["step"],
        d=CONFIG["d"],
        algorithm=algo,
    )
    # result may be vector_map_as_tensor or list; flatten for display
    flat = []
    def _flatten(o):
        if isinstance(o, (list, tuple)):
            for x in o: _flatten(x)
        elif hasattr(o, "__iter__") and not isinstance(o, (str, bytes)):
            for x in o: _flatten(x)
        else:
            try: flat.append(float(o))
            except (TypeError, ValueError): pass
    _flatten(result)
    print(f"[passive] {len(flat)} output values, first = {flat[0] if flat else 0:.4f}, max = {max(flat) if flat else 0:.4f}")


def demo_active() -> None:
    """Demo 3: Active mode (template matching)."""
    from cos_comparison.core import cos_comparison_active
    algo = get_algo(CONFIG["algorithm"])
    result = cos_comparison_active(
        CONFIG["tensor_2d"],
        kernel=CONFIG["kernel"],
        step=CONFIG["step"],
        algorithm=algo,
    )
    flat = []
    def _flatten(o):
        try:
            flat.append(float(o))
            return
        except (TypeError, ValueError):
            pass
        if isinstance(o, (list, tuple)):
            for x in o: _flatten(x)
        elif hasattr(o, "__iter__") and not isinstance(o, (str, bytes)):
            for x in o: _flatten(x)
    _flatten(result)
    print(f"[active] {len(flat)} values, max = {max(flat) if flat else 0:.6f}, min = {min(flat) if flat else 0:.6f}")


def demo_statistics() -> None:
    """Demo 4: Local mean and variance."""
    from cos_comparison.core import mean_local, local_variance
    mean = mean_local(CONFIG["tensor_2d"], local_size=(2, 2), step=(1, 1))
    var = local_variance(CONFIG["tensor_2d"], local_size=(2, 2), step=(1, 1))
    print(f"[mean_local] first = {mean[0][0] if mean and mean[0] else 0:.4f}")
    print(f"[variance] first = {var[0][0] if var and var[0] else 0:.4f}")


def demo_tensor_class() -> None:
    """Demo 5: vector_map_as_tensor operations."""
    from cos_comparison.core import vector_map_as_tensor
    t = vector_map_as_tensor(
        vector=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape=(2, 3),
        strides=(3, 1),
    )
    print(f"[tensor] shape={t.__shape__()}, mean={t.mean():.4f}, variance={t.variance():.4f}")
    print(f"[tensor] t[0,1]={t[0, 1]}, len={len(t)}")


def demo_threshold() -> None:
    """Demo 6: Threshold filter and map."""
    from cos_comparison.core import threshold_filter, threshold_map
    filtered = threshold_filter(CONFIG["tensor_2d"], low=2.0, high=7.0)
    mapped = threshold_map(
        CONFIG["tensor_2d"],
        pairs=[(0.0, 3.0, 0.0), (3.0, 6.0, 1.0), (6.0, 10.0, 2.0)],
        default_value=-1.0,
    )
    print(f"[threshold_filter] non-zero count = {sum(1 for row in filtered for v in row if v != 0)}")
    print(f"[threshold_map] result = {mapped}")


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    print("=== cos_comparison Quick Call Template ===\n")
    setup(CONFIG["backend"])
    print()

    demo_cosine()
    demo_passive()
    demo_active()
    demo_statistics()
    demo_tensor_class()
    demo_threshold()

    print("\n=== All demos complete ===")


if __name__ == "__main__":
    main()
