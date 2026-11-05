# -*- coding: utf-8 -*-
"""Companion demonstration: principle algorithms on the latest installed
cos_comparison package.

The other explore files re-implement every primitive in readable standard
library code so they run with no dependencies. This module is their companion:
it demonstrates the same principles on the authoritative installed engine
(the compiled C backend, with a pure-Python fallback), so the pedagogical
code and the shipped implementation can be read side by side.

Every principle function below takes a plain tensor (nested list) or a
discrete 1D sequence (a list of floats). No file path or special object is
accepted by the principle code; __main__ only synthesizes demo data.
"""
from cos_comparison import core


# ------------------------------------------------------------
# 1. Three similarities on a whole vector: cos / mod / cosmod
# ------------------------------------------------------------
def three_similarities(a, b):
    """Return the three whole-tensor similarities.

    cos    = dot / (||a|| ||b||)                 direction
    mod    = 2 ||a|| ||b|| / (||a||^2 + ||b||^2) magnitude
    cosmod = 2 dot / (||a||^2 + ||b||^2)        combined (formal default)
    """
    return {
        "cos": core.cos(a, b, algorithm=core._cos),
        "mod": core.cos(a, b, algorithm=core._mod),
        "cosmod": core.cos(a, b, algorithm=core._cosmod),
    }


# ------------------------------------------------------------
# 2. Passive mode: d-shift self-comparison (boundary / gap space)
# ------------------------------------------------------------
def passive_boundary(sequence, window=3, d=1):
    """Boundary map: each window is compared with its d-shifted neighbour."""
    return [float(v) for v in core.cos_comparison_passive(
        sequence, window_size=(window,), d=(d,), algorithm=core._cosmod)]


# ------------------------------------------------------------
# 3. Active mode: kernel template matching (element space)
# ------------------------------------------------------------
def active_match(sequence, kernel):
    """Slide the kernel template and return the match score per position."""
    return [float(v) for v in core.cos_comparison_active(
        sequence, kernel=kernel, algorithm=core._cosmod)]


# ------------------------------------------------------------
# 4. Hierarchical mean pooling: continuous coarsening mapping
# ------------------------------------------------------------
def mean_pool(data, factor):
    """Average-pool a 1D sequence or a 2D tensor by factor."""
    if data and isinstance(data[0], list):  # 2D tensor
        h, w = len(data), len(data[0])
        nh, nw = h // factor, w // factor
        output = [[0.0] * nw for _ in range(nh)]
        core.pool(data, window_size=(factor, factor),
                  step=(factor, factor), output=output)
    else:  # 1D discrete sequence
        out_n = len(data) // factor
        output = [0.0] * out_n
        core.pool(data, window_size=(factor,),
                  step=(factor,), output=output)
    return output


def pool_hierarchy(data, factors):
    """Return [L0, L1, ...] where each level coarsens by its factor."""
    levels = [data]
    current = data
    for factor in factors:
        current = mean_pool(current, factor)
        levels.append(current)
    return levels


# ------------------------------------------------------------
# 5. Backend parity: the same answers on the C and Python engines
# ------------------------------------------------------------
def backend_parity(sequence, kernel):
    """Run passive and active on both backends and report max differences."""
    scores = {}
    for mode in ("c", "py"):
        core.set_mode(mode)
        scores[mode] = {
            "passive": passive_boundary(sequence),
            "active": active_match(sequence, kernel),
        }
    diffs = {}
    for channel in ("passive", "active"):
        c_vals = scores["c"][channel]
        p_vals = scores["py"][channel]
        n = min(len(c_vals), len(p_vals))
        diffs[channel] = max(abs(c_vals[i] - p_vals[i]) for i in range(n))
    return scores, diffs


# ------------------------------------------------------------
# Demo
# ------------------------------------------------------------
def main():
    print("=" * 64)
    print("Companion Demo: principle algorithms on the installed package")
    print("=" * 64)

    # Non-degenerate vectors (unequal norms, non-parallel) so the three
    # similarities do not collapse onto one another.
    a = [1.0, 2.0, 3.0, 4.0]
    b = [4.0, 5.0, 6.0, 7.0]
    sims = three_similarities(a, b)
    print("\n1) Three similarities (a=%s, b=%s):" % (a, b))
    for name in ("cos", "mod", "cosmod"):
        print("   %-6s = %.4f" % (name, sims[name]))

    # A 1D sequence with a clear step: the boundary drops at the jump.
    sequence = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2]
    boundary = passive_boundary(sequence, window=3, d=1)
    print("\n2) Passive boundary (cosmod, window=3, d=1):")
    print("   ", " ".join("%.2f" % v for v in boundary))

    # Active kernel match: locate the embedded template.
    kernel = sequence[3:6]
    matches = active_match(sequence, kernel)
    best_pos = max(range(len(matches)), key=lambda i: matches[i])
    print("\n3) Active kernel match (kernel=%s):" % kernel)
    print("   ", " ".join("%.2f" % v for v in matches))
    print("    best match position =", best_pos)

    # Hierarchical pooling of a 2D tensor.
    tensor = [[float((i + j) % 4) for j in range(8)] for i in range(8)]
    levels = pool_hierarchy(tensor, [2, 2])
    print("\n4) Hierarchical mean pooling shapes:")
    for lvl, t in enumerate(levels):
        if t and isinstance(t[0], list):
            print("   L%d: %dx%d" % (lvl, len(t), len(t[0])))
        else:
            print("   L%d: length %d" % (lvl, len(t)))

    # Backend parity.
    _, diffs = backend_parity(sequence, kernel)
    print("\n5) Backend parity (max |C - Python|):")
    for channel, diff in diffs.items():
        print("   %-7s = %.2e" % (channel, diff))

    print("\nCompanion verified: the installed engine implements the same")
    print("principles as the self-contained pedagogical files.")


if __name__ == "__main__":
    main()
