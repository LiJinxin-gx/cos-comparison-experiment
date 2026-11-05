# -*- coding: utf-8 -*-
"""
Minimal successful principle: the local-comparison core in pure stdlib.

Continuous mapping group:
  down = f_{i,i+1} (coarsen, average pool)
  up   = f_{i+1,i} (structural inverse, nearest)

Primitive isolation (never average active+passive):
  active  = raw values
  passive = local differences (gap space, a DIFFERENT data type)

Level isolation: each level stores features independently, no value flow.
Fusion at decision layer by max, not weighted average.

Three similarity families (same-primitive only), matching the formal core:
  cos    = dot / (||a|| ||b||)                 direction
  mod    = 2||a|| ||b|| / (||a||^2 + ||b||^2) magnitude
  cosmod = 2 dot / (||a||^2 + ||b||^2)         combined (the default)

The principle primitives run on the installed cos_comparison engine
(core.pool, core.cos_comparison_passive, core.cos); only the structural
up-sampling and flatten helpers remain local.
"""
from cos_comparison import core


def down(t, f):
    """continuous coarsening f_{i,i+1} via the formal average pool."""
    h, w = len(t), len(t[0])
    output = core.create_void_list((h // f, w // f))
    core.pool(t, window_size=(f, f), step=(f, f), output=output)
    return output


def up(t, H, W):
    """structural inverse f_{i+1,i} (nearest, not exact)."""
    h, w = len(t), len(t[0])
    return [[t[min(i*h//H, h-1)][min(j*w//W, w-1)] for j in range(W)] for i in range(H)]


def passive(t):
    """Passive d-shift boundary map via the formal engine (gap space)."""
    h, w = len(t), len(t[0])
    output = core.create_void_list((h - 1, w))
    core.cos_comparison_passive(t, window_size=(1, 1),
                                d=(1, 0), output=output)
    return output


def cos(a, b):
    """same-primitive cosine via the formal engine."""
    return core.cos(a, b, algorithm=core._cos)


def mod(a, b):
    """Magnitude similarity via the formal engine (good for sparse data)."""
    return core.cos(a, b, algorithm=core._mod)


def cosmod(a, b):
    """Combined direction-and-magnitude similarity: the formal default."""
    return core.cos(a, b, algorithm=core._cosmod)


def flat(t):
    return [x for row in t for x in row]


# Self-verification: the three similarities must agree with their closed
# form and satisfy their basic identities.
if __name__ == "__main__":
    import math
    x = [1.0, 2.0, 3.0, 4.0]
    y = [4.0, 5.0, 6.0, 7.0]

    # Identical vectors score 1 under every family.
    assert abs(cos(x, x) - 1.0) < 1e-12
    assert abs(mod(x, x) - 1.0) < 1e-12
    assert abs(cosmod(x, x) - 1.0) < 1e-12

    # Cross-check against the closed-form formulas.
    d = sum(u*v for u, v in zip(x, y))
    nx2 = sum(u*u for u in x)
    ny2 = sum(v*v for v in y)
    nx, ny = math.sqrt(nx2), math.sqrt(ny2)
    assert abs(cos(x, y) - d/(nx*ny)) < 1e-12
    assert abs(mod(x, y) - 2*nx*ny/(nx2+ny2)) < 1e-12
    assert abs(cosmod(x, y) - 2*d/(nx2+ny2)) < 1e-12

    # Structural inverse round-trip keeps the shape.
    t = [[float(i + j) for j in range(4)] for i in range(4)]
    coarse = down(t, 2)
    back = up(coarse, 4, 4)
    assert len(back) == 4 and len(back[0]) == 4
    # Formal passive boundary of a 4x4 tensor is 3x4 (d-shift drops one row).
    assert len(flat(passive(t))) == 12

    print("minimal principle verified: cos/mod/cosmod + down/up + passive")

