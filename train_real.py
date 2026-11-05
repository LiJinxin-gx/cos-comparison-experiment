# -*- coding: utf-8 -*-
"""Train real classification with group-theoretic continuous mapping.

The principle primitives run on the installed cos_comparison engine
(core.pool, core.cos_comparison_passive, core.cos); only the structural
flatten helper and data synthesis remain local. Every primitive takes a
plain tensor (2D nested list) or a discrete sequence (1D list).
"""
import random

from cos_comparison import core


def flat(t):
    return [x for row in t for x in row]


def down(t, f):
    """Average-pool coarsening f_{i,i+1} via the formal engine."""
    h, w = len(t), len(t[0])
    output = core.create_void_list((h // f, w // f))
    core.pool(t, window_size=(f, f), step=(f, f), output=output)
    return output


def passive(t):
    """Passive d-shift boundary map via the formal engine (gap space).

    A tensor with no neighbour on the shift axis cannot have a boundary;
    the formal engine rejects it, so return the degenerate zero map (the
    same "no difference" result a lone element would carry).
    """
    h, w = len(t), len(t[0])
    if h < 2:
        return [[0.0] * w for _ in range(h)]
    output = core.create_void_list((h - 1, w))
    core.cos_comparison_passive(t, window_size=(1, 1),
                                d=(1, 0), output=output)
    return output


def cos(a, b):
    """Direction similarity via the formal engine (structural alignment)."""
    return core.cos(a, b, algorithm=core._cos)


def mod(a, b):
    """Magnitude similarity via the formal engine (norm only)."""
    return core.cos(a, b, algorithm=core._mod)


def cosmod(a, b):
    """Combined direction-and-magnitude similarity, the formal default."""
    return core.cos(a, b, algorithm=core._cosmod)


SZ = 14


def digit(d, noise=0.0):
    """draw digit d as 14x14 dot matrix."""
    img = [[0.0]*SZ for _ in range(SZ)]
    if d == 0:
        pts = [(1+i, 4+j) for i in range(7) for j in range(6)
               if (i == 0 or i == 6 or j == 0 or j == 5)]
    elif d == 1:
        pts = [(1+i, 4+j) for i in range(7) for j in range(6) if i % 2 == 0]
    else:
        pts = [(1+i, 4+j) for i in range(7) for j in range(6) if i == j]
    for i, j in pts:
        img[i][j] = 200.0
    if noise:
        for i in range(SZ):
            for j in range(SZ):
                img[i][j] += random.gauss(0, noise)
    return img


def levels(img):
    """L0, L1, L2 active; plus passive channel.

    Each level is reduced relative to the ORIGINAL image (1, 1/2, 1/4), so
    every level keeps a genuine 2D mapping rather than collapsing to a single
    scalar. A 1x1 level cannot carry a spatial pattern: cosine of positive
    scalars is always 1 and would trivially match every prototype.
    """
    img_l1 = down(img, 2)
    img_l2 = down(img, 4)
    la = [flat(img), flat(img_l1), flat(img_l2)]
    lp = [flat(passive(img)), flat(passive(img_l1)), flat(passive(img_l2))]
    return la, lp


def main():
    random.seed(42)
    train = {}
    for d in range(3):
        train[d] = levels(digit(d))
    ok = 0
    n = 30
    for d in range(3):
        for _ in range(n):
            ql, qp = levels(digit(d, noise=10))
            best, bl = -1, 0.0
            for k in range(3):
                # Direction (cosine) is the correct structural measure: two
                # patterns with the same energy but different layouts must not
                # match. mod (magnitude only) would confuse them, so it is not
                # used for the decision.
                a = max(cos(ql[i], train[k][0][i]) for i in range(3))
                p = max(cos(qp[i], train[k][1][i]) for i in range(3))
                s = max(a, p)
                if s > bl: bl, best = s, k
            if best == d: ok += 1
    print("accuracy: %.1f%%" % (100.0*ok/(3*n)))


if __name__ == "__main__":
    main()
