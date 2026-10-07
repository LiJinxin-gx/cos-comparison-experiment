"""
candidates.cand_context_states -- context states as constraint-satisfaction.

Promising, immature (status: candidate). Ports the self-emergent context-state
idea to the platform with fully self-contained data, then pushes it toward
generation: coherent content is an arrangement whose every local context is a
recorded state (an epsilon-machine / constraint-satisfaction view). Branch
points without a unique satisfying state are abstained (three-valued).

Bottleneck targeted: coherent generation without learned weights.
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from platform import change_field

DEPENDENCIES = {
    "python": ">=3.8",
    "requires": {"numpy": ">=1.20"},
    "optional": {"cos_comparison": "~=0.5"},
    "backends": ["formal", "numpy"],
    "status": "candidate",
    "targets": "coherent generation via contextual constraint satisfaction",
}


def make_shapes(n=16):
    yy, xx = np.indices((n, n)); c = (n - 1) / 2.0
    d = np.hypot(xx - c, yy - c)
    ring = ((d >= 3) & (d <= 5)).astype(float)
    cross = ((np.abs(xx - c) <= 1) | (np.abs(yy - c) <= 1)).astype(float)
    block = ((np.abs(xx - c) <= 3) & (np.abs(yy - c) <= 3)).astype(float)
    data, labels = [], []
    for cls, base in enumerate((ring, cross, block)):
        for v in range(8):
            img = base.copy()
            if v % 3 == 0:
                img = np.roll(img, 1, axis=0)
            data.append(img); labels.append(cls)
    return data, labels


def coarse(img, G=4):
    ch = np.asarray(change_field(img, 3), dtype=float)
    g = np.zeros((G, G))
    H, W = ch.shape
    for a in range(G):
        for b in range(G):
            i0, i1 = a * H // G, (a + 1) * H // G
            j0, j1 = b * W // G, (b + 1) * W // G
            g[a, b] = ch[i0:i1, j0:j1].mean()
    return g


def descriptor(g, y, x, R=1):
    out = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            yy, xx = y + a, x + b
            out.append(g[yy, xx] if 0 <= yy < G and 0 <= xx < G else 0.0)
    return out


G = 4


def main():
    data, labels = make_shapes()
    fields = [coarse(img) for img in data]

    sample = [descriptor(fields[k], a, b) for k in range(6)
              for a in range(1, G - 1) for b in range(1, G - 1)]
    dists = [np.mean(np.abs(np.array(sample[i]) - np.array(sample[j])))
             for i in range(0, len(sample), 3)
             for j in range(i + 1, min(i + 3, len(sample)))]
    eps = float(np.percentile(dists, 10))

    vocab, assigns = [], []
    for f in fields:
        amap = np.zeros((G, G), dtype=int)
        for a in range(G):
            for b in range(G):
                c = descriptor(f, a, b); best, bd = -1, 1e9
                for t, v in enumerate(vocab):
                    dd = np.mean(np.abs(np.array(c) - np.array(v)))
                    if dd < bd:
                        bd, best = dd, t
                if best >= 0 and bd <= eps:
                    amap[a, b] = best
                else:
                    vocab.append(c); amap[a, b] = len(vocab) - 1
        assigns.append(amap)
    print("self-emergent context states:", len(vocab))

    transitions = set()
    for amap in assigns:
        for a in range(G):
            for b in range(G):
                if a + 1 < G:
                    transitions.add((amap[a, b], amap[a + 1, b]))
                if b + 1 < G:
                    transitions.add((amap[a, b], amap[a, b + 1]))

    groups = {}
    for amap, lab in zip(assigns, labels):
        groups.setdefault(tuple(amap.ravel()), []).append(lab)
    pure = sum(max([ls.count(c) for c in set(ls)]) for ls in groups.values())
    print("unsupervised purity: %.1f%% over %d groups"
          % (100 * pure / len(labels), len(groups)))

    seed = assigns[0].copy()
    canvas = -np.ones((G, G), dtype=int)
    canvas[G // 2, :] = seed[G // 2, :]
    placed = int((canvas != -1).sum()); abstained = set()
    for _ in range(G * G):
        progress = False
        for a in range(G):
            for b in range(G):
                if canvas[a, b] != -1:
                    continue
                req = []
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    y, x = a + da, b + db
                    if 0 <= y < G and 0 <= x < G and canvas[y, x] != -1:
                        req.append((canvas[y, x], da, db))
                if not req:
                    continue
                cand = list(range(len(vocab)))
                for nb, da, db in req:
                    if da == 1:
                        cand = [s for s in cand if (s, nb) in transitions]
                    elif da == -1:
                        cand = [s for s in cand if (nb, s) in transitions]
                    elif db == 1:
                        cand = [s for s in cand if (s, nb) in transitions]
                    else:
                        cand = [s for s in cand if (nb, s) in transitions]
                if len(cand) == 1:
                    canvas[a, b] = cand[0]; placed += 1
                else:
                    abstained.add((a, b))
                progress = True
        if not progress:
            break
    edges = [(canvas[a, b], canvas[a + 1, b]) for a in range(G - 1)
             for b in range(G) if -1 not in (canvas[a, b], canvas[a + 1, b])]
    edges += [(canvas[a, b], canvas[a, b + 1]) for a in range(G)
              for b in range(G - 1) if -1 not in (canvas[a, b], canvas[a, b + 1])]
    closure = (float(np.mean([1.0 if e in transitions else 0.0 for e in edges]))
               if edges else float("nan"))
    print("generation: placed=%d/%d abstained=%d closure=%s"
          % (placed, G * G, len(abstained),
             ("%.1f%%" % (100 * closure)) if edges else "N/A (no edges)"))


if __name__ == "__main__":
    main()
