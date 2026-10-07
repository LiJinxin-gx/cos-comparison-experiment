"""
candidates.cand_template_search -- few-template heuristic routing.

Promising, immature (status: candidate).

Naive matching compares a query fully against every library item: O(N * L).
Instead compute a cheap local coarse signature once (O(L)), compare it against
stored coarse signatures (N cheap comparisons), and exact-compare only a shortlist.
Per query cost becomes linear in library size with a small constant.

Bottleneck targeted: scaling local comparison to large libraries.
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
    "targets": "linear-time library search via coarse template routing",
}


def library(n=16, per_class=12):
    yy, xx = np.indices((n, n)); c = (n - 1) / 2.0
    d = np.hypot(xx - c, yy - c)
    bases = (((d >= 3) & (d <= 5)),
             ((np.abs(xx - c) <= 1) | (np.abs(yy - c) <= 1)),
             ((np.abs(xx - c) <= 3) & (np.abs(yy - c) <= 3)))
    items, labels = [], []
    rng = np.arange(per_class)
    for cls, base in enumerate(bases):
        for v in rng:
            img = np.roll(base, int(v % 5), axis=int(v % 2)).astype(float)
            items.append(img); labels.append(cls)
    return items, labels


def full_signature(img):
    return np.asarray(change_field(img, 3), dtype=float).ravel()


def coarse_signature(img, G=4):
    ch = np.asarray(change_field(img, 3), dtype=float)
    g = np.zeros((G, G)); H, W = ch.shape
    for a in range(G):
        for b in range(G):
            g[a, b] = ch[a * H // G:(a + 1) * H // G,
                         b * W // G:(b + 1) * W // G].mean()
    return g.ravel()


def L1(a, b):
    return float(np.abs(a - b).sum())


def main():
    items, labels = library()
    N = len(items)
    full_lib = [full_signature(x) for x in items]
    coarse_lib = [coarse_signature(x) for x in items]
    L = items[0].size
    K = 3

    brute_ops, routed_ops, hits, correct = 0, 0, 0, 0
    for qi in range(0, N, 3):
        qf, qc = full_lib[qi], coarse_lib[qi]
        brute = [L1(qf, f) for f in full_lib]
        brute_answer = int(np.argmin(brute))
        brute_ops += N * L
        cd = [L1(qc, c) for c in coarse_lib]
        shortlist = np.argsort(cd)[:K]
        routed_ops += L + N * len(qc) + K * L
        answer = int(shortlist[np.argmin([brute[j] for j in shortlist])])
        hits += int(brute_answer in shortlist)
        correct += int(answer == brute_answer)

    trials = len(range(0, N, 3))
    print("library N=%d, signal L=%d, shortlist K=%d" % (N, L, K))
    print("shortlist recall: %.1f%%, routed==brute: %.1f%%"
          % (100 * hits / trials, 100 * correct / trials))
    print("elementwise ops -- brute: %d, routed: %d (%.1fx fewer)"
          % (brute_ops, routed_ops, brute_ops / max(1, routed_ops)))


if __name__ == "__main__":
    main()
