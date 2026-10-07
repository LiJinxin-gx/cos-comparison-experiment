"""
candidates.cand_feedback -- ambiguity-driven granularity with DB separation.

Promising, immature (status: candidate).

  main DB  = accumulated from training; read-only during application.
  inner DB = application-time learning, NOT committed to main until a merge.
Ambiguity = gap between best and second-best match. A confident gap yields an
answer; a small gap triggers finer-grained extraction on the disputed item and
pushes it to the inner DB (the answer is abstained). A controlled merge then
promotes inner entries into main for continual, reviewable learning.

Bottleneck targeted: in-application adaptation without polluting the main DB.
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
    "targets": "controlled continual learning via main/inner DB separation",
}


def make_data(n=16):
    yy, xx = np.indices((n, n)); c = (n - 1) / 2.0
    d = np.hypot(xx - c, yy - c)
    ring = ((d >= 3) & (d <= 5)).astype(float)
    cross = ((np.abs(xx - c) <= 1) | (np.abs(yy - c) <= 1)).astype(float)
    block = ((np.abs(xx - c) <= 3) & (np.abs(yy - c) <= 3)).astype(float)
    classes = (ring, cross, block)
    train, test = [], []
    for cls, base in enumerate(classes):
        train += [(np.roll(base, s, axis=(s % 2)), cls) for s in range(3)]
        test += [(np.roll(base, 1, axis=0), cls),
                 (0.5 * np.roll(base, 1, axis=0) + 0.5 * np.roll(base, 1, axis=1), cls)]
    return train, test


def profile(img, fine=False):
    ch = np.asarray(change_field(img, 3), dtype=float)
    G = 4 if not fine else 8
    g = np.zeros((G, G)); H, W = ch.shape
    for a in range(G):
        for b in range(G):
            g[a, b] = ch[a * H // G:(a + 1) * H // G,
                         b * W // G:(b + 1) * W // G].mean()
    return g.ravel()


def similarity(p, q):
    p, q = p - p.mean(), q - q.mean()
    den = np.sqrt((p * p).sum() * (q * q).sum())
    return float((p * q).sum() / den) if den > 1e-12 else 0.0


def main():
    train, test = make_data()
    main_coarse = [[profile(img) for img, cls in train if cls == k] for k in range(3)]
    inner_fine = [[] for _ in range(3)]
    main_fine = [[] for _ in range(3)]

    correct, answered, abstained, finer = 0, 0, 0, 0
    for img, true in test:
        p = profile(img)
        peaks = [max(similarity(p, t) for t in main_coarse[k]) for k in range(3)]
        order = sorted(range(3), key=lambda k: -peaks[k])
        gap = peaks[order[0]] - peaks[order[1]]
        if gap >= 0.05:
            answered += 1; correct += int(order[0] == true); continue
        pf = profile(img, fine=True)
        if inner_fine and any(inner_fine):
            fpeaks = [max((similarity(pf, t) for t in inner_fine[k]),
                          default=-1.0) for k in range(3)]
            fo = sorted(range(3), key=lambda k: -fpeaks[k])
            fgap = fpeaks[fo[0]] - fpeaks[fo[1]]
            if fgap >= 0.05:
                answered += 1; correct += int(fo[0] == true)
                inner_fine[true].append(pf); finer += 1; continue
        abstained += 1
        inner_fine[true].append(pf); finer += 1

    print("confident accuracy: %.1f%% (%d answered)"
          % (100 * correct / max(1, answered), answered))
    print("abstained (ambiguous): %d, fine profiles sent to inner DB: %d"
          % (abstained, finer))
    for k in range(3):
        main_fine[k] += inner_fine[k]
    print("after merge, main fine section sizes:", [len(x) for x in main_fine])


if __name__ == "__main__":
    main()
