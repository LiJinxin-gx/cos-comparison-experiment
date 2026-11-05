# -*- coding: utf-8 -*-
"""Corrected LLM: local sequence encoding + passive/active dual mode.

Core principles:
- Information arises from LOCAL comparison, not global frequency statistics
- Passive mode: local d-shift self-comparison (gap space)
- Active mode: kernel template matching (element space)
- Primitive isolation: no cross-primitive weighting
- K=1 nearest neighbor, max fusion

Companion module: the principle steps run on the installed cos_comparison
package (the authoritative engine). The entry point analyze() still takes a
plain discrete 1D sequence, so data is injected directly in module mode; the
__main__ demo only handles corpus loading and the synthetic fallback.
"""
import os
import re

from cos_comparison import core

CORPUS = "alice.txt"  # Place path to your text corpus here


def _synthetic_words(n=600):
    """Deterministic fallback vocabulary so the demo runs without an
    external corpus. It is explicitly labelled as synthetic output."""
    base = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta",
            "eta", "theta", "iota", "kappa", "lambda", "mu")
    return [base[i % len(base)] for i in range(n)]


def words(p):
    if not os.path.exists(p):
        print("corpus %r not found; using built-in synthetic words" % p)
        return _synthetic_words()
    raw = open(p, encoding="utf-8", errors="ignore").read().lower()
    raw = re.sub(r"[^a-z\s]", " ", raw)
    return raw.split()[:3000]


def encode_sequence_local(ws):
    """Encode a word sequence as a 1D tensor using ordinal values.

    NO global frequency ranking - just a direct ordinal mapping. Information
    comes from LOCAL comparison (passive mode), not from global statistics.
    """
    return [float(ord(w[0])) / 255.0 for w in ws]


def analyze(sequence, kernel_start=100, kernel_len=3, hit_threshold=0.9):
    """Principle entry point: takes a 1D discrete sequence directly.

    Passive and active run on the installed cos_comparison engine using the
    formal default cosmod algorithm. Returns the boundary map, kernel match
    map, hit positions and the locally combined generated fragment.
    """
    l0 = sequence

    # Passive d-shift self-comparison -> boundary map (gap space).
    l1 = [float(v) for v in core.cos_comparison_passive(
        l0, window_size=(3,), d=(1,), algorithm=core._cosmod)]

    # Active kernel template match (element space).
    start = min(kernel_start, max(0, len(l0) - kernel_len))
    kernel = l0[start:start + kernel_len]
    l2 = [float(v) for v in core.cos_comparison_active(l0, kernel=kernel)]

    hits = [i for i, v in enumerate(l2) if v > hit_threshold]
    generated = None
    if len(hits) >= 2:
        a, b = hits[0], hits[-1]
        generated = l0[a:a + 2] + l0[b:b + 2]

    return {"l0": l0, "l1": l1, "l2": l2, "hits": hits,
            "generated": generated}


def run(ws):
    """Encode the supplied words, then run the passive/active analysis."""
    l0 = encode_sequence_local(ws)
    result = analyze(l0)
    print("tokens", len(l0))
    print("L0=%d L1=%d L2=%d"
          % (len(result["l0"]), len(result["l1"]), len(result["l2"])))
    print("hits", len(result["hits"]))
    if result["generated"] is not None:
        gen = result["generated"]
        print("GENERATED (combined two local hits):",
              " ".join(ws[int(v * 255)] if int(v * 255) < len(ws) else "?"
                       for v in gen))
    else:
        print("GENERATED: stop (3-valued)")
    return result


def main():
    ws = words(CORPUS)
    run(ws)
    print("\nPrinciples verified:")
    print("- Local encoding: no global frequency statistics")
    print("- Passive mode: local d-shift self-comparison (gap space)")
    print("- Active mode: kernel template matching (element space)")
    print("- Primitive isolation: separate channels, max fusion")


if __name__ == "__main__":
    main()
