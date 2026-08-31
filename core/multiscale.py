"""
Multi-scale cosine matching.

Core idea: represent data at multiple scales (original + downsampled),
match independently at each scale, then ensemble with finer scales weighted
higher. This is the most broadly applicable algorithm in the project —
verified across face, captcha, text, video, and audio domains.

Zero-dependency. Downsampling uses simple averaging (no numpy).
"""
from .cosine import cosine, flatten, as_tensor


def downsample(tensor, factor=2):
    """Average-pool a 1D or 2D tensor by integer factor.

    1D: groups of `factor` elements averaged.
    2D: non-overlapping factor x factor blocks averaged.
    """
    tensor = as_tensor(tensor)
    if len(tensor) == 0:
        return []
    # 1D
    if not isinstance(tensor[0], (list, tuple)):
        n = len(tensor) // factor
        result = []
        for i in range(n):
            block = tensor[i * factor:(i + 1) * factor]
            result.append(sum(block) / len(block))
        return result
    # 2D
    rows = len(tensor) // factor
    cols = len(tensor[0]) // factor
    result = []
    for r in range(rows):
        row = []
        for c in range(cols):
            total = 0.0
            count = 0
            for dr in range(factor):
                for dc in range(factor):
                    total += float(tensor[r * factor + dr][c * factor + dc])
                    count += 1
            row.append(total / count)
        result.append(row)
    return result


def multiscale_represent(tensor, scales=(1, 2, 4)):
    """Generate multi-scale representations of a tensor.

    Returns list of (scale, flattened_representation) tuples.
    scale=1 is original, scale=2 is 2x downsampled, etc.
    """
    tensor = as_tensor(tensor)
    reps = []
    for s in scales:
        if s == 1:
            reps.append((1, flatten(tensor)))
        else:
            t = tensor
            for _ in range(s.bit_length() - 1):
                t = downsample(t, 2)
            # handle non-power-of-two scales
            if s not in (1, 2, 4, 8, 16):
                t = downsample(tensor, s)
            reps.append((s, flatten(t)))
    return reps


def multiscale_cosine(a, b, scales=(1, 2, 4), weights=None):
    """Multi-scale cosine similarity with weighted ensemble.

    Weights default to 1/sqrt(scale) — finer scales contribute more.
    Returns weighted average of per-scale cosine similarities.
    """
    if weights is None:
        from math import sqrt
        weights = [1.0 / sqrt(s) for s in scales]
    reps_a = multiscale_represent(a, scales)
    reps_b = multiscale_represent(b, scales)
    total_w = 0.0
    total_sim = 0.0
    for (s, ra), (_, rb), w in zip(reps_a, reps_b, weights):
        sim = cosine(ra, rb)
        total_sim += w * sim
        total_w += w
    if total_w == 0.0:
        return 0.0
    return total_sim / total_w


def multiscale_match(query, database, scales=(1, 2, 4), top_k=5):
    """Match query against a database of tensors using multi-scale cosine.

    database: list of (label, tensor) tuples.
    Returns list of (label, similarity) sorted descending, top_k entries.
    """
    results = []
    for label, item in database:
        sim = multiscale_cosine(query, item, scales)
        results.append((label, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
