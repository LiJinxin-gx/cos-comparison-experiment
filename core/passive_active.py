"""
Passive-active joint extraction.

Core principle: information arises from difference.
  - Passive mode: extract boundaries / high-variance regions (where data
    changes) — these are structural anchors.
  - Active mode: template matching against stored patterns — these are
    semantic matches.
  - Joint: passive points define WHERE to look, active templates define
    WHAT to match. Complementary, not redundant.

Zero-dependency. Works on 1D and 2D tensors.
"""
from .cosine import cosine, as_tensor, flatten
from .multiscale import downsample


def local_variance(tensor, window=3):
    """Compute local variance of a 1D tensor with given window size.

    Returns list of variance values (same length, edge-padded).
    High variance = boundary / transition point (passive detection).
    """
    tensor = as_tensor(tensor)
    n = len(tensor)
    half = window // 2
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        block = [float(tensor[j]) for j in range(lo, hi)]
        if len(block) < 2:
            result.append(0.0)
            continue
        mean = sum(block) / len(block)
        var = sum((x - mean) ** 2 for x in block) / len(block)
        result.append(var)
    return result


def local_mean(tensor, window=3):
    """Compute local mean of a 1D tensor (smoothing)."""
    tensor = as_tensor(tensor)
    n = len(tensor)
    half = window // 2
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        block = [float(tensor[j]) for j in range(lo, hi)]
        result.append(sum(block) / len(block))
    return result


def passive_extract(tensor, window=3, threshold=0.5):
    """Extract high-variance (boundary) points from a 1D tensor.

    Returns list of (index, variance) for points above threshold * max_variance.
    These are passive-mode structural anchors.
    """
    variances = local_variance(tensor, window)
    if not variances:
        return []
    max_var = max(variances)
    if max_var == 0.0:
        return []
    threshold_abs = threshold * max_var
    points = []
    for i, v in enumerate(variances):
        if v >= threshold_abs:
            points.append((i, v))
    return points


def passive_extract_2d(matrix, window=3, threshold=0.5):
    """Extract high-variance points from a 2D tensor.

    Returns list of (row, col, variance) for boundary points.
    """
    matrix = as_tensor(matrix)
    rows = len(matrix)
    if rows == 0:
        return []
    cols = len(matrix[0])
    flat = flatten(matrix)
    # reshape for 1D variance per row
    all_points = []
    for r in range(rows):
        row_points = passive_extract(matrix[r], window, threshold)
        for c, v in row_points:
            all_points.append((r, c, v))
    return all_points


def active_match(tensor, templates, window=3):
    """Active mode: slide templates over tensor, return best match per template.

    templates: list of (name, template_vector) tuples.
    Returns list of (name, best_position, best_similarity).
    """
    tensor = as_tensor(tensor)
    results = []
    for name, tmpl in templates:
        tmpl = as_tensor(tmpl)
        tlen = len(tmpl)
        if tlen > len(tensor):
            results.append((name, -1, 0.0))
            continue
        best_sim = -1.0
        best_pos = 0
        for i in range(len(tensor) - tlen + 1):
            segment = tensor[i:i + tlen]
            sim = cosine(segment, tmpl)
            if sim > best_sim:
                best_sim = sim
                best_pos = i
        results.append((name, best_pos, best_sim))
    return results


def joint_extract(tensor, templates, passive_window=3, passive_threshold=0.5):
    """Joint passive-active extraction.

    1. Passive: find boundary points (structural anchors).
    2. Active: match templates only near boundary points (where information
       density is highest).
    Returns dict with passive_points and active_matches (filtered to regions
    near passive points).
    """
    passive_points = passive_extract(tensor, passive_window, passive_threshold)
    active_matches = active_match(tensor, templates)
    # Filter active matches to positions near passive points
    passive_indices = set(p[0] for p in passive_points)
    filtered_active = []
    for name, pos, sim in active_matches:
        near = any(abs(pos - pi) <= passive_window for pi in passive_indices)
        if near or sim > 0.8:
            filtered_active.append((name, pos, sim))
    return {
        "passive_points": passive_points,
        "active_matches": filtered_active,
        "n_passive": len(passive_points),
        "n_active": len(filtered_active),
    }
