"""
BHSM — Bidirectional Hierarchical Similarity Matching.

Algorithm evolution: v6 (same-points only, 9.2%) → v12 (current best, 73.3%
on standard face benchmark). v12 is a local optimum — any incremental change
to the coarse-ranking stage degrades performance.

v12 pipeline (11 atomic stages):
  1. Boundary extraction (passive mode)
  2. Convex-hull masking (environment suppression)
  3. Multi-threshold response (7 thresholds)
  4. Local extrema selection
  5. Boundary-weighted point scoring
  6. Top-K point selection
  7. Geometric signature (distance-angle histogram)
  8. Template construction per class
  9. Soft pose estimation
  10. Coarse ranking (multi-level weighted)
  11. Fine re-ranking (descriptor-filtered matching)

This zero-dependency implementation provides the core matching logic with
tensor inputs. Preprocessing (image loading, etc.) is intentionally excluded.
"""
from core.cosine import cosine, as_tensor, flatten
from core.passive_active import passive_extract, local_variance
from core.multiscale import multiscale_cosine


def extract_keypoints(tensor, n_thresholds=7, top_k=67):
    """Extract keypoints using multi-threshold passive response.

    Simplified v12 keypoint extraction:
      - Compute local variance at multiple thresholds
      - Select top-K highest-variance points
      - Weight by boundary strength

    Returns list of (index, response, boundary_weight).
    """
    tensor = as_tensor(tensor)
    variances = local_variance(tensor, window=3)
    if not variances:
        return []
    max_var = max(variances)
    if max_var == 0.0:
        return []

    # Multi-threshold: count how many thresholds each point passes
    threshold_responses = []
    for i, v in enumerate(variances):
        passes = 0
        for t_idx in range(n_thresholds):
            threshold = (t_idx + 1) / (n_thresholds + 1) * max_var
            if v >= threshold:
                passes += 1
        # boundary weight = normalized variance * threshold passes
        boundary_weight = (v / max_var) * (passes / n_thresholds)
        threshold_responses.append((i, v, boundary_weight))

    # Select top-K by boundary weight
    threshold_responses.sort(key=lambda x: x[2], reverse=True)
    return threshold_responses[:top_k]


def geometric_signature(keypoints, n_bins=128):
    """Build geometric signature from keypoints (distance-angle histogram).

    For each pair of keypoints, compute distance and angle, bin into 2D
    histogram. This is the core discriminative feature in v12.
    """
    if len(keypoints) < 2:
        return [0.0] * n_bins
    from math import sqrt, atan2, pi
    distances = []
    angles = []
    for i in range(len(keypoints)):
        for j in range(i + 1, len(keypoints)):
            dx = keypoints[j][0] - keypoints[i][0]
            dy = keypoints[j][1] - keypoints[j][1] if len(keypoints[i]) > 1 else 0
            dist = sqrt(dx * dx + dy * dy)
            angle = atan2(dy, dx)
            distances.append(dist)
            angles.append(angle)

    if not distances:
        return [0.0] * n_bins

    max_dist = max(distances) if distances else 1.0
    # 1D signature: distance histogram (simplified from 2D)
    hist = [0.0] * n_bins
    for d in distances:
        bin_idx = min(int(d / max_dist * n_bins), n_bins - 1)
        hist[bin_idx] += 1.0
    # normalize
    total = sum(hist)
    if total > 0:
        hist = [h / total for h in hist]
    return hist


def build_template(keypoints_list, n_bins=128):
    """Build a class template from multiple samples' keypoints.

    Template = average geometric signature across samples.
    """
    signatures = []
    for kps in keypoints_list:
        sig = geometric_signature(kps, n_bins)
        signatures.append(sig)
    if not signatures:
        return [0.0] * n_bins
    # average
    template = [0.0] * n_bins
    for sig in signatures:
        for i in range(n_bins):
            template[i] += sig[i]
    return [t / len(signatures) for t in template]


def coarse_rank(query_keypoints, templates, n_bins=128):
    """Coarse ranking: match query signature against class templates.

    Returns list of (class_label, similarity) sorted descending.
    """
    query_sig = geometric_signature(query_keypoints, n_bins)
    results = []
    for label, template in templates:
        sim = cosine(query_sig, template)
        results.append((label, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def fine_rerank(query_tensor, candidates, database, scales=(1, 2, 4)):
    """Fine re-ranking: multi-scale cosine match against top candidates.

    candidates: list of (label, coarse_sim) from coarse_rank.
    database: dict label -> list of sample tensors.
    Returns re-ranked list with combined score.
    """
    results = []
    for label, coarse_sim in candidates:
        if label not in database:
            results.append((label, coarse_sim))
            continue
        best_fine = 0.0
        for sample in database[label]:
            fine_sim = multiscale_cosine(query_tensor, sample, scales)
            if fine_sim > best_fine:
                best_fine = fine_sim
        # combined: 50% coarse + 50% fine
        combined = 0.5 * coarse_sim + 0.5 * best_fine
        results.append((label, combined))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def bhsm_match(query_tensor, templates, database, top_k_coarse=10,
               n_bins=128, scales=(1, 2, 4)):
    """Full BHSM pipeline: extract → coarse rank → fine re-rank.

    Args:
        query_tensor: input tensor (1D or 2D)
        templates: list of (label, template_signature)
        database: dict label -> list of sample tensors (for fine ranking)
        top_k_coarse: number of candidates to keep after coarse ranking

    Returns:
        list of (label, final_score) sorted descending
    """
    keypoints = extract_keypoints(flatten(as_tensor(query_tensor)))
    # For 2D, keypoints need (row, col) — simplified here
    coarse = coarse_rank(keypoints, templates, n_bins)
    top_candidates = coarse[:top_k_coarse]
    final = fine_rerank(query_tensor, top_candidates, database, scales)
    return final
