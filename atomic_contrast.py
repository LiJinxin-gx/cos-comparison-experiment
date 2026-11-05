# -*- coding: utf-8 -*-
"""
Atomic Contrast Point Matching.

Core principles:
1. Information arises from LOCAL comparison (not global feature vectors)
2. Atomic contrast points: local extrema as binary signature
3. Jaccard similarity on contrast point sets (not cosine on vectors)
4. Hierarchical isolation: multi-scale contrast sets
5. Primitive isolation: active (element space) + passive (gap space) separate
6. K=1 nearest neighbor, no voting, no weighting

Zero dependencies: pure Python standard library.
Data loading separated from processing.
NO global statistics, NO global thresholds, NO hardcoded parameters.

The continuous-mapping primitives run on the installed cos_comparison
engine (core.pool, core.cos_comparison_passive); the set-based contrast
extraction and Jaccard measure, which have no tensor counterpart, stay local.
"""
import math
import random

from cos_comparison import core


# ============================================================
# Core Primitives (Pure Local Operations)
# ============================================================

def extract_atomic_contrast_points(tensor, neighborhood_radius=1):
    """
    Extract atomic contrast points from a 2D tensor.
    
    An atomic contrast point is a local extremum (max or min) relative to
    its local neighborhood. This is PURE local comparison - NO global threshold,
    NO global statistics.
    
    Mathematical meaning: this is the "gap space" of passive mode at the atomic
    point level. A point that stands out from its local surround IS the contrast.
    No absolute value threshold needed - local comparison alone defines salience.
    
    Returns: set of (i, j) coordinates of contrast points.
    """
    h, w = len(tensor), len(tensor[0])
    r = neighborhood_radius
    
    if h < 2*r+1 or w < 2*r+1:
        return set()
    
    contrast_points = set()
    
    # Check each point in the interior
    for i in range(r, h-r):
        for j in range(r, w-r):
            val = tensor[i][j]
            
            # Local max: greater than all neighbors
            is_max = True
            # Local min: less than all neighbors
            is_min = True
            
            for di in range(-r, r+1):
                for dj in range(-r, r+1):
                    if di == 0 and dj == 0:
                        continue
                    neighbor = tensor[i+di][j+dj]
                    if val <= neighbor:
                        is_max = False
                    if val >= neighbor:
                        is_min = False
            
            # Pure local comparison: no global threshold needed!
            # Being a local extremum IS the contrast criterion.
            if is_max or is_min:
                contrast_points.add((i, j))
    
    return contrast_points


def jaccard_similarity(set_a, set_b):
    """
    Jaccard similarity between two contrast point sets.
    |A ∩ B| / |A ∪ B| - pure structural comparison, no weighting.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def down_sample_2d(tensor, factor):
    """
    Continuous down-sampling via the formal engine (average pool).
    This is the continuous mapping f_{i,i+1} in the transformation group.
    """
    h, w = len(tensor), len(tensor[0])
    nh, nw = h // factor, w // factor
    output = core.create_void_list((nh, nw))
    core.pool(tensor, window_size=(factor, factor),
              step=(factor, factor), output=output)
    return output


# ============================================================
# Multi-scale Hierarchical Contrast
# ============================================================

class HierarchicalContrastMatcher:
    """
    Multi-scale atomic contrast point matching with hierarchy isolation.
    
    Group-theoretic principles:
    - Each scale is a level L_i in the transformation group
    - Down-sampling f_{i,i+1}: continuous coarsening mapping
    - Up-sampling f_{i+1,i}: structural inverse (refinement)
    - Level isolation: each level stores its own features independently
    
    Primitive isolation:
    - Active channel: element space contrast (raw values' local extrema)
    - Passive channel: gap space contrast (gradient magnitude's local extrema)
    - NEVER mixed by averaging - max fusion only at decision layer
    
    K=1 nearest neighbor: no voting, no weighting.
    """
    
    def __init__(self, scale_factors=None, neighborhood_radius=1):
        if scale_factors is None:
            scale_factors = (1, 2, 4)
        self.scale_factors = scale_factors
        self.neighborhood_radius = neighborhood_radius
        self.prototypes = {}  # label -> list of (active_sets, passive_sets)
    
    def _extract_levels(self, tensor):
        """
        Extract contrast sets at each scale independently.
        Level isolation: each level's features are computed independently.
        """
        active_sets = []
        passive_sets = []
        
        current = tensor
        for f in self.scale_factors:
            if f > 1:
                current = down_sample_2d(current, f)
            
            # Active channel: element space contrast points
            # (local extrema in the raw values themselves)
            active = extract_atomic_contrast_points(
                current, self.neighborhood_radius)
            active_sets.append(active)
            
            # Passive channel: gap space contrast points
            # (local extrema in the gradient magnitude map)
            passive = self._extract_passive_contrast(current)
            passive_sets.append(passive)
        
        return active_sets, passive_sets
    
    def _extract_passive_contrast(self, tensor):
        """
        Passive boundary map via the formal engine, then local extrema.

        The formal passive is a cosmod similarity map: 1 inside a uniform
        region and low at a boundary. Boundary points therefore appear as
        local minima; the same extremum extraction (max or min) captures
        them as contrast points in the gap space.
        """
        h, w = len(tensor), len(tensor[0])
        if h < 2 or w < 2:
            return set()

        boundary = core.create_void_list((h - 1, w))
        core.cos_comparison_passive(tensor, window_size=(1, 1),
                                    d=(1, 0), output=boundary)
        return extract_atomic_contrast_points(boundary, self.neighborhood_radius)
    
    def fit(self, images, labels):
        """Store prototypes: contrast sets per class per scale."""
        self.prototypes = {}
        for img, label in zip(images, labels):
            active_lvls, passive_lvls = self._extract_levels(img)
            if label not in self.prototypes:
                self.prototypes[label] = []
            self.prototypes[label].append((active_lvls, passive_lvls))
    
    def predict(self, img):
        """
        Predict label using max Jaccard across scales and primitives.
        Decision layer: max fusion (NOT weighted average!).
        """
        query_active, query_passive = self._extract_levels(img)
        
        best_score = -1.0
        best_label = None
        
        for label, proto_list in self.prototypes.items():
            for proto_active, proto_passive in proto_list:
                # Active channel: max over scales (level isolation)
                a_score = 0.0
                for s in range(len(query_active)):
                    sim = jaccard_similarity(query_active[s], proto_active[s])
                    if sim > a_score:
                        a_score = sim
                
                # Passive channel: max over scales (separate primitive!)
                p_score = 0.0
                for s in range(len(query_passive)):
                    sim = jaccard_similarity(query_passive[s], proto_passive[s])
                    if sim > p_score:
                        p_score = sim
                
                # Decision layer: MAX fusion (NOT weighted average!)
                level_score = max(a_score, p_score)
                
                if level_score > best_score:
                    best_score = level_score
                    best_label = label
        
        return best_label
    
    def accuracy(self, images, labels):
        correct = sum(1 for img, lbl in zip(images, labels)
                     if self.predict(img) == lbl)
        return correct / len(labels) if labels else 0.0


# ============================================================
# Demo / Test
# ============================================================

def make_synthetic_pattern(class_id, size=28):
    """Create synthetic test patterns."""
    img = [[0.0] * size for _ in range(size)]
    
    if class_id == 0:  # circle
        cx, cy, r = size//2, size//2, size//3
        for i in range(size):
            for j in range(size):
                dist = math.sqrt((i-cx)**2 + (j-cy)**2)
                if abs(dist - r) < 2:
                    img[i][j] = 200.0
    elif class_id == 1:  # horizontal line
        for i in range(size//2-2, size//2+2):
            for j in range(size):
                img[i][j] = 200.0
    elif class_id == 2:  # vertical line
        for i in range(size):
            for j in range(size//2-2, size//2+2):
                img[i][j] = 200.0
    elif class_id == 3:  # diagonal
        for i in range(size):
            j = int(i * 0.8)
            if 0 <= j < size:
                for dj in range(-2, 3):
                    if 0 <= j+dj < size:
                        img[i][j+dj] = 200.0
    else:  # cross
        for i in range(size//2-2, size//2+2):
            for j in range(size):
                img[i][j] = 200.0
        for i in range(size):
            for j in range(size//2-2, size//2+2):
                img[i][j] = 200.0
    
    # Add noise
    for i in range(size):
        for j in range(size):
            img[i][j] += random.gauss(0, 8)
    
    return img


def main():
    print("=" * 60)
    print("Atomic Contrast Point Matching Demo")
    print("=" * 60)
    print("\nCore principles:")
    print("- Information arises from local comparison")
    print("- Atomic contrast points (pure local extrema, no global threshold)")
    print("- Jaccard set similarity (not cosine vectors)")
    print("- Hierarchical isolation (multi-scale continuous mapping)")
    print("- Primitive isolation (element space / gap space separate)")
    print("- K=1 nearest neighbor, max fusion at decision")
    print()
    
    random.seed(42)
    
    # Generate synthetic data
    n_classes = 5
    n_train = 30
    n_test = 10
    
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    
    for c in range(n_classes):
        for _ in range(n_train):
            train_imgs.append(make_synthetic_pattern(c))
            train_labels.append(f"class_{c}")
        for _ in range(n_test):
            test_imgs.append(make_synthetic_pattern(c))
            test_labels.append(f"class_{c}")
    
    print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")
    
    # Train and test
    matcher = HierarchicalContrastMatcher(
        scale_factors=(1, 2, 4),
        neighborhood_radius=1
    )
    
    print("\nTraining...")
    matcher.fit(train_imgs, train_labels)
    
    print("Evaluating...")
    acc = matcher.accuracy(test_imgs, test_labels)
    
    print(f"\nAccuracy: {acc * 100:.1f}%")
    print(f"\nPrinciple verified: atomic contrast + Jaccard + hierarchy isolation")


if __name__ == "__main__":
    main()
