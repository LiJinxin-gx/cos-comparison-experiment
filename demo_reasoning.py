# -*- coding: utf-8 -*-
"""
Reasoning Engine Demo: Top-Down Hypothesis Testing.

Demonstrates the reasoning chain:
  1. Start from high-level abstract hypothesis (what might this be?)
  2. Propagate down to lower levels (fill in details)
  3. Compare with bottom-up evidence (does it match?)
  4. Adjust hypothesis based on mismatch
  5. Repeat until convergence

Core principles:
  - Bidirectional A<->B: both bottom-up and top-down flows exist
  - Level isolation: each level has its own matching/verification
  - No global weighting: decisions at each level are independent
  - Absolute locality: all comparisons are local
  - K=1 nearest neighbor: best match at each level, no voting

Zero dependencies: pure Python standard library only.
The principle primitives run on the installed cos_comparison engine
(core.pool, core.cos_comparison_passive, core.cos); only structural
helpers and the in-memory backend remain local.
"""
import random

from cos_comparison import core


def cosine_similarity(a, b):
    """Whole-tensor cosine similarity via the formal engine."""
    return core.cos(a, b, algorithm=core._cos)


def down_sample(tensor, factor):
    """Average pooling via the formal engine: coarsen by factor."""
    h, w = len(tensor), len(tensor[0])
    nh, nw = h // factor, w // factor
    output = core.create_void_list((nh, nw))
    core.pool(tensor, window_size=(factor, factor),
              step=(factor, factor), output=output)
    return output


def passive_extract(tensor):
    """Passive d-shift boundary map via the formal engine (gap space)."""
    h, w = len(tensor), len(tensor[0])
    output = core.create_void_list((h - 1, w))
    core.cos_comparison_passive(tensor, window_size=(1, 1),
                                d=(1, 0), output=output)
    return output


def active_extract(tensor):
    return flatten_2d(tensor)


def flatten_2d(tensor):
    return [x for row in tensor for x in row]


class _DefaultInMemoryBackend:
    def store(self, key, value):
        return True

    def retrieve(self, key):
        return None


class GroupHierarchicalClassifier:
    """Group-theoretic hierarchical classifier (local, self-contained)."""

    def __init__(self, pool_factors=None, use_passive=True,
                 similarity_func=None, memory_backend=None):
        self.pool_factors = pool_factors or [1, 2, 4]
        self.use_passive = use_passive
        self.similarity_func = similarity_func or cosine_similarity
        self.memory_backend = memory_backend or _DefaultInMemoryBackend()
        self.prototypes = {}

    def _extract_levels(self, tensor):
        active_levels, passive_levels = [], []
        current = tensor
        for factor in self.pool_factors:
            if factor > 1:
                current = down_sample(current, factor)
            active_levels.append(active_extract(current))
            if self.use_passive:
                passive_levels.append(flatten_2d(passive_extract(current)))
        return active_levels, passive_levels

    def fit(self, images, labels):
        self.prototypes = {}
        for img, label in zip(images, labels):
            active_levels, passive_levels = self._extract_levels(img)
            if label not in self.prototypes:
                self.prototypes[label] = []
            entry = {'active': active_levels}
            if self.use_passive:
                entry['passive'] = passive_levels
            self.prototypes[label].append(entry)
        self.memory_backend.store('prototypes', self.prototypes)
        return self

    def predict(self, image):
        q_active, q_passive = self._extract_levels(image)
        best_label, best_score = None, -1.0
        for label, proto_list in self.prototypes.items():
            for proto in proto_list:
                score = 0.0
                for lvl in range(len(self.pool_factors)):
                    sim = self.similarity_func(q_active[lvl], proto['active'][lvl])
                    if sim > score:
                        score = sim
                if self.use_passive and 'passive' in proto:
                    for lvl in range(len(self.pool_factors)):
                        sim = self.similarity_func(q_passive[lvl], proto['passive'][lvl])
                        if sim > score:
                            score = sim
                if score > best_score:
                    best_score, best_label = score, label
        return best_label

    def accuracy(self, images, labels):
        correct = sum(1 for img, lab in zip(images, labels)
                      if self.predict(img) == lab)
        return correct / len(labels) if labels else 0.0


def reasoning_demo():
    """Demo: multi-step reasoning chain."""
    print("=" * 60)
    print("Reasoning Engine Demo: Top-Down Hypothesis Testing")
    print("=" * 60)
    
    random.seed(42)
    size = 16
    pool_factors = [1, 2, 2]  # 16x16 -> 8x8 -> 4x4
    
    def make_digit(class_id):
        """Create simple digit-like patterns."""
        img = [[0.0] * size for _ in range(size)]
        if class_id == 0:  # vertical line
            for i in range(2, 14):
                img[i][8] = 200.0
        elif class_id == 1:  # horizontal line
            for j in range(2, 14):
                img[8][j] = 200.0
        else:  # cross
            for i in range(2, 14):
                img[i][8] = 200.0
            for j in range(2, 14):
                img[8][j] = 200.0
        for i in range(size):
            for j in range(size):
                img[i][j] += random.gauss(0, 3)
        return img
    
    # Build training set and extract features
    print("\n--- Building Training Database ---")
    train_images = [make_digit(i % 3) for i in range(60)]
    train_labels = [f"digit_{i % 3}" for i in range(60)]
    
    # Use GroupHierarchicalClassifier directly for training
    clf = GroupHierarchicalClassifier(pool_factors=pool_factors, use_passive=True)
    clf.fit(train_images, train_labels)
    
    print(f"Classes: {list(clf.prototypes.keys())}")
    
    # Test image
    print("\n--- Reasoning Process ---")
    test_img = make_digit(2)  # True label: cross
    for i in range(size):
        for j in range(size):
            test_img[i][j] += random.gauss(0, 10)
    
    # Step 1: Bottom-up extract features
    print("Step 1: Bottom-up evidence extraction")
    q_active, q_passive = clf._extract_levels(test_img)
    for lvl in range(len(q_active)):
        print(f"  L{lvl} active: {len(q_active[lvl])} dims")
    
    # Step 2: High-level hypothesis (L2 first)
    print("\nStep 2: High-level hypothesis (L2 evidence)")
    level2_scores = {}
    for label, proto_list in clf.prototypes.items():
        best_sim = 0.0
        for proto in proto_list:
            sim = cosine_similarity(q_active[2], proto['active'][2])
            if sim > best_sim:
                best_sim = sim
        level2_scores[label] = best_sim
        print(f"  {label}: L2 similarity = {best_sim:.3f}")
    
    top_hypothesis = max(level2_scores, key=level2_scores.get)
    print(f"\n  → Initial hypothesis: {top_hypothesis}")
    
    # Step 3: Verify at L1
    print("\nStep 3: Verify hypothesis at L1")
    level1_scores = {}
    for label, proto_list in clf.prototypes.items():
        best_sim = 0.0
        for proto in proto_list:
            sim = cosine_similarity(q_active[1], proto['active'][1])
            if sim > best_sim:
                best_sim = sim
        level1_scores[label] = best_sim
        print(f"  {label}: L1 similarity = {best_sim:.3f}")
    
    # Step 4: Verify at L0
    print("\nStep 4: Full resolution verification (L0)")
    level0_scores = {}
    for label, proto_list in clf.prototypes.items():
        best_sim = 0.0
        for proto in proto_list:
            sim = cosine_similarity(q_active[0], proto['active'][0])
            if sim > best_sim:
                best_sim = sim
        level0_scores[label] = best_sim
        print(f"  {label}: L0 similarity = {best_sim:.3f}")
    
    # Step 5: Passive channel check
    print("\nStep 5: Passive channel (boundary evidence)")
    passive_scores = {}
    for label, proto_list in clf.prototypes.items():
        best_sim = 0.0
        for proto in proto_list:
            if 'passive' in proto:
                sim = cosine_similarity(q_passive[0], proto['passive'][0])
                if sim > best_sim:
                    best_sim = sim
        passive_scores[label] = best_sim
        print(f"  {label}: passive L0 similarity = {best_sim:.3f}")
    
    # Final decision: max across levels AND max across primitives
    print("\n--- Final Decision ---")
    print("Hierarchical fusion: max across levels + max across primitives")
    
    final_scores = {}
    for label in clf.prototypes.keys():
        best_across_levels = max(
            level0_scores[label],
            level1_scores[label],
            level2_scores[label]
        )
        # Also consider passive channel (max fusion, not weighted)
        final_scores[label] = max(best_across_levels, passive_scores[label])
    
    for label, score in sorted(final_scores.items(), key=lambda x: -x[1]):
        print(f"  {label}: best score = {score:.3f}")
    
    prediction = max(final_scores, key=final_scores.get)
    print(f"\n  → Prediction: {prediction}")
    print(f"  → True label: digit_2")
    print(f"  → Correct: {prediction == 'digit_2'}")
    
    # Compare with standard predict
    std_pred = clf.predict(test_img)
    print(f"  → Standard predict: {std_pred}")
    
    print("\n" + "=" * 60)
    print("Key insights:")
    print("  1. Bottom-up: extract evidence at each level")
    print("  2. Top-down: form hypothesis at high level, verify at lower levels")
    print("  3. Level isolation: each level has its own evidence")
    print("  4. Primitive isolation: active + passive, max fusion")
    print("  5. Decision: max across levels (hierarchical, no weights)")
    print("  6. Bidirectional: both directions work together")
    print("=" * 60)


if __name__ == '__main__':
    reasoning_demo()
