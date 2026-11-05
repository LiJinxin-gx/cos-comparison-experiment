# -*- coding: utf-8 -*-
"""
Local k-NN Learner.

The simplest rigorous non-parametric learner:
- Knowledge stored as local prototypes
- Prediction = unweighted mean of k nearest local experiences
- Strictly local: no weights, no gradient, no global function
- Zero-forgetting: new prototypes never overwrite old ones

Core principles:
1. Information arises from local comparison
2. Absolute locality: each prototype is an independent local memory
3. Level isolation: prototypes stored independently, no global averaging
4. K=1 base: k is neighborhood size parameter, not voting weight

Zero dependencies: pure Python standard library.

The neighbor comparison can run on the installed cos_comparison engine
(core.cos) for pattern states. The formal similarities are scale-invariant,
so they degenerate on a one-dimensional coordinate axis (cosine of two
scalars is only their sign); a 1D coordinate regression therefore uses the
local Euclidean metric, which has no formal counterpart. Both paths are
exposed through one learner.
"""
from cos_comparison import core


# ============================================================
# Core Primitives
# ============================================================

def euclidean_distance(a, b):
    """Euclidean distance between two vectors. Local metric (no formal
    counterpart); required for coordinate-space regression."""
    if len(a) != len(b):
        raise ValueError("Vector dimensions must match")
    return core.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def formal_similarity(a, b):
    """Whole-tensor cosine via the formal engine: the local comparison for
    multi-dimensional pattern states."""
    return core.cos(a, b, algorithm=core._cos)


# ============================================================
# Local k-NN Learner
# ============================================================

class LocalKNNLearner:
    """
    Local k-NN world model / predictor.
    
    Principles:
    - Each experience is stored as an independent prototype (zero-forgetting)
    - Prediction: unweighted average of k nearest local prototypes
    - Strictly local: no global function, no weights, no gradient
    - As prototypes grow denser, k-nearest window shrinks effectively
    
    This is the non-parametric memory-and-retrieval system:
    strong as a transparent similarity core.
    """
    
    def __init__(self, k=5, metric='cosine'):
        self.k = k
        # 'cosine' uses the formal engine (pattern states); 'euclidean' uses
        # the local metric (1D coordinate regression).
        self.metric = metric
        self.prototypes = []  # list of (state, action) pairs

    def _score(self, query, state):
        """Higher score means nearer. Formal cosine for pattern states,
        negated Euclidean distance for coordinate states."""
        if self.metric == 'euclidean':
            return -euclidean_distance(query, state)
        return formal_similarity(query, state)
    
    def remember(self, state, action):
        """
        Store a new experience as an independent prototype.
        Zero-forgetting: never overwrite or modify existing prototypes.
        """
        self.prototypes.append((state, action))
    
    def predict(self, query_state):
        """
        Predict by averaging the k nearest local experiences.
        Unweighted: all k neighbors contribute equally.
        """
        if not self.prototypes:
            raise ValueError("No prototypes stored yet")
        
        # Score all prototypes (local comparison; higher means nearer)
        scored = [(self._score(query_state, state), action)
                  for state, action in self.prototypes]

        # Sort by score (descending), take k nearest
        scored.sort(key=lambda x: x[0], reverse=True)
        k_nearest = scored[:self.k]
        
        # Unweighted average (simple local mean). A single neighbour is
        # returned as-is, which also supports non-numeric label actions.
        n = len(k_nearest)
        if n == 0:
            return None
        if n == 1:
            return k_nearest[0][1]

        # Handle scalar or vector actions
        first_action = k_nearest[0][1]
        if isinstance(first_action, (int, float)):
            return sum(a for _, a in k_nearest) / n
        else:
            # Vector action
            dim = len(first_action)
            result = [0.0] * dim
            for _, action in k_nearest:
                for i in range(dim):
                    result[i] += action[i]
            return [x / n for x in result]
    
    @property
    def size(self):
        return len(self.prototypes)


# ============================================================
# Demo / Test
# ============================================================

def test_simple_regression():
    """Test simple 1D regression: y = x^2 + noise"""
    print("=" * 60)
    print("Local k-NN Learner Demo")
    print("=" * 60)
    print("\nCore principles:")
    print("- Local prototypes (zero-forgetting memory)")
    print("- Unweighted k-nearest local mean")
    print("- No global function, no weights, no gradient")
    print("- Strictly local: global is special local")
    print()
    
    import random
    random.seed(42)
    
    # Generate training data: y = x^2.
    # 1D coordinate states degenerate under scale-invariant cosine, so this
    # regression uses the local Euclidean metric.
    learner = LocalKNNLearner(k=5, metric='euclidean')
    
    print("Training (storing prototypes)...")
    for _ in range(200):
        x = random.uniform(-5, 5)
        y = x ** 2 + random.gauss(0, 0.5)
        learner.remember([x], y)
    
    print(f"Stored {learner.size} prototypes")
    
    # Test
    print("\nTesting...")
    test_points = [-4, -2, 0, 2, 4]
    correct = 0
    for x in test_points:
        pred = learner.predict([x])
        true_val = x ** 2
        error = abs(pred - true_val)
        print(f"  x={x:+.1f}: predicted={pred:.2f}, true={true_val:.2f}, error={error:.2f}")
        if error < 1.0:
            correct += 1
    
    print(f"\nAccuracy (error < 1.0): {correct}/{len(test_points)}")
    
    # Zero-forgetting test
    print("\nZero-forgetting test:")
    print(f"  Before adding new region: {learner.size} prototypes")
    for _ in range(20):
        x = random.uniform(10, 12)
        y = x ** 2 + random.gauss(0, 0.5)
        learner.remember([x], y)
    print(f"  After adding new region: {learner.size} prototypes")
    
    # Check old region still works
    pred_old = learner.predict([0])
    print(f"  Prediction at old region (x=0): {pred_old:.2f} (should be ~0)")
    print(f"  Old region unaffected: {'YES' if abs(pred_old) < 2.0 else 'NO'}")

    # Formal-engine path: multi-dimensional pattern states use the package's
    # cosine similarity. The matching pattern must rank above a different one.
    print("\nFormal cosine path (pattern states):")
    pat = LocalKNNLearner(k=1, metric='cosine')
    pat.remember([1.0, 0.0, 1.0], 'A')
    pat.remember([0.0, 1.0, 0.0], 'B')
    query = [0.9, 0.1, 0.9]
    print("  query %s -> nearest label %s (expected A)"
          % (query, pat.predict(query)))

    print("\nPrinciple verified: local k-NN = memory + retrieval")


if __name__ == "__main__":
    test_simple_regression()
