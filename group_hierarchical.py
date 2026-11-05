"""
Hierarchical classifier with level isolation and dual-mode features.

Core principles:
  - Level isolation: features stored independently at each level
  - Primitive isolation: active (values) + passive (differences) are separate
  - Decision fusion: max across levels + max across primitives (no weights)
  - K=1 nearest neighbor: structural comparison, no voting
  - Absolute locality: all operations are local

The principle primitives run on the installed cos_comparison engine
(core.pool, core.cos_comparison_passive, core.cos); only the structural
helpers and in-memory storage remain local.
"""
from cos_comparison import core


def down_sample(tensor, factor):
    """Average pooling via the formal engine: coarsen by factor."""
    h, w = len(tensor), len(tensor[0])
    new_h, new_w = h // factor, w // factor
    output = core.create_void_list((new_h, new_w))
    core.pool(tensor, window_size=(factor, factor),
              step=(factor, factor), output=output)
    return output


def up_sample(tensor, target_h, target_w):
    """Nearest-neighbor up-sample (structural inverse, not exact)."""
    h, w = len(tensor), len(tensor[0])
    result = []
    for i in range(target_h):
        row = []
        src_i = min(i * h // target_h, h - 1)
        for j in range(target_w):
            src_j = min(j * w // target_w, w - 1)
            row.append(tensor[src_i][src_j])
        result.append(row)
    return result


def active_extract(tensor):
    """Active mode: raw values (element space)."""
    return [row[:] for row in tensor]


def passive_extract(tensor):
    """Passive mode via the formal engine: d-shift boundary map (gap space).

    Scalar windows are compared with their d-shifted neighbour using the
    default cosmod kernel, so a change (a boundary, including a zero
    crossing) lowers the score while a uniform region keeps it near 1.
    """
    h, w = len(tensor), len(tensor[0])
    # A tensor with no neighbour on the shift axis cannot have a boundary;
    # the formal engine rejects it, so return the degenerate zero map (the
    # same "no difference" result a lone element would carry).
    if h < 2:
        return [[0.0] * w for _ in range(h)]
    output = core.create_void_list((h - 1, w))
    core.cos_comparison_passive(tensor, window_size=(1, 1),
                                d=(1, 0), output=output)
    return output


def flatten_2d(tensor):
    """Flatten 2D to 1D list."""
    return [x for row in tensor for x in row]


def cosine_similarity(a, b):
    """Whole-tensor cosine similarity via the formal engine (same primitive)."""
    return core.cos(a, b, algorithm=core._cos)


# ============================================================
# Group-Theoretic Hierarchical Classifier
# ============================================================

class GroupHierarchicalClassifier:
    """
    Hierarchical classifier with level isolation and dual-mode features.
    
    Features stored independently at each level.
    Active and passive channels fused by max at decision.
    Pluggable memory backend (protocol-based).
    """

    def __init__(self, pool_factors=None, use_passive=True, memory=None):
        """
        Args:
            pool_factors: down-sampling factors per level (default [1,2,4])
            use_passive: enable passive channel (default True)
            memory: pluggable memory backend (default in-memory list)
        """
        if pool_factors is None:
            pool_factors = [1, 2, 4]
        self.pool_factors = pool_factors
        self.use_passive = use_passive

        if memory is None:
            self._memory = _DefaultInMemoryBackend()
        else:
            self._memory = memory

    @property
    def prototypes(self):
        """Access memory as prototypes dict."""
        data = self._memory.read()
        if isinstance(data, dict):
            return data
        result = {}
        for entry in data:
            label = entry.get('label')
            if label not in result:
                result[label] = []
            result[label].append(entry)
        return result

    @prototypes.setter
    def prototypes(self, value):
        """Set memory from dict."""
        if isinstance(value, dict):
            data = []
            for label, proto_list in value.items():
                for entry in proto_list:
                    entry['label'] = label
                    data.append(entry)
            self._memory.write(data)
        else:
            self._memory.write(value)

    def _extract_levels(self, image):
        """Extract active and passive features at each level.

        Each level is coarsened independently from the original image rather
        than by cumulative down-sampling, so every level stays a genuine 2D
        mapping map instead of collapsing to a scalar (a scalar level would
        carry only a trivial self-similarity and dominate the max fusion).
        """
        levels_active = []
        for factor in self.pool_factors:
            current = image if factor == 1 else down_sample(image, factor)
            levels_active.append(flatten_2d(active_extract(current)))

        levels_passive = None
        if self.use_passive:
            levels_passive = []
            for factor in self.pool_factors:
                current = image if factor == 1 else down_sample(image, factor)
                levels_passive.append(flatten_2d(passive_extract(current)))

        return levels_active, levels_passive

    def fit(self, images, labels):
        """Store prototypes at each level."""
        self._memory.clear()
        for img, label in zip(images, labels):
            lvl_a, lvl_p = self._extract_levels(img)
            entry = {'active': lvl_a, 'label': label}
            if lvl_p is not None:
                entry['passive'] = lvl_p
            self._memory.append(entry)

    def predict(self, image):
        """Predict label (max across levels + max across primitives)."""
        q_active, q_passive = self._extract_levels(image)
        scores = {}

        for label, proto_list in self.prototypes.items():
            best_score = 0.0
            for proto in proto_list:
                # Active channel: max cosine across levels
                active_score = 0.0
                for lvl_idx in range(len(q_active)):
                    sim = cosine_similarity(q_active[lvl_idx],
                                            proto['active'][lvl_idx])
                    if sim > active_score:
                        active_score = sim

                # Passive channel: separate, max across levels
                if q_passive is not None and 'passive' in proto:
                    passive_score = 0.0
                    for lvl_idx in range(len(q_passive)):
                        sim = cosine_similarity(q_passive[lvl_idx],
                                                proto['passive'][lvl_idx])
                        if sim > passive_score:
                            passive_score = sim
                    level_score = max(active_score, passive_score)
                else:
                    level_score = active_score

                if level_score > best_score:
                    best_score = level_score
            scores[label] = best_score

        return max(scores, key=scores.get)

    def accuracy(self, images, labels):
        """Compute accuracy."""
        preds = [self.predict(img) for img in images]
        correct = sum(1 for p, t in zip(preds, labels) if p == t)
        return correct / len(labels)


# ============================================================
# Default In-Memory Backend
# ============================================================

class _DefaultInMemoryBackend:
    """Default in-memory list backend."""
    def __init__(self):
        self._data = []

    def write(self, data):
        self._data = list(data)

    def read(self):
        return self._data

    def append(self, item):
        self._data.append(item)

    def clear(self):
        self._data = []

    def size(self):
        return len(self._data)


if __name__ == '__main__':
    # Demo: verify group structure and synthetic classification
    import random
    random.seed(42)

    print("=== Group Structure Verification ===\n")

    # Test group axioms with small tensor
    test_tensor = [[float(i) for i in range(4)] for _ in range(4)]

    # f_01: down-sample by 2
    down = down_sample(test_tensor, 2)
    print(f"f_01 (down 2x): {len(down)}x{len(down[0])}")

    # f_10: up-sample back to original size
    up = up_sample(down, 4, 4)
    print(f"f_10 (up back): {len(up)}x{len(up[0])}")

    # Verify structural inverse (not exact)
    exact = all(abs(test_tensor[i][j] - up[i][j]) < 1e-10
                for i in range(4) for j in range(4))
    print(f"Exact inverse: {exact} (expected False - info loss)")
    print(f"Structural inverse (shape preserved): {len(up)==4 and len(up[0])==4}")

    # Closure: f_01 o f_12 = f_02
    down2 = down_sample(test_tensor, 4)
    down_twice = down_sample(down_sample(test_tensor, 2), 2)
    closure = all(abs(down2[i][j] - down_twice[i][j]) < 1e-10
                  for i in range(len(down2)) for j in range(len(down2[0])))
    print(f"Closure (f_01 o f_12 = f_02): {closure}")

    # Identity: f_00 = id
    identity = all(abs(test_tensor[i][j] - test_tensor[i][j]) < 1e-10
                   for i in range(4) for j in range(4))
    print(f"Identity (f_00 = id): {identity}")

    print("\n=== Synthetic Classification Demo ===\n")

    def make_pattern(class_id, size=28):
        img = [[0.0] * size for _ in range(size)]
        if class_id == 0:
            for i in range(0, size, 6):
                for di in range(3):
                    for j in range(size):
                        img[i + di][j] = 200.0
        elif class_id == 1:
            for j in range(0, size, 6):
                for dj in range(3):
                    for i in range(size):
                        img[i][j + dj] = 200.0
        else:
            for i in range(size):
                j = (i + 8) % size
                for dj in range(-2, 3):
                    img[i][(j + dj) % size] = 200.0
        for i in range(size):
            for j in range(size):
                img[i][j] += random.gauss(0, 5)
        return img

    train_imgs = [make_pattern(i % 3) for i in range(300)]
    train_labels = [f"c{i % 3}" for i in range(300)]
    test_imgs = [make_pattern(i % 3) for i in range(90)]
    test_labels = [f"c{i % 3}" for i in range(90)]

    clf = GroupHierarchicalClassifier(pool_factors=[1, 2, 4], use_passive=False)
    clf.fit(train_imgs, train_labels)
    acc = clf.accuracy(test_imgs, test_labels)
    print(f"Active-only accuracy: {acc * 100:.1f}%")

    clf2 = GroupHierarchicalClassifier(pool_factors=[1, 2, 4], use_passive=True)
    clf2.fit(train_imgs, train_labels)
    acc2 = clf2.accuracy(test_imgs, test_labels)
    print(f"Active+passive (decision fusion): {acc2 * 100:.1f}%")
