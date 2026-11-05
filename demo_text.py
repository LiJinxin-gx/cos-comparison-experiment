"""
Text Classification Demo: frequency encoding + hierarchical matching.

Demonstrates:
1. Text -> 1D tensor via character frequency (unit_map principle)
2. 3-level hierarchical pooling on 1D sequence
3. Active cosine matching, K=1 nearest neighbor
4. Level isolation: take max across levels (no weighted fusion)

Data loading separated from processing:
- load_text(): file -> string
- encode_text(): string -> 1D tensor (frequency map)
- HierarchicalTextClassifier1D: tensor -> classification

Pure standard library only.
The principle primitives run on the installed cos_comparison engine
(core.pool, core.cos_comparison_passive, core.cos); only text loading and
structural helpers remain local.
"""
import os
import sys

from cos_comparison import core


def load_text(path):
    """Load text file as string."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def encode_text_sequence(text, max_len=512):
    """
    Encode text as 1D raw sequence tensor.
    Preserves local structure - information arises from local comparison,
    not global frequency statistics.

    Each character is mapped to its ordinal value (normalized).
    The sequence itself is the tensor; passive mode extracts local differences.
    """
    seq = [0.0] * max_len
    for i, ch in enumerate(text[:max_len]):
        seq[i] = ord(ch) / 255.0  # normalize to [0, 1]
    return seq


def passive_extract_1d(seq):
    """Passive d-shift boundary map via the formal engine (gap space)."""
    output = core.create_void_list((len(seq) - 1,))
    core.cos_comparison_passive(seq, window_size=(1,),
                                d=(1,), output=output)
    return output


def pool_1d(flat, factor):
    """1D average pooling via the formal engine (continuous down-sampling)."""
    new_n = len(flat) // factor
    output = core.create_void_list((new_n,))
    core.pool(flat, window_size=(factor,), step=(factor,), output=output)
    return output


def cosine_sim(a, b):
    """Whole-tensor cosine similarity via the formal engine."""
    return core.cos(a, b, algorithm=core._cos)


class HierarchicalTextClassifier1D:
    """
    1D hierarchical classifier for text sequences.

    Group-theoretic principles:
    - 3 pooling levels [1,2,4] (continuous down-sampling)
    - Active (raw sequence) + passive (local differences) channels
    - K=1 nearest neighbor
    - Level isolation: take max across levels (no weighted fusion)
    - Primitive isolation: active/passive never averaged, max at decision
    """

    def __init__(self, pool_factors=None):
        if pool_factors is None:
            pool_factors = [1, 2, 4]
        self.pool_factors = pool_factors
        self.prototypes = {}

    def _extract_levels(self, seq):
        """Extract active and passive features at each level independently."""
        active_levels = []
        passive_levels = []

        # Active channel: raw sequence, down-sampled
        current = seq
        for factor in self.pool_factors:
            if factor > 1:
                current = pool_1d(current, factor)
            active_levels.append(list(current))

        # Passive channel: local differences, then down-sampled
        current = passive_extract_1d(seq)
        for factor in self.pool_factors:
            if factor > 1:
                current = pool_1d(current, factor)
            passive_levels.append(list(current))

        return active_levels, passive_levels

    def fit(self, texts, labels):
        """Store prototypes at each level independently."""
        self.prototypes = {}
        for text, label in zip(texts, labels):
            seq = encode_text_sequence(text)
            active_lvls, passive_lvls = self._extract_levels(seq)
            if label not in self.prototypes:
                self.prototypes[label] = []
            self.prototypes[label].append((active_lvls, passive_lvls))

    def predict(self, text):
        """Predict label using max across levels and max across primitives."""
        seq = encode_text_sequence(text)
        query_active, query_passive = self._extract_levels(seq)

        scores = {}
        for label, proto_list in self.prototypes.items():
            best_score = 0.0
            for proto_active, proto_passive in proto_list:
                # Active channel: max over levels
                a_score = 0.0
                for lvl_idx in range(len(query_active)):
                    sim = cosine_sim(query_active[lvl_idx], proto_active[lvl_idx])
                    if sim > a_score:
                        a_score = sim

                # Passive channel: max over levels (separate primitive!)
                p_score = 0.0
                for lvl_idx in range(len(query_passive)):
                    sim = cosine_sim(query_passive[lvl_idx], proto_passive[lvl_idx])
                    if sim > p_score:
                        p_score = sim

                # Decision layer: max fusion (NOT weighted average!)
                level_score = max(a_score, p_score)
                if level_score > best_score:
                    best_score = level_score

            scores[label] = best_score

        return max(scores, key=scores.get)

    def accuracy(self, texts, labels):
        correct = sum(1 for t, l in zip(texts, labels) if self.predict(t) == l)
        return correct / len(labels)


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_text.py <data_dir>")
        print("Expected structure: data_dir/class_name/texts...")
        print("\nRunning synthetic demo...")

        def make_text(category, length=500):
            if category == "news":
                chars = "aeioubcdfghlmnprst " * 3 + "newsreport"
            elif category == "legal":
                chars = "abcdefilmnoprstu " * 3 + "whereasthereof"
            else:
                chars = "aeioursthnl " * 3 + "productbuyprice"
            import random
            random.seed(hash(category) % 2**31)
            return ''.join(random.choice(chars) for _ in range(length))

        train_texts = [make_text(i % 3) for i in range(150)]
        train_labels = [f"cat_{i % 3}" for i in range(150)]
        test_texts = [make_text(i % 3) for i in range(45)]
        test_labels = [f"cat_{i % 3}" for i in range(45)]
    else:
        data_dir = sys.argv[1]
        classes = sorted(d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d)))
        train_texts, train_labels = [], []
        test_texts, test_labels = [], []
        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            files = sorted(os.listdir(cls_dir))[:30]
            for f in files[9:]:
                train_texts.append(load_text(os.path.join(cls_dir, f)))
                train_labels.append(cls)
            for f in files[:9]:
                test_texts.append(load_text(os.path.join(cls_dir, f)))
                test_labels.append(cls)

    print("Training text classifier...")
    clf = HierarchicalTextClassifier1D()
    clf.fit(train_texts, train_labels)

    print("Evaluating...")
    acc = clf.accuracy(test_texts, test_labels)
    print(f"\nAccuracy: {acc * 100:.1f}%")
    print(f"Principle: raw sequence + passive local diff + 3-level pooling + K=1 max fusion")


if __name__ == '__main__':
    main()
