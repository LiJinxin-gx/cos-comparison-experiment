"""
Image Classification Demo: 3-level hierarchical pooling + active cosine matching.

Demonstrates the validated core principles:
1. Continuous mapping level isolation (3 pooling levels [1,2,4])
2. Active mode only (cosine similarity, no passive averaging)
3. K=1 nearest neighbor (no voting, no weighted fusion)
4. Level isolation: take max across levels (no weighted fusion)

Data loading is separated from processing:
- load_image(): file -> 2D tensor (tkinter, standard library only)
- HierarchicalClassifier: tensor -> classification

Pure standard library: no numpy, no third-party deps.
Image loading uses tkinter PhotoImage (GIF/PNG/PGM/PPM).
The principle primitives run on the installed cos_comparison engine
(core.pool, core.cos_comparison_passive, core.cos); only structural
helpers and image loading stay local.
"""
import os
import sys

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
    """Minimal local storage backend (no external dependency)."""

    def store(self, key, value):
        return True

    def retrieve(self, key):
        return None


class GroupHierarchicalClassifier:
    """Group-theoretic hierarchical classifier.

    Level isolation: features are extracted independently at each level.
    Primitive isolation: active (element space) and passive (gap space) are
    never mixed; the decision layer fuses by max. K=1 nearest neighbor.
    """

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


def load_image(path, size=28):
    """
    Load image to 2D grayscale tensor using tkinter.
    Supports GIF, PNG, PGM, PPM formats.

    This is the ONLY data-loading function.
    Processing code receives pure 2D tensors.
    """
    import tkinter
    root = tkinter.Tk()
    root.withdraw()
    try:
        photo = tkinter.PhotoImage(file=path)
        orig_w, orig_h = photo.width(), photo.height()
        result = []
        for i in range(size):
            row = []
            src_y = min(int(i * orig_h / size), orig_h - 1)
            for j in range(size):
                src_x = min(int(j * orig_w / size), orig_w - 1)
                rgb = photo.get(src_x, src_y)
                if isinstance(rgb, str):
                    parts = rgb.split()
                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                else:
                    r, g, b = rgb[0], rgb[1], rgb[2]
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                row.append(gray)
            result.append(row)
        return result
    finally:
        root.destroy()


def load_dataset(data_dir, size=28, n_per_class=50, test_ratio=0.3):
    """
    Load image dataset from directory structure:
        data_dir/
            class1/
                img1.png
                ...
            class2/
                ...

    Returns: (train_images, train_labels, test_images, test_labels)
    """
    classes = sorted(d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)))
    print(f"Found {len(classes)} classes")

    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []
    exts = ('.png', '.gif', '.pgm', '.ppm')

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(exts))[:n_per_class]
        n_test = int(len(files) * test_ratio)
        for f in files[n_test:]:
            img = load_image(os.path.join(cls_dir, f), size=size)
            if img:
                train_imgs.append(img)
                train_labels.append(cls)
        for f in files[:n_test]:
            img = load_image(os.path.join(cls_dir, f), size=size)
            if img:
                test_imgs.append(img)
                test_labels.append(cls)

    print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")
    return train_imgs, train_labels, test_imgs, test_labels


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_image.py <data_dir> [size] [n_per_class]")
        print("\nRunning synthetic demo...")
        import random
        random.seed(42)

        def make_pattern(class_id, size=28):
            img = [[0.0] * size for _ in range(size)]
            if class_id == 0:  # horizontal
                for i in range(0, size, 6):
                    for di in range(3):
                        for j in range(size):
                            img[i + di][j] = 200.0
            elif class_id == 1:  # vertical
                for j in range(0, size, 6):
                    for dj in range(3):
                        for i in range(size):
                            img[i][j + dj] = 200.0
            else:  # diagonal
                for i in range(size):
                    j = (i + 8) % size
                    for dj in range(-2, 3):
                        img[i][(j + dj) % size] = 200.0
            for i in range(size):
                for j in range(size):
                    img[i][j] += random.gauss(0, 5)
            return img

        train_imgs = [make_pattern(i % 3) for i in range(300)]
        train_labels = [f"pattern_{i % 3}" for i in range(300)]
        test_imgs = [make_pattern(i % 3) for i in range(90)]
        test_labels = [f"pattern_{i % 3}" for i in range(90)]
    else:
        data_dir = sys.argv[1]
        size = int(sys.argv[2]) if len(sys.argv) > 2 else 28
        n_per = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        train_imgs, train_labels, test_imgs, test_labels = load_dataset(
            data_dir, size=size, n_per_class=n_per)

    # Train and evaluate
    print("\nTraining hierarchical classifier...")
    clf = GroupHierarchicalClassifier(
        pool_factors=[1, 2, 4],
        use_passive=True  # Passive mode: local difference edges (gap space)
    )
    clf.fit(train_imgs, train_labels)

    print("Evaluating...")
    acc = clf.accuracy(test_imgs, test_labels)
    print(f"\nAccuracy: {acc * 100:.1f}%")
    print(f"Principle: 3-level pooling [1,2,4] + active(element space) + passive(gap space) + K=1 max fusion")


if __name__ == '__main__':
    main()
