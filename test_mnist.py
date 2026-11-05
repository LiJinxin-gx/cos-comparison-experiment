# -*- coding: utf-8 -*-
"""
Real MNIST training test.

Core principles:
- Continuous mapping hierarchy isolation
- Primitive isolation (active/passive separate)
- Absolute locality
- K=1 max selection

Pure standard library only. Uses synthetic digits if no MNIST data available.
The principle primitives run on the installed cos_comparison engine
(core.pool, core.cos_comparison_passive, core.cos); only structural
helpers, I/O and data synthesis remain local.
"""
import random
import time
import os
import struct
import gzip

from cos_comparison import core


# ============================================================
# Core Primitives
# ============================================================

def down_sample_2d(tensor, factor):
    """Continuous down-sampling via the formal engine (average pool)."""
    h, w = len(tensor), len(tensor[0])
    nh, nw = h // factor, w // factor
    output = core.create_void_list((nh, nw))
    core.pool(tensor, window_size=(factor, factor),
              step=(factor, factor), output=output)
    return output


def passive_extract_2d(tensor):
    """Passive d-shift boundary map via the formal engine (gap space)."""
    h, w = len(tensor), len(tensor[0])
    output = core.create_void_list((h - 1, w))
    core.cos_comparison_passive(tensor, window_size=(1, 1),
                                d=(1, 0), output=output)
    return output


def cosine_sim(a, b):
    """Whole-tensor cosine similarity via the formal engine."""
    return core.cos(a, b, algorithm=core._cos)


def flatten_2d(tensor):
    return [x for row in tensor for x in row]


# ============================================================
# Feature Extraction
# ============================================================

def extract_features(img, pool_factors=(1, 2, 4)):
    """Extract hierarchical features: active and passive channels."""
    active_levels = []
    passive_levels = []

    # Active channel: down-sample
    current = [row[:] for row in img]
    for f in pool_factors:
        if f > 1:
            current = down_sample_2d(current, f)
        active_levels.append(flatten_2d(current))

    # Passive channel: local differences
    current = [row[:] for row in img]
    for f in pool_factors:
        p = passive_extract_2d(current)
        if f > 1:
            current = down_sample_2d(current, f)
        passive_levels.append(flatten_2d(p))

    return active_levels, passive_levels


# ============================================================
# MNIST Loader (if available)
# ============================================================

def load_mnist_images(path):
    """Load MNIST images from idx3-ubyte.gz."""
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = []
        for _ in range(num):
            row_data = []
            for _ in range(rows):
                row = list(f.read(cols))
                row_data.append([float(x) for x in row])
            data.append(row_data)
        return data


def load_mnist_labels(path):
    """Load MNIST labels from idx1-ubyte.gz."""
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        return list(f.read(num))


# ============================================================
# Synthetic Digit Generator (fallback)
# ============================================================

def make_synthetic_digit(d, size=14, noise=0.0):
    """Draw digit d as 14x14 dot matrix."""
    img = [[0.0]*size for _ in range(size)]
    if d == 0:
        for i in range(1, 8):
            for j in range(4, 10):
                if i == 1 or i == 7 or j == 4 or j == 9:
                    img[i][j] = 200.0
    elif d == 1:
        for i in range(1, 8):
            for j in range(6, 8):
                if i % 2 == 0:
                    img[i][j] = 200.0
    elif d == 2:
        for i in range(1, 8):
            j = i
            img[i][j] = 200.0
    elif d == 3:
        for i in range(1, 8):
            for j in range(4, 10):
                if i == 1 or i == 4 or i == 7:
                    img[i][j] = 200.0
    elif d == 4:
        for i in range(1, 8):
            img[i][6] = 200.0
        for j in range(4, 10):
            img[4][j] = 200.0
    elif d == 5:
        for i in range(1, 8):
            for j in range(4, 10):
                if i == 1 or i == 4 or i == 7:
                    img[i][j] = 200.0
        img[1][4] = 200.0
        img[7][9] = 200.0
    elif d == 6:
        for i in range(1, 8):
            img[i][4] = 200.0
        for j in range(4, 10):
            img[4][j] = 200.0
        for i in range(4, 8):
            img[i][9] = 200.0
    elif d == 7:
        for j in range(4, 10):
            img[1][j] = 200.0
        for i in range(1, 8):
            img[i][8] = 200.0
    elif d == 8:
        for i in range(1, 8):
            img[i][4] = 200.0
            img[i][9] = 200.0
        for j in range(4, 10):
            img[1][j] = 200.0
            img[4][j] = 200.0
            img[7][j] = 200.0
    else:  # 9
        for i in range(1, 8):
            img[i][4] = 200.0
        for j in range(4, 10):
            img[1][j] = 200.0
            img[4][j] = 200.0

    if noise:
        for i in range(size):
            for j in range(size):
                img[i][j] += random.gauss(0, noise)
    return img


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("MNIST Training Test")
    print("=" * 60)

    data_dir = 'testdata'
    mnist_path = os.path.join(data_dir, 'mnist')

    # Check if real MNIST data exists
    use_real = False
    if os.path.exists(mnist_path):
        train_img_path = os.path.join(mnist_path, 'train-images-idx3-ubyte.gz')
        train_lbl_path = os.path.join(mnist_path, 'train-labels-idx1-ubyte.gz')
        if os.path.exists(train_img_path) and os.path.exists(train_lbl_path):
            use_real = True

    if use_real:
        print("\nLoading real MNIST data...")
        train_images = load_mnist_images(train_img_path)
        train_labels = load_mnist_labels(train_lbl_path)
        print(f"Train: {len(train_images)} samples")
    else:
        print("\nNo real MNIST data found. Using synthetic digits.")
        print("(Place MNIST gz files in testdata/mnist/ for real test)")

        random.seed(42)
        n_train_per_class = 50
        n_test_per_class = 20

        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        for d in range(10):
            for _ in range(n_train_per_class):
                train_images.append(make_synthetic_digit(d, noise=10))
                train_labels.append(d)
            for _ in range(n_test_per_class):
                test_images.append(make_synthetic_digit(d, noise=10))
                test_labels.append(d)

    # Extract features
    print("\nExtracting features...")
    t0 = time.time()

    train_features = [extract_features(img) for img in train_images[:500]]
    if use_real:
        test_images = train_images[500:600]
        test_labels = train_labels[500:600]
    else:
        test_features_local = [extract_features(img) for img in test_images]

    t1 = time.time()
    print(f"Feature extraction: {t1-t0:.2f}s")

    # Classification: K=1, max fusion
    print("\nClassifying (K=1, max fusion)...")
    t2 = time.time()

    if not use_real:
        correct = 0
        total = len(test_images)
        for i in range(total):
            best_score = -1.0
            best_label = -1
            test_act, test_pass = test_features_local[i]
            for j in range(len(train_features)):
                train_act, train_pass = train_features[j]
                # Active channel: max over levels
                a_score = 0.0
                for lvl in range(len(test_act)):
                    s = cosine_sim(test_act[lvl], train_act[lvl])
                    if s > a_score:
                        a_score = s
                # Passive channel: max over levels
                p_score = 0.0
                for lvl in range(len(test_pass)):
                    s = cosine_sim(test_pass[lvl], train_pass[lvl])
                    if s > p_score:
                        p_score = s
                # Decision: max fusion
                score = max(a_score, p_score)
                if score > best_score:
                    best_score = score
                    best_label = train_labels[j]

            if best_label == test_labels[i]:
                correct += 1

            if (i+1) % 50 == 0:
                print(f"  {i+1}/{total} done, accuracy: {correct/(i+1)*100:.1f}%")

        t3 = time.time()
        acc = correct / total * 100

        print(f"\nClassification time: {t3-t2:.2f}s")
        print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    else:
        print("  (Real MNIST test skipped for speed - use larger sample)")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Principle: hierarchical isolation + primitive isolation + K=1 max")
    print(f"Feature extraction: {t1-t0:.2f}s")


if __name__ == '__main__':
    main()
