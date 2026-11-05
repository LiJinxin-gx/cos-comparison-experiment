# -*- coding: utf-8 -*-
"""
Batch heatmap analysis: generate and analyze 1400+ heatmaps in memory.
Core principles: continuous mapping hierarchy isolation, primitive isolation,
absolute locality, K=1 max selection.

The principle primitives run on the installed cos_comparison engine
(core.pool, core.cos_comparison_passive, core.cos); only the structural
up-sampling and flatten helpers remain local.
"""
import math
import random
from collections import defaultdict

from cos_comparison import core

random.seed(42)


# ============================================================
# Core Primitives
# ============================================================

def down_sample(tensor, factor):
    h, w = len(tensor), len(tensor[0])
    nh, nw = h // factor, w // factor
    output = core.create_void_list((nh, nw))
    core.pool(tensor, window_size=(factor, factor),
              step=(factor, factor), output=output)
    return output


def up_sample(tensor, th, tw):
    h, w = len(tensor), len(tensor[0])
    return [[tensor[min(i*h//th, h-1)][min(j*w//tw, w-1)] for j in range(tw)] for i in range(th)]


def passive_extract(tensor):
    h, w = len(tensor), len(tensor[0])
    output = core.create_void_list((h - 1, w))
    core.cos_comparison_passive(tensor, window_size=(1, 1),
                                d=(1, 0), output=output)
    return output


def flatten(tensor):
    return [x for row in tensor for x in row]


def cosine_sim(a, b):
    return core.cos(a, b, algorithm=core._cos)


# ============================================================
# Synthetic Digits
# ============================================================

def make_digit(digit_id, size=28):
    img = [[0.0]*size for _ in range(size)]
    if digit_id == 0:  # circle
        cx, cy, r = size//2, size//2, size//3
        for i in range(size):
            for j in range(size):
                d = math.sqrt((i-cx)**2 + (j-cy)**2)
                if abs(d-r) < 2: img[i][j] = 200.0
    elif digit_id == 1:  # vertical
        for i in range(4, size-4):
            for j in range(size//2-2, size//2+2): img[i][j] = 200.0
    elif digit_id == 2:  # horizontal
        for i in range(size//2-2, size//2+2):
            for j in range(4, size-4): img[i][j] = 200.0
    elif digit_id == 3:  # diagonal
        for i in range(size):
            j = int(i*0.8)
            for dj in range(-2, 3):
                if 0 <= j+dj < size: img[i][j+dj] = 200.0
    elif digit_id == 4:  # cross
        for i in range(size//2-2, size//2+2):
            for j in range(4, size-4): img[i][j] = 200.0
        for j in range(size//2-2, size//2+2):
            for i in range(4, size-4): img[i][j] = 200.0
    elif digit_id == 5:  # L
        for i in range(4, size-4): img[i][6] = 200.0
        for j in range(6, size-6): img[size-6][j] = 200.0
    elif digit_id == 6:  # U
        for i in range(4, size-4):
            img[i][6] = 200.0; img[i][size-6] = 200.0
        for j in range(6, size-6): img[size-6][j] = 200.0
    elif digit_id == 7:  # rectangle border
        for i in range(4, size-4):
            img[i][4] = 200.0; img[i][size-5] = 200.0
        for j in range(4, size-4):
            img[4][j] = 200.0; img[size-5][j] = 200.0
    elif digit_id == 8:  # two circles
        cx1, cy1 = size//3, size//2
        cx2, cy2 = 2*size//3, size//2
        for i in range(size):
            for j in range(size):
                d1 = math.sqrt((i-cy1)**2 + (j-cx1)**2)
                d2 = math.sqrt((i-cy2)**2 + (j-cx2)**2)
                if abs(d1-size//4) < 2 or abs(d2-size//4) < 2: img[i][j] = 200.0
    else:  # triangle
        for i in range(size//2, size-4):
            width = int((i-size//2)*0.8)
            for j in range(size//2-width, size//2+width):
                if 0 <= j < size: img[i][j] = 200.0
    for i in range(size):
        for j in range(size):
            img[i][j] += random.gauss(0, 8)
    return img


# ============================================================
# Heatmap Statistics
# ============================================================

def heatmap_stats(tensor, name):
    """Compute statistics of a heatmap."""
    flat = flatten(tensor)
    n = len(flat)
    s = sum(flat)
    s2 = sum(x*x for x in flat)
    mean = s/n
    var = s2/n - mean*mean
    mx = max(flat)
    mn = min(flat)
    nonzero = sum(1 for x in flat if x > 0.01*mx)
    return {
        'name': name,
        'shape': f'{len(tensor)}x{len(tensor[0])}',
        'mean': mean,
        'std': math.sqrt(max(var, 0)),
        'max': mx,
        'min': mn,
        'nonzero_ratio': nonzero/n,
    }


# ============================================================
# Main Analysis
# ============================================================

def main():
    n_digits = 10
    n_per_digit = 20  # 200 samples
    pool_factors = [1, 2, 4]

    print(f"=== Generating {n_per_digit * n_digits} samples ===")
    print(f"=== Each sample: 7 heatmaps = {n_per_digit * n_digits * 7} total heatmaps ===\n")

    # Collect all heatmap statistics
    all_stats = defaultdict(list)
    train_features = []
    train_labels = []

    for d in range(n_digits):
        for s in range(n_per_digit):
            img = make_digit(d)

            # L0: original (active)
            st = heatmap_stats(img, f'L0_original')
            all_stats['L0_original'].append(st)

            # L1: passive (local differences)
            p1 = passive_extract(img)
            st = heatmap_stats(p1, f'L1_passive')
            all_stats['L1_passive'].append(st)

            # L2: downsample 2x (active)
            d2 = down_sample(img, 2)
            st = heatmap_stats(d2, f'L2_active_pool2')
            all_stats['L2_active_pool2'].append(st)

            # L2: passive on L2
            p2 = passive_extract(d2)
            st = heatmap_stats(p2, f'L2_passive_pool2')
            all_stats['L2_passive_pool2'].append(st)

            # L3: downsample 4x (active)
            d3 = down_sample(img, 4)
            st = heatmap_stats(d3, f'L3_active_pool4')
            all_stats['L3_active_pool4'].append(st)

            # L3: passive on L3
            p3 = passive_extract(d3)
            st = heatmap_stats(p3, f'L3_passive_pool4')
            all_stats['L3_passive_pool4'].append(st)

            # Structural inverse test
            u3 = up_sample(d3, len(img), len(img[0]))
            st = heatmap_stats(u3, f'L3_inv_up4')
            all_stats['L3_inv_up4'].append(st)

            # Store features for classification
            active_levels = []
            current = img
            for f in pool_factors:
                if f > 1: current = down_sample(current, f)
                active_levels.append(flatten(current))
            passive_levels = []
            current = img
            for f in pool_factors:
                if f > 1: current = down_sample(current, f)
                passive_levels.append(flatten(passive_extract(current)))
            train_features.append((active_levels, passive_levels))
            train_labels.append(d)

    # ============================================================
    # Heatmap Summary
    # ============================================================
    print("=== Heatmap Statistics Summary (200 samples x 7 types = 1400 heatmaps) ===\n")
    print(f"{'Type':<25} {'Shape':<10} {'Mean':>8} {'Std':>8} {'Max':>8} {'Nonzero':>10}")
    print("-" * 75)

    for type_name in sorted(all_stats.keys()):
        stats_list = all_stats[type_name]
        n = len(stats_list)
        avg_mean = sum(s['mean'] for s in stats_list) / n
        avg_std = sum(s['std'] for s in stats_list) / n
        avg_max = sum(s['max'] for s in stats_list) / n
        avg_nz = sum(s['nonzero_ratio'] for s in stats_list) / n
        shape = stats_list[0]['shape']
        print(f"{type_name:<25} {shape:<10} {avg_mean:>8.3f} {avg_std:>8.3f} {avg_max:>8.1f} {avg_nz:>10.3f}")

    # ============================================================
    # Classification Test
    # ============================================================
    print("\n=== Classification Test (K=1, max fusion) ===\n")

    test_features = []
    test_labels = []
    for d in range(n_digits):
        for s in range(5):  # 50 test samples
            img = make_digit(d)
            active_levels = []
            current = img
            for f in pool_factors:
                if f > 1: current = down_sample(current, f)
                active_levels.append(flatten(current))
            passive_levels = []
            current = img
            for f in pool_factors:
                if f > 1: current = down_sample(current, f)
                passive_levels.append(flatten(passive_extract(current)))
            test_features.append((active_levels, passive_levels))
            test_labels.append(d)

    # Active only
    correct_a = 0
    for tf, tl in zip(test_features, test_labels):
        best_score = -1
        best_label = -1
        for j, (trf, trl) in enumerate(zip(train_features, train_labels)):
            score = 0.0
            for lvl in range(3):
                s = cosine_sim(tf[0][lvl], trf[0][lvl])
                if s > score: score = s
            if score > best_score:
                best_score = score
                best_label = trl
        if best_label == tl: correct_a += 1

    # Passive only
    correct_p = 0
    for tf, tl in zip(test_features, test_labels):
        best_score = -1
        best_label = -1
        for j, (trf, trl) in enumerate(zip(train_features, train_labels)):
            score = 0.0
            for lvl in range(3):
                s = cosine_sim(tf[1][lvl], trf[1][lvl])
                if s > score: score = s
            if score > best_score:
                best_score = score
                best_label = trl
        if best_label == tl: correct_p += 1

    # Max fusion
    correct_f = 0
    for tf, tl in zip(test_features, test_labels):
        best_score = -1
        best_label = -1
        for j, (trf, trl) in enumerate(zip(train_features, train_labels)):
            a_score = 0.0
            for lvl in range(3):
                s = cosine_sim(tf[0][lvl], trf[0][lvl])
                if s > a_score: a_score = s
            p_score = 0.0
            for lvl in range(3):
                s = cosine_sim(tf[1][lvl], trf[1][lvl])
                if s > p_score: p_score = s
            score = max(a_score, p_score)
            if score > best_score:
                best_score = score
                best_label = trl
        if best_label == tl: correct_f += 1

    n_test = len(test_labels)
    print(f"Active only:    {correct_a}/{n_test} = {correct_a/n_test*100:.1f}%")
    print(f"Passive only:   {correct_p}/{n_test} = {correct_p/n_test*100:.1f}%")
    print(f"Max fusion:     {correct_f}/{n_test} = {correct_f/n_test*100:.1f}%")

    # ============================================================
    # Per-digit Analysis
    # ============================================================
    print("\n=== Per-Digit Accuracy (Max Fusion) ===\n")
    per_digit = defaultdict(lambda: [0, 0])
    for tf, tl in zip(test_features, test_labels):
        best_score = -1
        best_label = -1
        for j, (trf, trl) in enumerate(zip(train_features, train_labels)):
            a_score = max(cosine_sim(tf[0][lvl], trf[0][lvl]) for lvl in range(3))
            p_score = max(cosine_sim(tf[1][lvl], trf[1][lvl]) for lvl in range(3))
            score = max(a_score, p_score)
            if score > best_score:
                best_score = score
                best_label = trl
        per_digit[tl][1] += 1
        if best_label == tl:
            per_digit[tl][0] += 1

    for d in range(n_digits):
        c, t = per_digit[d]
        print(f"Digit {d}: {c}/{t} = {c/t*100:.0f}%")

    # ============================================================
    # Layer-wise Analysis
    # ============================================================
    print("\n=== Layer-wise Contribution Analysis ===\n")
    for lvl in range(3):
        correct_l = 0
        for tf, tl in zip(test_features, test_labels):
            best_score = -1
            best_label = -1
            for j, (trf, trl) in enumerate(zip(train_features, train_labels)):
                s = cosine_sim(tf[0][lvl], trf[0][lvl])
                if s > best_score:
                    best_score = s
                    best_label = trl
            if best_label == tl: correct_l += 1
        print(f"L{lvl+1} active only: {correct_l}/{n_test} = {correct_l/n_test*100:.1f}%")

    print("\n=== Done! ===")


if __name__ == '__main__':
    main()
