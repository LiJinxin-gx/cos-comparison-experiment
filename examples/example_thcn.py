"""
Example: Pure Template Network (THCN) — classification and generation.

Key point: NO residual connections. Variation is captured by multiple
templates per class per level. Each data point = template path (sequence
of nearest template indices per level).

Demonstrates:
  - Training from tensor data
  - Classification via hierarchical weighted voting
  - Generation from template paths (cross-class allowed)
"""
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms import PureTemplateNetwork


def make_sample(base, variation, size=64):
    """Create a sample: base pattern + variation noise."""
    return [(base[i % len(base)] + variation * ((i * 7) % 5) / 10.0)
            for i in range(size)]


def main():
    print("=== Generating Synthetic Data ===")
    # Two "classes" with different base patterns
    class_a_base = [1.0, 0.0, 1.0, 0.0]  # alternating
    class_b_base = [1.0, 1.0, 0.0, 0.0]  # blocks

    train_data = {
        "A": [make_sample(class_a_base, v) for v in range(5)],
        "B": [make_sample(class_b_base, v) for v in range(5)],
    }
    print(f"Class A: {len(train_data['A'])} samples, pattern={class_a_base}")
    print(f"Class B: {len(train_data['B'])} samples, pattern={class_b_base}")

    print("\n=== Training Pure Template Network ===")
    net = PureTemplateNetwork(levels=(64, 32, 16, 8), k_per_level=2)
    net.train(train_data)
    print(f"Classes: {net.classes}")
    for label in net.classes:
        print(f"  {label}: templates per level = "
              f"{[len(net.templates[label][lv]) for lv in range(4)]}")

    print("\n=== Classification ===")
    test_a = make_sample(class_a_base, 99)  # unseen variation
    test_b = make_sample(class_b_base, 99)
    pred_a, score_a = net.classify(test_a)
    pred_b, score_b = net.classify(test_b)
    print(f"Test sample (A pattern): predicted '{pred_a}' (score={score_a:.4f})")
    print(f"Test sample (B pattern): predicted '{pred_b}' (score={score_b:.4f})")

    print("\n=== Template Path ===")
    path = net.get_template_path(test_a)
    print(f"Template path for A sample: {path}")

    print("\n=== Generation from Template Path ===")
    # Generate using A's fine templates and B's coarse templates (cross-class)
    mixed_path = [
        ("A", 0),  # fine level: A
        ("A", 0),
        ("B", 0),  # coarse levels: B
        ("B", 0),
    ]
    generated = net.generate(mixed_path, alpha=0.6)
    print(f"Generated (A-fine + B-coarse): length={len(generated)}")
    print(f"  First 16 values: {[f'{v:.3f}' for v in generated[:16]]}")

    # Verify generation consistency
    pred_gen, score_gen = net.classify(generated)
    print(f"  Generated sample classified as: '{pred_gen}' (score={score_gen:.4f})")


if __name__ == "__main__":
    main()
