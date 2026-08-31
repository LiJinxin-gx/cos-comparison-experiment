"""
Example: Hierarchical Representation with Isolation and Top-Down Drive.

Demonstrates the foundational principle:
  - Low levels store fine detail, high levels store structure
  - Isolation: irrelevant low-level detail does NOT propagate up
  - Top-down drive: high-level context can recover masked low-level detail
"""
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import HierarchicalRepresentation


def main():
    print("=== Building Hierarchical Representation ===")
    # A 64-element "image" with structure + noise
    base = [float((i // 8) % 2) for i in range(64)]  # block structure
    noise = [0.01 * (i % 7) for i in range(64)]       # fine-grained noise
    tensor = [b + n for b, n in zip(base, noise)]

    hr = HierarchicalRepresentation(tensor, n_levels=4, downsample_factor=2)
    print(f"Level sizes: {[len(hr.get_level(lv)) for lv in range(4)]}")

    print("\n=== Adding Templates ===")
    hr.add_template(0, tensor[:16], label="fine_pattern_A")
    hr.add_template(1, hr.get_level(1)[:8], label="medium_pattern_A")
    hr.add_template(2, hr.get_level(2)[:4], label="coarse_structure")
    hr.add_template(3, hr.get_level(3), label="global_structure")

    print("Templates per level:")
    for lv in range(4):
        print(f"  Level {lv}: {len(hr.levels[lv]['templates'])} templates")

    print("\n=== Isolation: Mask Fine-Grained Noise ===")
    # Isolate positions 0-7 (considered "noise" at level 0)
    hr.isolate(0, list(range(0, 8)))
    report = hr.info_loss_report()
    for r in report:
        print(f"  Level {r['level']}: {r['active']}/{r['size']} active "
              f"({r['isolation_ratio']*100:.0f}% isolated)")

    print("\n=== Top-Down Drive: Recover Masked Detail ===")
    # High-level template at index 0 drives recovery at level 0
    hr.drive(high_level=3, low_level=0, template_indices=[0])
    report = hr.info_loss_report()
    print(f"After drive, Level 0: {report[0]['active']}/{report[0]['size']} active")

    print("\n=== Template Matching at Each Level ===")
    for lv in range(4):
        matches = hr.match_templates(lv)
        print(f"  Level {lv}: {matches}")

    print("\n=== Active Representation (mask applied) ===")
    active = hr.active_representation(0)
    print(f"  First 16 values: {[f'{v:.2f}' for v in active[:16]]}")


if __name__ == "__main__":
    main()
