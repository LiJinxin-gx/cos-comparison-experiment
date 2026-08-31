"""
Example: Passive-Active Joint Extraction.

Demonstrates the core principle: information arises from difference.
Passive mode finds boundaries (where data changes); active mode matches
templates; joint mode uses passive points to guide active matching.
"""
import sys
sys.path.insert(0, r"C:\AI项目\explore")
from core import passive_extract, active_match, joint_extract, local_variance


def main():
    print("=== Passive Mode: Boundary Extraction ===")
    # A signal with clear transitions (boundaries)
    signal = [0, 0, 0, 1, 1, 1, 0, 0, 0.5, 0.5, 1, 1, 0, 0]
    variances = local_variance(signal, window=3)
    print(f"Signal:    {signal}")
    print(f"Variance:  {[f'{v:.2f}' for v in variances]}")

    points = passive_extract(signal, window=3, threshold=0.3)
    print(f"Boundary points (index, variance):")
    for idx, var in points:
        print(f"  position {idx}: variance={var:.3f} (value={signal[idx]})")

    print("\n=== Active Mode: Template Matching ===")
    templates = [
        ("rising_edge", [0, 0, 1, 1]),
        ("falling_edge", [1, 1, 0, 0]),
        ("plateau", [1, 1, 1, 1]),
    ]
    matches = active_match(signal, templates)
    for name, pos, sim in matches:
        print(f"  {name}: best at position {pos}, similarity={sim:.4f}")

    print("\n=== Joint Mode: Passive-Guided Active ===")
    joint = joint_extract(signal, templates, passive_window=3, passive_threshold=0.3)
    print(f"  Passive points found: {joint['n_passive']}")
    print(f"  Active matches near boundaries: {joint['n_active']}")
    for name, pos, sim in joint["active_matches"]:
        print(f"    {name} at pos {pos}: sim={sim:.4f}")

    print("\n=== String Input ===")
    text = "aabbbcccd"
    text_points = passive_extract([ord(c) for c in text], window=2, threshold=0.2)
    print(f"Text: '{text}'")
    print(f"Transition points: {[(text[i], v) for i, v in text_points]}")


if __name__ == "__main__":
    main()
