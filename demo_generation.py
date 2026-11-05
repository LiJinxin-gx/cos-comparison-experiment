# -*- coding: utf-8 -*-
"""
Generation Demo: Structural Inverse Propagation.

Demonstrates the generation side of continuous mapping group:
  Forward: L0 --down--> L1 --down--> L2  (coarsening, feature extraction)
  Backward: L2 --up--> L1 --up--> L0  (structural inverse, reconstruction)

Core principles:
  - Structural inverse: shape preserved, not exact inverse (info loss)
  - Primitive isolation: active and passive channels are independent
  - Level isolation: each level has its own templates
  - Absolute locality: all operations are local

Zero dependencies: pure Python standard library only.
The mapping primitives run on the installed cos_comparison engine
(core.pool, core.cos_comparison_passive); the structural up-sampling
helper remains local. All operations stay local over plain tensors.
"""
import random

from cos_comparison import core


def down_sample(tensor, factor):
    """Average pooling via the formal engine: coarsen by factor."""
    h, w = len(tensor), len(tensor[0])
    nh, nw = h // factor, w // factor
    output = core.create_void_list((nh, nw))
    core.pool(tensor, window_size=(factor, factor),
              step=(factor, factor), output=output)
    return output


def up_sample(tensor, target_h, target_w):
    """Nearest-neighbor up-sample (structural inverse, not exact)."""
    h, w = len(tensor), len(tensor[0])
    return [[tensor[min(i*h//target_h, h-1)][min(j*w//target_w, w-1)]
             for j in range(target_w)] for i in range(target_h)]


def passive_extract(tensor):
    """Passive d-shift boundary map via the formal engine (gap space)."""
    h, w = len(tensor), len(tensor[0])
    output = core.create_void_list((h - 1, w))
    core.cos_comparison_passive(tensor, window_size=(1, 1),
                                d=(1, 0), output=output)
    return output


def generate_from_level(low_level_tensor, target_h, target_w, factor):
    """
    Structural inverse: generate higher-resolution from lower-resolution.
    
    This is NOT an exact inverse - information was lost during down-sampling.
    We reconstruct shape, but fine details are lost.
    
    Args:
        low_level_tensor: coarse tensor from lower level
        target_h: target height (original level)
        target_w: target width (original level)
        factor: down-sampling factor that was used
    Returns:
        Reconstructed tensor (same shape as original, but approximate)
    """
    return up_sample(low_level_tensor, target_h, target_w)


def generation_demo():
    """Demo: encode then decode (autoencoder-like, but structural)."""
    print("=" * 60)
    print("Generation Demo: Structural Inverse Propagation")
    print("=" * 60)
    
    # Create a test pattern (like a small digit)
    size = 16
    original = [[0.0] * size for _ in range(size)]
    # Draw a simple "2" shape
    for i in range(3, 13):
        original[3][i] = 200.0  # top horizontal
    for i in range(3, 8):
        original[i][12] = 200.0  # top right vertical
    for i in range(3, 13):
        original[8][i] = 200.0  # middle horizontal
    for i in range(8, 13):
        original[i][3] = 200.0  # bottom left vertical
    for i in range(3, 13):
        original[12][i] = 200.0  # bottom horizontal
    
    print(f"\nOriginal shape: {size}x{size}")
    
    # Forward pass: encode to 3 levels
    levels_active = []
    current = original
    pool_factors = [1, 2, 4]
    
    print("\n--- Forward Pass (Encoding) ---")
    for i, factor in enumerate(pool_factors):
        if factor > 1:
            current = down_sample(current, factor)
        levels_active.append(current)
        print(f"Level L{i} (factor={factor}): {len(current)}x{len(current[0])}")
    
    # Backward pass: decode from highest level
    print("\n--- Backward Pass (Generation / Reconstruction) ---")
    reconstructed = levels_active[-1]  # Start from most compressed
    print(f"Start from L{len(pool_factors)-1}: {len(reconstructed)}x{len(reconstructed[0])}")
    
    for i in range(len(pool_factors) - 2, -1, -1):
        factor = pool_factors[i + 1]
        target_h = size // pool_factors[i]
        target_w = size // pool_factors[i]
        reconstructed = up_sample(reconstructed, target_h, target_w)
        print(f"Up-sample to L{i}: {len(reconstructed)}x{len(reconstructed[0])}")
    
    # Compute reconstruction error
    print("\n--- Reconstruction Quality ---")
    total_error = 0.0
    max_error = 0.0
    for i in range(size):
        for j in range(size):
            err = abs(original[i][j] - reconstructed[i][j])
            total_error += err
            max_error = max(max_error, err)
    
    avg_error = total_error / (size * size)
    print(f"Average pixel error: {avg_error:.2f}")
    print(f"Max pixel error: {max_error:.2f}")
    print(f"Normalized error: {avg_error / 255 * 100:.1f}%")
    
    # Show side-by-side (ASCII art)
    print("\n--- Original vs Reconstructed (ASCII) ---")
    print("Original:          Reconstructed:")
    for i in range(0, size, 2):
        orig_row = ""
        recon_row = ""
        for j in range(0, size, 2):
            orig_row += "#" if original[i][j] > 128 else "."
            recon_row += "#" if reconstructed[i][j] > 128 else "."
        print(f"{orig_row:<18} {recon_row}")
    
    # Passive mode generation: what about gap space?
    print("\n--- Passive Mode Generation ---")
    print("Note: Passive features (gap space) are also stored per-level.")
    print("We can reconstruct boundaries independently of raw values.")
    
    # The formal passive boundary is a cosmod map in [0,1]: low at a
    # boundary and high in a uniform region. Up-sample to the original size
    # for a like-for-like display, and mark a boundary where the value is low.
    original_passive = up_sample(passive_extract(original), size, size)
    passive_down = down_sample(passive_extract(original), 2)
    passive_recon = up_sample(passive_down, size, size)

    print("Original boundary map:")
    for i in range(0, size, 2):
        row = "".join("#" if original_passive[i][j] < 0.5 else "."
                      for j in range(0, size, 2))
        print(f"  {row}")

    print("\nReconstructed boundary map:")
    for i in range(0, size, 2):
        row = "".join("#" if passive_recon[i][j] < 0.5 else "."
                      for j in range(0, size, 2))
        print(f"  {row}")
    
    print("\n✅ Generation demo complete!")
    print("Key insight: structural inverse preserves SHAPE, not exact values.")
    print("This is the foundation for template-based generation.")


if __name__ == '__main__':
    generation_demo()
