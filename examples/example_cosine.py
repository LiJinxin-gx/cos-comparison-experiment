"""
Example: Basic cosine similarity and tensor utilities.

Zero dependencies. Inputs are lists (tensors) or strings.
"""
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import cosine, as_tensor, flatten, multiscale_cosine, multiscale_match


def main():
    print("=== 1. Basic Cosine Similarity ===")
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.0, 2.0, 3.0, 4.0]
    c = [4.0, 3.0, 2.0, 1.0]
    print(f"identical vectors: cos({a}, {b}) = {cosine(a, b):.4f}")
    print(f"opposite vectors:  cos({a}, {c}) = {cosine(a, c):.4f}")

    print("\n=== 2. String Input (auto-converted to code-point tensor) ===")
    s1 = "hello"
    s2 = "hello"
    s3 = "world"
    print(f"cos('{s1}', '{s2}') = {cosine(s1, s2):.4f}")
    print(f"cos('{s1}', '{s3}') = {cosine(s1, s3):.4f}")

    print("\n=== 3. 2D Tensor (matrix) ===")
    m1 = [[1, 2], [3, 4]]
    m2 = [[1, 2], [3, 4]]
    from core import cosine_2d
    print(f"cosine_2d(identical matrices) = {cosine_2d(m1, m2):.4f}")
    print(f"flatten(m1) = {flatten(m1)}")

    print("\n=== 4. Multi-scale Cosine Matching ===")
    # Simulate a "signal" with different scales
    signal = [float(i % 10) for i in range(100)]
    noisy = [x + 0.1 for x in signal]  # slightly perturbed
    different = [float((i * 3) % 7) for i in range(100)]

    print(f"same signal multi-scale:    {multiscale_cosine(signal, noisy):.4f}")
    print(f"different signal multi-scale: {multiscale_cosine(signal, different):.4f}")

    print("\n=== 5. Database Matching ===")
    database = [
        ("class_A", [1.0, 0.0, 1.0, 0.0, 1.0]),
        ("class_B", [0.0, 1.0, 0.0, 1.0, 0.0]),
        ("class_C", [1.0, 1.0, 0.0, 0.0, 1.0]),
    ]
    query = [0.9, 0.1, 0.9, 0.1, 0.9]
    results = multiscale_match(query, database, top_k=2)
    print(f"Query: {query}")
    for label, sim in results:
        print(f"  {label}: {sim:.4f}")


if __name__ == "__main__":
    main()
