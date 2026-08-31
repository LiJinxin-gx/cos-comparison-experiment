"""
Example: Martian Encoding — language-agnostic internal representation.

Demonstrates:
  - Symbol anonymization: structure preserved, identity forgotten
  - n-gram level anonymization: cross-lingual alignment
  - Structural feature encoding: language-agnostic similarity
  - Normalized Compression Distance: completely encoding-agnostic
"""
import sys
sys.path.insert(0, r"C:\AI项目\explore")
from algorithms import (martian_encode, martian_similarity,
                        anonymize_symbols, ngram_anonymize,
                        normalized_compression_distance)


def main():
    print("=== Symbol Anonymization ===")
    # Different symbols, same structure → same encoding
    seq1 = ["我", "你", "我", "他", "你"]
    seq2 = ["A", "B", "A", "C", "B"]
    print(f"  {seq1} → {anonymize_symbols(seq1)}")
    print(f"  {seq2} → {anonymize_symbols(seq2)}")
    print(f"  (identical structure → identical anonymized sequence)")

    print("\n=== n-gram Anonymization (cross-lingual) ===")
    zh = "你好世界"
    en = "hello world"
    print(f"  Chinese '{zh}' → {ngram_anonymize(zh, n=2)}")
    print(f"  English '{en}' → {ngram_anonymize(en, n=2)}")

    print("\n=== Martian Encoding (structural feature vector) ===")
    enc1 = martian_encode("The quick brown fox jumps over the lazy dog")
    enc2 = martian_encode("A fast auburn fox leaps above a sleepy canine")
    enc3 = martian_encode("完全不同的中文句子内容和结构")
    print(f"  English sentence 1: {[f'{v:.3f}' for v in enc1]}")
    print(f"  English sentence 2: {[f'{v:.3f}' for v in enc2]}")
    print(f"  Chinese sentence:   {[f'{v:.3f}' for v in enc3]}")

    print("\n=== Cross-Lingual Similarity ===")
    sim_en = martian_similarity(
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps above a sleepy canine")
    sim_cross = martian_similarity(
        "The quick brown fox jumps over the lazy dog",
        "敏捷的棕色狐狸跳过懒狗")
    sim_diff = martian_similarity(
        "The quick brown fox jumps over the lazy dog",
        "Stock markets rose sharply today on positive earnings")
    print(f"  Same meaning (EN-EN):     {sim_en:.4f}")
    print(f"  Same meaning (EN-ZH):     {sim_cross:.4f}")
    print(f"  Different meaning:        {sim_diff:.4f}")

    print("\n=== Normalized Compression Distance ===")
    ncd_same = normalized_compression_distance(
        ngram_anonymize("hello hello hello"),
        ngram_anonymize("hello hello hello"))
    ncd_diff = normalized_compression_distance(
        ngram_anonymize("hello hello hello"),
        ngram_anonymize("world world world"))
    print(f"  NCD identical: {ncd_same:.4f} (lower = more similar)")
    print(f"  NCD different: {ncd_diff:.4f}")


if __name__ == "__main__":
    main()
