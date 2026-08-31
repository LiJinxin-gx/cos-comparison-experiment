"""
Martian Encoding — language-agnostic internal representation.

Core philosophy: "Understanding the world precedes describing it." Humans
understand the world before they have language; language is a mapping onto
pre-existing understanding. The model should NOT store human language — it
should convert all input into a structure-based internal encoding. Same/similar
content in different languages should produce similar internal encodings,
enabling natural cross-lingual capability.

Core principle: STRUCTURE FIRST. Martian encoding is just the internal
encoding; training uses structure-first as the core criterion. No tokenization,
no dictionary lookup, no character recognition.

Evolution:
  v1: statistical features (frequency, entropy, n-gram diversity) — FAIL
      (all natural languages have similar statistics, distinction 0.0005)
  v2: byte-level symbol anonymization — PARTIAL (distinction 0.0088)
      re-number symbols by first-occurrence order; "我 你 我 他 你" and
      "A B A C B" both become [0,1,0,2,1]
  v3: n-gram level anonymization + normalized compression distance — BEST
      (distinction 0.0118, 23x improvement over v1)
      Sentence-level cross-lingual matching still at random level — needs
      multimodal grounding (image/audio first, then text mapping).
"""
from core.cosine import cosine


def anonymize_symbols(sequence):
    """Symbol anonymization: re-number by first-occurrence order.

    Doesn't care what the symbols ARE — only their structural pattern.
    "我 你 我 他 你" → [0, 1, 0, 2, 1]
    "A B A C B"     → [0, 1, 0, 2, 1]

    Works on any hashable sequence (characters, words, bytes, tokens).
    """
    mapping = {}
    result = []
    counter = 0
    for sym in sequence:
        if sym not in mapping:
            mapping[sym] = counter
            counter += 1
        result.append(mapping[sym])
    return result


def ngram_anonymize(text, n=3):
    """n-gram level anonymization for cross-lingual alignment.

    Chinese: 3 bytes/char, English: 1 byte/letter. Byte-level anonymization
    doesn't align. n-gram level converts both to same-granularity blocks.

    Returns anonymized n-gram sequence.
    """
    if isinstance(text, str):
        # Use characters as base units (handles CJK and Latin uniformly)
        units = list(text)
    else:
        units = list(text)
    # Generate n-grams
    ngrams = []
    for i in range(len(units) - n + 1):
        ngrams.append(tuple(units[i:i + n]))
    return anonymize_symbols(ngrams)


def normalized_compression_distance(a, b):
    """Normalized Compression Distance (NCD) — completely language-agnostic.

    NCD(x,y) = (Z(xy) - min(Z(x), Z(y))) / max(Z(x), Z(y))
    where Z = compressed size (approximated by run-length encoding here).

    Lower NCD = more similar. Works on any byte sequence regardless of
    language or encoding.
    """
    def rle_size(seq):
        """Approximate compressed size via run-length encoding."""
        if not seq:
            return 0
        size = 0
        count = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                count += 1
            else:
                size += 1 + (1 if count > 1 else 0)
                count = 1
        size += 1 + (1 if count > 1 else 0)
        return size

    a_anon = anonymize_symbols(a) if not isinstance(a, list) or (a and not isinstance(a[0], int)) else a
    b_anon = anonymize_symbols(b) if not isinstance(b, list) or (b and not isinstance(b[0], int)) else b

    za = rle_size(a_anon)
    zb = rle_size(b_anon)
    zab = rle_size(a_anon + b_anon)
    if max(za, zb) == 0:
        return 0.0
    return (zab - min(za, zb)) / max(za, zb)


def martian_encode(text, n=3):
    """Full Martian encoding: n-gram anonymization + structural features.

    Returns a feature vector (list of floats) suitable for cosine matching.
    """
    anon = ngram_anonymize(text, n)
    if not anon:
        return [0.0] * 8
    # Structural features (language-agnostic)
    n_unique = len(set(anon))
    n_total = len(anon)
    # frequency distribution shape
    freq = {}
    for x in anon:
        freq[x] = freq.get(x, 0) + 1
    freqs = sorted(freq.values(), reverse=True)
    # normalized frequency of top symbols
    top_freq = freqs[0] / n_total if freqs else 0.0
    # entropy approximation
    import math
    entropy = 0.0
    for f in freqs:
        p = f / n_total
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(n_unique) if n_unique > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    # repetition ratio
    repetition = 1.0 - (n_unique / n_total) if n_total > 0 else 0.0
    # first-occurrence spread
    first_occ = {}
    for i, x in enumerate(anon):
        if x not in first_occ:
            first_occ[x] = i / n_total
    avg_first = sum(first_occ.values()) / len(first_occ) if first_occ else 0.0

    return [
        float(n_total) / 100.0,          # normalized length
        float(n_unique) / max(n_total, 1),  # vocabulary ratio
        top_freq,                        # top frequency
        norm_entropy,                    # normalized entropy
        repetition,                      # repetition ratio
        avg_first,                       # average first occurrence position
        float(len(freqs)) / 10.0,        # normalized unique count
        float(sum(freqs[:3])) / n_total if n_total > 0 else 0.0,  # top-3 concentration
    ]


def martian_similarity(text_a, text_b, n=3):
    """Similarity between two texts using Martian encoding.

    Combines cosine similarity of structural features with NCD.
    """
    enc_a = martian_encode(text_a, n)
    enc_b = martian_encode(text_b, n)
    cos_sim = cosine(enc_a, enc_b)
    # NCD: lower is more similar, convert to similarity
    ncd = normalized_compression_distance(
        ngram_anonymize(text_a, n), ngram_anonymize(text_b, n))
    ncd_sim = max(0.0, 1.0 - ncd)
    return 0.5 * cos_sim + 0.5 * ncd_sim
