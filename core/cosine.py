"""
Core cosine similarity and tensor utilities.

Zero-dependency. Tensors are nested Python lists.
Supports 1D vectors and 2D matrices. Strings are treated as 1D char-code
sequences when passed directly.
"""
from math import sqrt


def as_tensor(data):
    """Convert input to a flat or nested list tensor.

    Accepts:
      - list / tuple of numbers -> 1D tensor
      - list of lists -> 2D tensor
      - str -> 1D tensor of Unicode code points
      - int / float -> 1D single-element tensor
    """
    if isinstance(data, str):
        return [ord(c) for c in data]
    if isinstance(data, (int, float)):
        return [float(data)]
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and isinstance(data[0], (list, tuple)):
            return [list(row) for row in data]
        return [float(x) for x in data]
    raise TypeError(f"Unsupported input type: {type(data)}")


def flatten(tensor):
    """Flatten a nested list tensor into a 1D list."""
    result = []
    for item in tensor:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(float(item))
    return result


def cosine(a, b):
    """Cosine similarity between two 1D tensors (lists of numbers).

    Returns 0.0 if either vector is zero-length or zero-norm.
    """
    a = as_tensor(a)
    b = as_tensor(b)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def cosine_2d(a, b):
    """Cosine similarity between two 2D tensors (matrices), flattened."""
    return cosine(flatten(a), flatten(b))


def normalize(vec):
    """L2-normalize a 1D vector. Returns zero vector if norm is zero."""
    vec = as_tensor(vec)
    norm = sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return [0.0] * len(vec)
    return [x / norm for x in vec]


def dot(a, b):
    """Dot product of two 1D vectors (truncated to shorter length)."""
    n = min(len(a), len(b))
    return sum(float(a[i]) * float(b[i]) for i in range(n))


def tensor_shape(tensor):
    """Return shape of a nested-list tensor as a tuple."""
    if not isinstance(tensor, (list, tuple)):
        return ()
    if len(tensor) == 0:
        return (0,)
    if isinstance(tensor[0], (list, tuple)):
        inner = tensor_shape(tensor[0])
        return (len(tensor),) + inner
    return (len(tensor),)


def zeros(shape):
    """Create a zero tensor with given shape (tuple of ints)."""
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [zeros(shape[1:]) for _ in range(shape[0])]
