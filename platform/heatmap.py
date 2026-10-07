"""
platform.heatmap -- fixed-range PNG recording (NumPy + standard library).

Change / binary fields are written with an explicit value range so uniform
fields and near-zero residuals are not distorted by auto-scaling.
"""
from __future__ import annotations
import os, struct, zlib

DEPENDENCIES = {
    "python": ">=3.8",
    "requires": {"numpy": ">=1.20"},
    "optional": {},
    "backends": ["numpy"],
    "status": "platform",
    "targets": "fixed-range forensic heatmaps",
}


def save_png(path, grid, lo: float = 0.0, hi: float = 1.0, upscale: int = 1):
    """Write a grayscale PNG. Values map linearly from [lo,hi] to [0,255]."""
    import numpy as np
    a = np.asarray(grid, dtype=float)
    rng = (hi - lo) or 1.0
    b = np.clip((a - lo) / rng, 0.0, 1.0)
    u = np.kron(b, np.ones((upscale, upscale)))
    h, w = u.shape
    raw = b"".join(b"\x00" + (row * 255).astype("uint8").tobytes()
                   for row in u)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xffffffff)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" +
                chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
