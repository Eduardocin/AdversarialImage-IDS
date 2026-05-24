"""Entropy helpers for normalized image filters."""

from __future__ import print_function

import numpy as np


def one_d_entropy(input_digit: np.ndarray) -> float:
    """Compute Shannon entropy of pixel intensities.

    H=0 para imagem constante; H≈8 para distribuição uniforme de 256 tons.
    """
    pixels = (input_digit.flatten() * 255).astype(np.uint8)
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255))
    if hist.sum() == 0:
        return 0.0

    p = hist.astype(np.float64) / hist.sum()
    p_nonzero = p[p > 0]
    H = -np.sum(p_nonzero * np.log2(p_nonzero))
    return float(H)


def image_entropy_255_chw(image: np.ndarray) -> float:
    """Compute mean per-channel Shannon entropy for CxHxW 0-255 image data."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 3:
        raise ValueError("image must have shape (C, H, W).")

    entropies = []
    clipped = np.clip(image_array, 0.0, 255.0).astype(np.uint8)
    for channel in clipped:
        counts = np.bincount(channel.ravel(), minlength=256).astype(np.float64)
        total = float(counts.sum())
        if total == 0.0:
            entropies.append(0.0)
            continue

        probabilities = counts[counts > 0.0] / total
        entropies.append(float(-np.sum(probabilities * np.log2(probabilities))))

    return float(np.mean(entropies)) if entropies else 0.0
