"""Entropy helpers for grayscale image filters."""

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
