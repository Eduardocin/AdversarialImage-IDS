"""CW Linf attack interface for the MNIST M2 reproduction path."""

from __future__ import print_function

from typing import Any

import numpy as np


class CwLinfUnavailableError(NotImplementedError):
    """Raised when the installed CleverHans stack has no CW Linf attack."""


def generate_cw_linf_examples(
    sess: Any,
    model: Any,
    x_placeholder: Any,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    max_iterations: int,
    learning_rate: float,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray:
    """Generate CW Linf examples when a compatible implementation exists.

    CleverHans 3.1.0 exposes CarliniWagnerL2, but not a stable CW Linf attack
    API compatible with this legacy TF1/Keras project. This function keeps the
    pipeline interface explicit and fails with a clear reason instead of
    inventing results.
    """
    del sess, model, x_placeholder, images, labels, batch_size
    del max_iterations, learning_rate, clip_min, clip_max
    raise CwLinfUnavailableError(
        "CW Linf is not implemented for the installed TF1/Keras/CleverHans stack."
    )
