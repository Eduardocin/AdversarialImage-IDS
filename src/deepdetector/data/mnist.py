"""MNIST data loading through the CleverHans helper."""

from __future__ import print_function

import inspect
from typing import Any, Tuple

import numpy as np


MnistArrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def load_mnist_data(
    train_start: int = 0,
    train_end: int = 60000,
    test_start: int = 0,
    test_end: int = 10000,
    rng: Any = None,
) -> MnistArrays:
    """Load MNIST as NHWC images and one-hot labels.

    The returned image arrays follow the CleverHans MNIST data convention:
    ``X_train`` has shape ``(n_train, 28, 28, 1)`` and ``X_test`` has shape
    ``(n_test, 28, 28, 1)`` with pixel values in ``[0, 1]``. Label arrays are
    one-hot encoded with 10 classes, so ``Y_train`` and ``Y_test`` have shapes
    ``(n_train, 10)`` and ``(n_test, 10)``.
    """
    from cleverhans.utils_mnist import data_mnist

    kwargs = {
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
    }
    signature = inspect.signature(data_mnist)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if rng is not None and ("rng" in signature.parameters or accepts_kwargs):
        kwargs["rng"] = rng

    return data_mnist(**kwargs)
