"""MNIST data loading through the CleverHans helper."""

from __future__ import print_function

import inspect
import warnings
from typing import Any, Tuple

import numpy as np


MnistArrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
_CLEVERHANS_MNIST_DEPRECATION = (
    r"cleverhans\.utils_mnist is deprecrated.*"
)


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
    kwargs = {
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
    }
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=_CLEVERHANS_MNIST_DEPRECATION,
                category=UserWarning,
            )
            from cleverhans.utils_mnist import data_mnist

        signature = inspect.signature(data_mnist)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if rng is not None and ("rng" in signature.parameters or accepts_kwargs):
            kwargs["rng"] = rng

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=_CLEVERHANS_MNIST_DEPRECATION,
                category=UserWarning,
            )
            return data_mnist(**kwargs)
    except (ImportError, IOError, OSError):
        return _load_mnist_data_keras(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )


def _one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Return integer labels as one-hot float32 rows."""
    label_array = np.asarray(labels, dtype=np.int64)
    output = np.zeros((len(label_array), int(num_classes)), dtype=np.float32)
    output[np.arange(len(label_array)), label_array] = 1.0
    return output


def _images_to_nhwc(images: np.ndarray) -> np.ndarray:
    """Return MNIST uint8 images as normalized NHWC float32 data."""
    image_array = np.asarray(images, dtype=np.float32) / 255.0
    if image_array.ndim != 3:
        raise ValueError("MNIST images must have shape (N, 28, 28).")
    return image_array.reshape((image_array.shape[0], 28, 28, 1)).astype(np.float32)


def _load_mnist_data_keras(
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
) -> MnistArrays:
    """Load MNIST through Keras when the legacy CleverHans URL is unavailable."""
    from keras.datasets import mnist as keras_mnist

    (x_train, y_train), (x_test, y_test) = keras_mnist.load_data()
    return (
        _images_to_nhwc(x_train[train_start:train_end]),
        _one_hot(y_train[train_start:train_end]),
        _images_to_nhwc(x_test[test_start:test_end]),
        _one_hot(y_test[test_start:test_end]),
    )
