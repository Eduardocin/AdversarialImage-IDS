import numpy as np

from src.mnist.noise_reduction import (
    adaptive_reduce,
    choose_closer_filter,
    cross_mean_filter,
    one_d_entropy,
    scalar_quantization,
)


def test_entropy_of_constant_image_is_zero() -> None:
    image = np.zeros((28, 28), dtype=np.float32)

    assert one_d_entropy(image) == 0.0


def test_scalar_quantization_uses_reference_bucket_rule() -> None:
    image = np.array([[0.0, 128.0 / 255.0, 1.0]], dtype=np.float32)

    quantized = scalar_quantization(image, 128)

    assert np.allclose(quantized, np.array([[0.0, 128.0 / 255.0, 128.0 / 255.0]], dtype=np.float32))


def test_choose_closer_filter_selects_per_pixel_values() -> None:
    original = np.array([[0.2, 0.8]], dtype=np.float32)
    first = np.array([[0.1, 0.1]], dtype=np.float32)
    second = np.array([[0.9, 0.7]], dtype=np.float32)

    chosen = choose_closer_filter(original, first, second)

    assert np.allclose(chosen, np.array([[0.1, 0.7]], dtype=np.float32))


def test_cross_mean_filter_preserves_shape() -> None:
    image = np.ones((28, 28), dtype=np.float32)

    filtered = cross_mean_filter(image, start=3, end=25, coefficient=13)

    assert filtered.shape == image.shape
    assert np.allclose(filtered[3:25, 3:25], 1.0)


def test_adaptive_reduce_returns_same_shape() -> None:
    image = np.zeros((28, 28), dtype=np.float32)

    reduced = adaptive_reduce(image)

    assert reduced.shape == image.shape
