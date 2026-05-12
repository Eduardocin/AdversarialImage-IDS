import numpy as np

from src.mnist.data import clip_m1_images, clip_m2_images, to_m1_range, to_m2_range


def test_m1_clip_range() -> None:
    values = np.array([-1.0, 0.25, 2.0], dtype=np.float32)

    clipped = clip_m1_images(values)

    assert clipped.tolist() == [0.0, 0.25, 1.0]


def test_m2_clip_range() -> None:
    values = np.array([-1.0, 0.25, 2.0], dtype=np.float32)

    clipped = clip_m2_images(values)

    assert clipped.tolist() == [-0.5, 0.25, 0.5]


def test_m1_m2_range_conversion_round_trip() -> None:
    image = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    converted = to_m2_range(image)

    assert converted.tolist() == [-0.5, 0.0, 0.5]
    assert np.allclose(to_m1_range(converted), image)

