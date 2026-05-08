import math
import unittest

import numpy as np

from src.detector.features import one_d_entropy_mnist, one_d_entropy_uint8_chw
from src.detector.transformations import (
    box_mean_filter_chw,
    choose_closer_filter,
    cross_mean_filter_chw,
    diamond_mean_filter_chw,
    find_nonuniform_border_mnist,
    nonuniform_quantization_mnist,
    normalize_values,
    scalar_quantization_mnist,
    scalar_quantization_uint8,
)


class DetectorTransformationTests(unittest.TestCase):
    def test_scalar_quantization_mnist_preserves_legacy_scale(self):
        image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

        quantized = scalar_quantization_mnist(image, 128)
        centered = scalar_quantization_mnist(image, 128, left=False)

        np.testing.assert_allclose(quantized, [[0.0, 0.0, 128.0 / 255.0]])
        np.testing.assert_allclose(
            centered,
            [[64.0 / 255.0, 64.0 / 255.0, 192.0 / 255.0]],
        )

    def test_scalar_quantization_uint8_preserves_legacy_scale(self):
        image = np.array([[0.0, 42.0, 127.0, 128.0, 255.0]], dtype=np.float32)

        quantized = scalar_quantization_uint8(image, 128)
        centered = scalar_quantization_uint8(image, 128, left=False)

        np.testing.assert_allclose(quantized, [[0.0, 0.0, 0.0, 128.0, 128.0]])
        np.testing.assert_allclose(centered, [[64.0, 64.0, 64.0, 192.0, 192.0]])

    def test_entropy_mnist_constant_and_binary_images(self):
        constant = np.zeros((28, 28), dtype=np.float32)
        binary = np.zeros((28, 28), dtype=np.float32)
        binary[:14, :] = 1.0

        self.assertEqual(one_d_entropy_mnist(constant), 0.0)
        self.assertTrue(math.isclose(one_d_entropy_mnist(binary), 1.0))

    def test_entropy_chw_averages_channel_entropies(self):
        image = np.zeros((3, 2, 2), dtype=np.float32)
        image[1, :1, :] = 255.0

        self.assertTrue(math.isclose(one_d_entropy_uint8_chw(image), 1.0 / 3.0))

    def test_nonuniform_quantization_uses_legacy_histogram_border(self):
        image = np.zeros((28, 28), dtype=np.float32)
        image[14:, :] = 1.0

        border = find_nonuniform_border_mnist(image)
        quantized = nonuniform_quantization_mnist(image)

        self.assertEqual(border, 255)
        np.testing.assert_allclose(quantized, np.zeros_like(image))

    def test_mean_filters_update_interior_and_keep_edges(self):
        image = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        diamond_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

        box = box_mean_filter_chw(image, start=1, end=2, coefficient=9)
        diamond = diamond_mean_filter_chw(
            image,
            diamond_kernel,
            start=1,
            end=2,
            coefficient=5,
        )
        cross = cross_mean_filter_chw(image, start=1, end=2, coefficient=5)

        np.testing.assert_allclose(box[0, 1, 1], 4.0)
        np.testing.assert_allclose(diamond[0, 1, 1], 4.0)
        np.testing.assert_allclose(cross[0, 1, 1], 4.0)
        np.testing.assert_allclose(box[0, 0, :], image[0, 0, :])

    def test_choose_closer_filter_matches_legacy_tie_behavior(self):
        original = np.array([0.0, 10.0, 20.0])
        first = np.array([1.0, 9.0, 18.0])
        second = np.array([5.0, 11.0, 22.0])

        chosen = choose_closer_filter(original, first, second)

        np.testing.assert_allclose(chosen, [1.0, 11.0, 22.0])

    def test_normalize_values_can_clip_copy_or_in_place(self):
        image = np.array([-1.0, 0.5, 2.0], dtype=np.float32)

        clipped = normalize_values(image, 0.0, 1.0)
        returned = normalize_values(image, 0.0, 1.0, inplace=True)

        np.testing.assert_allclose(clipped, [0.0, 0.5, 1.0])
        self.assertIs(returned, image)
        np.testing.assert_allclose(image, [0.0, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
