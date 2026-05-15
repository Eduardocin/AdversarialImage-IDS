from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.filters.quantization import (
    find_border,
    nonuniform_quantization,
    normalize_image_range,
    scalar_quantization,
)


class QuantizationNumpyTests(unittest.TestCase):
    def test_normalize_image_range_clips_to_unit_interval(self):
        image = np.array([-1.0, 0.5, 2.0], dtype=np.float32)

        normalized = normalize_image_range(image)

        np.testing.assert_allclose(normalized, [0.0, 0.5, 1.0])
        self.assertEqual(normalized.dtype, np.float32)

    def test_scalar_quantization_preserves_shape_and_range_for_image_shapes(self):
        image_2d = np.linspace(0.0, 1.0, 28 * 28, dtype=np.float32).reshape(28, 28)
        image_3d = image_2d.reshape(28, 28, 1)
        batch = np.stack([image_3d, image_3d], axis=0)

        for image in (image_2d, image_3d, batch):
            quantized = scalar_quantization(image, interval=128)
            self.assertEqual(quantized.shape, image.shape)
            self.assertGreaterEqual(float(quantized.min()), 0.0)
            self.assertLessEqual(float(quantized.max()), 1.0)

    def test_scalar_quantization_reduces_unique_values(self):
        image = np.linspace(0.0, 1.0, 28 * 28, dtype=np.float32).reshape(28, 28)

        quantized = scalar_quantization(image, interval=128)

        self.assertLess(len(np.unique(quantized)), len(np.unique(image)))

    def test_scalar_quantization_matches_legacy_left_and_centered_bins(self):
        image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

        left = scalar_quantization(image, interval=128, left=True)
        centered = scalar_quantization(image, interval=128, left=False)

        np.testing.assert_allclose(left, [[0.0, 0.0, 128.0 / 255.0]])
        np.testing.assert_allclose(centered, [[64.0 / 255.0, 64.0 / 255.0, 192.0 / 255.0]])

    def test_find_border_uses_legacy_half_pixel_histogram_threshold(self):
        image = np.zeros((28, 28), dtype=np.float32)
        image[14:, :] = 1.0

        border = find_border(image)

        self.assertEqual(border, 255)

    def test_nonuniform_quantization_preserves_shape_and_range(self):
        image_2d = np.zeros((28, 28), dtype=np.float32)
        image_2d[14:, :] = 1.0
        image_3d = image_2d.reshape(28, 28, 1)
        batch = np.stack([image_3d, 1.0 - image_3d], axis=0)

        for image in (image_2d, image_3d, batch):
            quantized = nonuniform_quantization(image)
            self.assertEqual(quantized.shape, image.shape)
            self.assertGreaterEqual(float(quantized.min()), 0.0)
            self.assertLessEqual(float(quantized.max()), 1.0)

    def test_nonuniform_quantization_matches_legacy_half_black_half_white_case(self):
        image = np.zeros((28, 28), dtype=np.float32)
        image[14:, :] = 1.0

        quantized = nonuniform_quantization(image)

        np.testing.assert_allclose(quantized, np.zeros_like(image))


if __name__ == "__main__":
    unittest.main()
