import importlib.util
import unittest

from src.detector.deepdetector import MnistDetectorCounts


class DetectorCountsTests(unittest.TestCase):
    def test_precision_and_recall_are_zero_safe(self):
        empty = MnistDetectorCounts(
            evaluated=0,
            original_wrong=0,
            attack_failed=0,
            tp=0,
            fp=0,
            fn=0,
            ttp=0,
        )
        non_empty = MnistDetectorCounts(
            evaluated=3,
            original_wrong=1,
            attack_failed=2,
            tp=2,
            fp=1,
            fn=1,
            ttp=1,
        )

        self.assertEqual(empty.precision, 0.0)
        self.assertEqual(empty.recall, 0.0)
        self.assertAlmostEqual(non_empty.precision, 2 / 3)
        self.assertAlmostEqual(non_empty.recall, 2 / 3)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class PredictionChangeDetectorTests(unittest.TestCase):
    def test_uniform_transform_preserves_batch_shape_and_range(self):
        import torch

        from src.detector.deepdetector import transform_mnist_batch

        images = torch.tensor(
            [
                [[[0.0] * 28 for _ in range(28)]],
                [[[1.0] * 28 for _ in range(28)]],
            ],
            dtype=torch.float32,
        )

        transformed = transform_mnist_batch(images, "uniform", interval=128)

        self.assertEqual(tuple(transformed.shape), (2, 1, 28, 28))
        self.assertTrue(torch.all(transformed >= 0.0))
        self.assertTrue(torch.all(transformed <= 1.0))

    def test_box_transform_preserves_batch_shape_and_range(self):
        import torch

        from src.detector.config import MnistDetectorConfig
        from src.detector.deepdetector import transform_mnist_batch

        images = torch.rand((2, 1, 28, 28), dtype=torch.float32)

        transformed = transform_mnist_batch(
            images,
            MnistDetectorConfig("box", kernel_size=3),
        )

        self.assertEqual(tuple(transformed.shape), (2, 1, 28, 28))
        self.assertTrue(torch.all(transformed >= 0.0))
        self.assertTrue(torch.all(transformed <= 1.0))

    def test_detect_prediction_change_flags_changed_prediction(self):
        import torch
        from torch import nn

        from src.detector.deepdetector import detect_prediction_change

        class ThresholdModel(nn.Module):
            def forward(self, inputs):
                score = inputs.mean(dim=(1, 2, 3))
                return torch.stack([1.0 - score, score], dim=1)

        model = ThresholdModel()
        images = torch.full((1, 1, 28, 28), 0.51)

        result = detect_prediction_change(model, images, "uniform", interval=128)

        self.assertEqual(result.original_predictions.tolist(), [1])
        self.assertEqual(result.transformed_predictions.tolist(), [0])
        self.assertEqual(result.is_adversarial.tolist(), [True])


if __name__ == "__main__":
    unittest.main()
