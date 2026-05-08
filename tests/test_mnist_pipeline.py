import unittest
from unittest.mock import patch

import numpy as np

from src.datasets.mnist import MNIST_SPLIT_SPECS, MnistNpzDataset, load_mnist_npz
from src.utils.metrics import classification_counts, precision_recall
from src.utils.seed import set_seed


class FakeNpz:
    def __init__(self, arrays):
        self.arrays = arrays

    def __enter__(self):
        return self.arrays

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class MnistPipelineTests(unittest.TestCase):
    def test_load_mnist_npz_validates_split_shapes(self):
        fake_npz = FakeNpz(
            {
                "x_train": np.zeros((4500, 28, 28), dtype=np.uint8),
                "y_train": np.arange(4500, dtype=np.int64) % 10,
                "x_validation": np.zeros((1000, 28, 28), dtype=np.uint8),
                "y_validation": np.arange(1000, dtype=np.int64) % 10,
                "x_test": np.zeros((4500, 28, 28), dtype=np.uint8),
                "y_test": np.arange(4500, dtype=np.int64) % 10,
            }
        )

        with patch("src.datasets.mnist.np.load", return_value=fake_npz):
            arrays = load_mnist_npz("mnist_splits.npz", "train")
            dataset = MnistNpzDataset("mnist_splits.npz", "train")
        image, label = dataset[0]

        self.assertEqual(arrays.images.shape, (4500, 28, 28))
        self.assertEqual(arrays.labels[:3].tolist(), [0, 1, 2])
        self.assertEqual(arrays.spec.start_index, 0)
        self.assertEqual(arrays.spec.end_index, 4499)
        self.assertEqual(len(dataset), 4500)
        self.assertEqual(dataset.original_index(0), 0)
        self.assertEqual(dataset.original_index(4499), 4499)
        self.assertEqual(tuple(image.shape), (1, 28, 28))
        self.assertEqual(label, 0)

    def test_mnist_split_specs_match_table_2_ranges(self):
        self.assertEqual(MNIST_SPLIT_SPECS["train"].start_index, 0)
        self.assertEqual(MNIST_SPLIT_SPECS["train"].end_index, 4499)
        self.assertEqual(MNIST_SPLIT_SPECS["train"].expected_count, 4500)
        self.assertEqual(MNIST_SPLIT_SPECS["validation"].start_index, 4500)
        self.assertEqual(MNIST_SPLIT_SPECS["validation"].end_index, 5499)
        self.assertEqual(MNIST_SPLIT_SPECS["validation"].expected_count, 1000)
        self.assertEqual(MNIST_SPLIT_SPECS["test"].start_index, 5500)
        self.assertEqual(MNIST_SPLIT_SPECS["test"].end_index, 9999)
        self.assertEqual(MNIST_SPLIT_SPECS["test"].expected_count, 4500)

    def test_load_mnist_npz_rejects_wrong_split_count_by_default(self):
        fake_npz = FakeNpz(
            {
                "x_train": np.zeros((2, 28, 28), dtype=np.uint8),
                "y_train": np.array([1, 2], dtype=np.int64),
            }
        )

        with patch("src.datasets.mnist.np.load", return_value=fake_npz):
            with self.assertRaisesRegex(ValueError, "expected 4500"):
                load_mnist_npz("mnist_splits.npz", "train")

    def test_metrics_use_binary_detection_semantics(self):
        counts = classification_counts(
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 1, 0]),
        )
        metrics = precision_recall(counts)

        self.assertEqual(counts.tp, 1)
        self.assertEqual(counts.fp, 1)
        self.assertEqual(counts.fn, 1)
        self.assertEqual(counts.tn, 1)
        self.assertEqual(metrics, {"precision": 0.5, "recall": 0.5})

    def test_set_seed_reproducibly_seeds_numpy(self):
        set_seed(42)
        first = np.random.rand(3)
        set_seed(42)
        second = np.random.rand(3)

        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
