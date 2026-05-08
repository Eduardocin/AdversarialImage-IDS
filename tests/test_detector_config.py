import unittest

from src.detector.config import (
    MNIST_FILTER_KERNEL_SIZES,
    MNIST_QUANTIZATION_INTERVALS,
    MnistDetectorConfig,
    default_mnist_detector_candidates,
)
from src.detector.deepdetector import MnistDetectorCounts
from src.detector.selection import (
    MnistDetectorCandidateResult,
    select_best_mnist_detector,
)


class DetectorConfigTests(unittest.TestCase):
    def test_default_mnist_candidates_match_original_groups(self):
        candidates = default_mnist_detector_candidates()

        uniform = [candidate for candidate in candidates if candidate.transform == "uniform"]
        nonuniform = [candidate for candidate in candidates if candidate.transform == "nonuniform"]
        box = [candidate for candidate in candidates if candidate.transform == "box"]
        diamond = [candidate for candidate in candidates if candidate.transform == "diamond"]
        cross = [candidate for candidate in candidates if candidate.transform == "cross"]

        self.assertEqual([candidate.interval for candidate in uniform], MNIST_QUANTIZATION_INTERVALS)
        self.assertEqual(len(nonuniform), 1)
        self.assertEqual([candidate.kernel_size for candidate in box], MNIST_FILTER_KERNEL_SIZES)
        self.assertEqual([candidate.kernel_size for candidate in diamond], MNIST_FILTER_KERNEL_SIZES)
        self.assertEqual([candidate.kernel_size for candidate in cross], MNIST_FILTER_KERNEL_SIZES)

    def test_detector_config_round_trips_dict_payload(self):
        config = MnistDetectorConfig("uniform", interval=128, left=False)

        restored = MnistDetectorConfig.from_dict(config.to_dict())

        self.assertEqual(restored, config)

    def test_select_best_uses_f1_then_precision(self):
        weaker = MnistDetectorCandidateResult(
            config=MnistDetectorConfig("uniform", interval=128),
            counts=MnistDetectorCounts(
                evaluated=10,
                original_wrong=0,
                attack_failed=0,
                tp=4,
                fp=4,
                fn=2,
                ttp=3,
            ),
        )
        stronger = MnistDetectorCandidateResult(
            config=MnistDetectorConfig("box", kernel_size=3),
            counts=MnistDetectorCounts(
                evaluated=10,
                original_wrong=0,
                attack_failed=0,
                tp=6,
                fp=1,
                fn=2,
                ttp=5,
            ),
        )

        self.assertEqual(select_best_mnist_detector([weaker, stronger]), stronger)


if __name__ == "__main__":
    unittest.main()
