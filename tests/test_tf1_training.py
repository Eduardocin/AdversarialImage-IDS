from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.training.tf1_training import (  # noqa: E402
    classification_loss,
    evaluate_clean_accuracy,
)


class FakeSession:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.feed_sizes = []

    def run(self, predictions, feed_dict):
        del predictions
        batch = next(iter(feed_dict.values()))
        self.feed_sizes.append(len(batch))
        return self.outputs.pop(0)


def test_evaluate_clean_accuracy_batches_argmax_predictions() -> None:
    """Clean evaluation should match one-hot labels by argmax."""
    session = FakeSession(
        [
            np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32),
            np.asarray([[0.7, 0.3]], dtype=np.float32),
        ]
    )
    images = np.zeros((3, 2), dtype=np.float32)
    labels = np.asarray([[0, 1], [1, 0], [0, 1]], dtype=np.float32)

    accuracy = evaluate_clean_accuracy(
        sess=session,
        x="x",
        predictions="predictions",
        images=images,
        labels=labels,
        batch_size=2,
    )

    assert accuracy == 2.0 / 3.0
    assert session.feed_sizes == [2, 1]


def test_classification_loss_uses_logits_cross_entropy() -> None:
    """M2 training should use TensorFlow logits cross-entropy."""
    calls = []
    fake_tf = SimpleNamespace(
        nn=SimpleNamespace(
            softmax_cross_entropy_with_logits=lambda labels, logits: calls.append(
                (labels, logits)
            )
            or np.asarray([2.0, 4.0], dtype=np.float32)
        ),
        reduce_mean=lambda values: float(np.mean(values)),
    )

    loss = classification_loss(
        fake_tf,
        "labels",
        "logits",
        output_is_probabilities=False,
    )

    assert calls == [("labels", "logits")]
    assert loss == 3.0


def test_classification_loss_uses_probability_cross_entropy() -> None:
    """M3 training should handle softmax probability outputs."""
    fake_tf = SimpleNamespace(
        clip_by_value=lambda values, min_value, max_value: np.clip(
            values,
            min_value,
            max_value,
        ),
        math=SimpleNamespace(log=np.log),
        reduce_sum=lambda values, axis: np.sum(values, axis=axis),
        reduce_mean=lambda values: float(np.mean(values)),
    )
    labels = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    probabilities = np.asarray([[0.5, 0.5], [0.1, 0.9]], dtype=np.float32)

    loss = classification_loss(
        fake_tf,
        labels,
        probabilities,
        output_is_probabilities=True,
    )

    assert np.isclose(loss, -float(np.mean([np.log(0.5), np.log(0.9)])))
