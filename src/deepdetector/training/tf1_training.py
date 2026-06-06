"""Small TF1-style training helpers that avoid CleverHans runtime gates."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def classification_loss(
    tf_module: Any,
    labels: Any,
    predictions: Any,
    *,
    output_is_probabilities: bool = False,
) -> Any:
    """Return mean cross-entropy for logits or probability outputs."""
    if output_is_probabilities:
        clipped = tf_module.clip_by_value(predictions, 1e-7, 1.0)
        per_sample = -tf_module.reduce_sum(labels * tf_module.math.log(clipped), axis=1)
    else:
        per_sample = tf_module.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=predictions,
        )
    return tf_module.reduce_mean(per_sample)


def evaluate_clean_accuracy(
    *,
    sess: Any,
    x: Any,
    predictions: Any,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    feed: Optional[Dict[Any, Any]] = None,
) -> float:
    """Evaluate clean classification accuracy over numpy arrays."""
    image_array = np.asarray(images, dtype=np.float32)
    label_array = np.asarray(labels)
    if label_array.ndim == 2:
        expected = np.argmax(label_array, axis=1)
    else:
        expected = label_array.astype(np.int64)

    predicted: list[np.ndarray] = []
    for start in range(0, len(image_array), int(batch_size)):
        end = min(start + int(batch_size), len(image_array))
        feed_dict: Dict[Any, Any] = {x: image_array[start:end]}
        if feed:
            feed_dict.update(feed)
        batch_output = np.asarray(sess.run(predictions, feed_dict=feed_dict))
        if batch_output.ndim > 1:
            batch_output = np.argmax(batch_output, axis=1)
        predicted.append(batch_output.astype(np.int64).reshape(-1))

    if not predicted:
        return 0.0
    predicted_labels = np.concatenate(predicted, axis=0)
    return float(np.mean(predicted_labels == expected[: len(predicted_labels)]))
