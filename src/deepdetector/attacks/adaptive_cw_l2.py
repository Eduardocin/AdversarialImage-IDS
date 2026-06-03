"""Defense-aware adaptive CW-L2 attack wrapper."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from deepdetector.attacks.cw_l2 import generate_cw_l2_attack


PredictFn = Callable[[np.ndarray], np.ndarray]
TransformFn = Callable[[np.ndarray], np.ndarray]


def _predict_ints(predict_fn: PredictFn, images: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predict_fn(np.asarray(images, dtype=np.float32)))
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    return predictions.astype(np.int64).reshape(-1)


def generate_adaptive_cw_l2_attack(
    model: Any,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    transform_fn: TransformFn | None = None,
    detector: Any = None,
    predict_fn: PredictFn | None = None,
    base_attack_fn: Callable[..., np.ndarray] = generate_cw_l2_attack,
    **kwargs: Any,
) -> np.ndarray:
    """Generate CW-L2 candidates and keep only defense-aware successes.

    A returned candidate must satisfy both required conditions:
    ``C(x_adv) != y`` and ``C(x_adv) == C(T(x_adv))``. Candidates that fail
    either condition are rejected by returning the original clean image at that
    position, allowing the evaluator to count them as attack failures.
    """
    if transform_fn is None:
        transform_fn = getattr(detector, "transform", None)
    if transform_fn is None:
        raise ValueError("adaptive CW-L2 requires transform_fn or detector.transform.")
    if predict_fn is None:
        predict_fn = getattr(model, "predict_label", None)
    if predict_fn is None:
        raise ValueError("adaptive CW-L2 requires predict_fn or model.predict_label.")

    image_array = np.asarray(images, dtype=np.float32)
    label_array = np.asarray(labels)
    label_ints = label_array.astype(np.int64) if label_array.ndim == 1 else np.argmax(label_array, axis=1)

    adversarial = np.asarray(
        base_attack_fn(
            model=model,
            images=image_array,
            labels=label_array,
            **kwargs,
        ),
        dtype=np.float32,
    )
    if adversarial.shape != image_array.shape:
        raise ValueError("adaptive CW-L2 base attack returned an unexpected shape.")

    accepted = image_array.copy()
    for index, candidate in enumerate(adversarial):
        candidate_batch = candidate.reshape((1,) + tuple(candidate.shape))
        candidate_pred = int(_predict_ints(predict_fn, candidate_batch)[0])
        transformed = np.asarray(transform_fn(candidate), dtype=np.float32).reshape(
            candidate.shape
        )
        transformed_pred = int(
            _predict_ints(
                predict_fn,
                transformed.reshape((1,) + tuple(transformed.shape)),
            )[0]
        )
        if candidate_pred != int(label_ints[index]) and candidate_pred == transformed_pred:
            accepted[index] = candidate

    return accepted.astype(np.float32)
