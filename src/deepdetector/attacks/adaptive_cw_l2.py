"""Defense-aware adaptive CW-L2 attack wrapper."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np

from deepdetector.attacks.nn_robust import (
    _load_nn_robust_carlini_l2_adaptive,
    _model_session,
    _one_hot,
    _require_root,
    _session_graph_context,
    generate_nn_robust_cw_l2_attack,
)


PredictFn = Callable[[np.ndarray], np.ndarray]
TransformFn = Callable[[np.ndarray], np.ndarray]


def _bounds(config: Any, default_min: float, default_max: float) -> Tuple[float, float]:
    if not isinstance(config, dict):
        return float(default_min), float(default_max)
    return float(config.get("min", default_min)), float(config.get("max", default_max))


def _predict_ints(predict_fn: PredictFn, images: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predict_fn(np.asarray(images, dtype=np.float32)))
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    return predictions.astype(np.int64).reshape(-1)


class _OriginalCarliniKerasPredictProxy(object):
    """Expose .predict(np_array) returning NumPy logits for CarliniL2Adaptive.

    The original `l2_adaptive_attack.py` calls:

        self.tempmodel.model.predict(imgTobePre)

    inside `mnistPredicate`. That call happens during the Python-side search and
    must return a NumPy array, not a symbolic Tensor.
    """

    def __init__(
        self,
        *,
        sess: Any,
        input_tensor: Any,
        logits_tensor: Any,
        input_shift: float = 0.5,
    ) -> None:
        self.sess = sess
        self.input_tensor = input_tensor
        self.logits_tensor = logits_tensor
        self.input_shift = float(input_shift)

    def predict(self, data: Any) -> np.ndarray:
        data_array = np.asarray(data, dtype=np.float32)
        return self.sess.run(
            self.logits_tensor,
            feed_dict={self.input_tensor: data_array + self.input_shift},
        )


class OriginalCarliniAdaptiveModelAdapter(object):
    """Adapter required by the original CarliniL2Adaptive implementation.

    The original attack expects two different prediction interfaces:

    1. `adapter.predict(symbolic_tensor)` during graph construction. This must
       return symbolic logits.
    2. `adapter.model.predict(np_array)` during `mnistPredicate`. This must
       return NumPy logits.

    The attack operates on centered images in [-0.5, 0.5], while the MNIST M2
    model in this project receives images in [0, 1]. Therefore both prediction
    paths add `input_shift=0.5` before calling M2.
    """

    image_size = 28
    num_channels = 1
    num_labels = 10

    def __init__(
        self,
        *,
        keras_model: Any,
        sess: Any,
        input_tensor: Any,
        logits_tensor: Any,
        input_shift: float = 0.5,
        image_size: int = 28,
        num_channels: int = 1,
        num_labels: int = 10,
    ) -> None:
        self.keras_model = keras_model
        self.sess = sess
        self.input_tensor = input_tensor
        self.logits_tensor = logits_tensor
        self.input_shift = float(input_shift)
        self.image_size = int(image_size)
        self.num_channels = int(num_channels)
        self.num_labels = int(num_labels)

        # Required by original l2_adaptive_attack.py:
        # self.tempmodel.model.predict(...) -> NumPy logits.
        self.model = _OriginalCarliniKerasPredictProxy(
            sess=sess,
            input_tensor=input_tensor,
            logits_tensor=logits_tensor,
            input_shift=input_shift,
        )

    def predict(self, centered_data: Any) -> Any:
        # Required by original l2_adaptive_attack.py:
        # model.predict(self.newimg) -> symbolic logits.
        return self.keras_model(centered_data + self.input_shift)


def generate_adaptive_cw_l2_attack(
    model: Any,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    transform_fn: Optional[TransformFn] = None,
    detector: Any = None,
    predict_fn: Optional[PredictFn] = None,
    base_attack_fn: Callable[..., np.ndarray] = generate_nn_robust_cw_l2_attack,
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
    label_ints = (
        label_array.astype(np.int64)
        if label_array.ndim == 1
        else np.argmax(label_array, axis=1)
    )

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


def generate_original_adaptive_cw_l2_attack(
    model: Any,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    nn_robust_attacks_root: Optional[str] = None,
    confidence: float = 0.0,
    batch_size: int = 1,
    max_iterations: int = 2000,
    learning_rate: float = 0.1,
    binary_search_steps: int = 5,
    initial_const: float = 1.0,
    abort_early: bool = True,
    targeted: bool = False,
    input_range: Any = None,
    attack_box: Any = None,
    model_input_shift: float = 0.5,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    **_: Any,
) -> np.ndarray:
    """Generate adaptive CW-L2 examples using the original CarliniL2Adaptive.

    Project tensors use [0, 1]. The original attack uses centered tensors in
    [-0.5, 0.5]. This function converts both ways and uses an adapter whose
    `.model.predict(...)` returns NumPy logits for the original attack's
    internal `mnistPredicate`.
    """
    image_array = np.asarray(images, dtype=np.float32)
    if image_array.ndim != 4:
        raise ValueError("original adaptive CW-L2 requires image batches.")

    input_min, input_max = _bounds(input_range, clip_min, clip_max)
    box_min, box_max = _bounds(attack_box, -0.5, 0.5)
    input_span = input_max - input_min
    box_span = box_max - box_min
    if input_span <= 0.0 or box_span <= 0.0:
        raise ValueError("original adaptive CW-L2 requires valid input_range and attack_box.")

    centered_images = (image_array - input_min) / input_span * box_span + box_min

    session = _model_session(model)
    input_tensor = getattr(model, "input_tensor", None)
    logits_tensor = getattr(model, "logits_tensor", None)
    if logits_tensor is None:
        logits_tensor = getattr(model, "predictions", None)

    if input_tensor is None:
        raise ValueError("M2 model must expose .input_tensor for original_adaptive_cw_l2.")
    if logits_tensor is None:
        raise ValueError(
            "M2 model must expose .logits_tensor or .predictions for original_adaptive_cw_l2."
        )

    one_hot_labels = _one_hot(np.asarray(labels), 10)
    CarliniL2Adaptive = _load_nn_robust_carlini_l2_adaptive(
        _require_root(nn_robust_attacks_root)
    )

    adapter = OriginalCarliniAdaptiveModelAdapter(
        keras_model=model,
        sess=session,
        input_tensor=input_tensor,
        logits_tensor=logits_tensor,
        input_shift=float(model_input_shift),
        image_size=int(centered_images.shape[1]),
        num_channels=int(centered_images.shape[3]),
        num_labels=10,
    )

    with _session_graph_context(session):
        attack = CarliniL2Adaptive(
            session,
            adapter,
            batch_size=int(batch_size),
            confidence=float(confidence),
            targeted=bool(targeted),
            learning_rate=float(learning_rate),
            binary_search_steps=int(binary_search_steps),
            max_iterations=int(max_iterations),
            abort_early=bool(abort_early),
            initial_const=float(initial_const),
            boxmin=float(box_min),
            boxmax=float(box_max),
        )
        centered_adversarial = attack.attack(centered_images, one_hot_labels)

    adversarial = (
        (np.asarray(centered_adversarial, dtype=np.float32) - box_min)
        / box_span
        * input_span
        + input_min
    )
    if adversarial.shape != image_array.shape:
        raise ValueError("original adaptive CW-L2 returned an unexpected shape.")
    return np.clip(adversarial, clip_min, clip_max).astype(np.float32)
