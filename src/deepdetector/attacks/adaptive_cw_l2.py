"""Defense-aware adaptive CW-L2 attack wrapper."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from deepdetector.attacks.nn_robust import (
    _model_session,
    _one_hot,
    _predict_with_model,
    _session_graph_context,
)
from deepdetector.attacks.tf_compat import patch_tensorflow_v1_symbols


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


def _cw_compare_scores(
    scores: np.ndarray,
    label: int,
    *,
    targeted: bool,
    confidence: float,
) -> bool:
    adjusted = np.array(scores, dtype=np.float32, copy=True)
    if targeted:
        adjusted[int(label)] -= float(confidence)
        return int(np.argmax(adjusted)) == int(label)
    adjusted[int(label)] += float(confidence)
    return int(np.argmax(adjusted)) != int(label)


def _cw_compare_label(
    candidate_label: int,
    true_label: int,
    *,
    targeted: bool,
) -> bool:
    if targeted:
        return int(candidate_label) == int(true_label)
    return int(candidate_label) != int(true_label)


def _as_batch(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image, dtype=np.float32)
    return image_array.reshape((1,) + tuple(image_array.shape))


def _empty_diagnostic_record(
    *,
    batch_index: int,
    true_label: int,
    context: Any = None,
) -> Dict[str, Any]:
    context_dict = dict(context or {})
    record: Dict[str, Any] = {
        "batch_index": int(batch_index),
        "true_label": int(true_label),
        "total_candidates": 0,
        "adversarial_candidates": 0,
        "detected_adversarial_candidates": 0,
        "evading_adversarial_candidates": 0,
        "best_adversarial_l2": None,
        "best_adversarial_squared_l2": None,
        "best_detected_adversarial_l2": None,
        "best_detected_adversarial_squared_l2": None,
        "best_defense_aware_l2": None,
        "best_defense_aware_squared_l2": None,
    }
    for key in ("sample_index", "valid_index", "clean_pred"):
        if key in context_dict:
            record[key] = int(context_dict[key])
    return record


def _record_best_l2(record: Dict[str, Any], prefix: str, squared_l2: float) -> None:
    squared_key = "best_{0}_squared_l2".format(prefix)
    current = record.get(squared_key)
    if current is None or float(squared_l2) < float(current):
        record[squared_key] = float(squared_l2)
        record["best_{0}_l2".format(prefix)] = float(np.sqrt(float(squared_l2)))


def _update_best_defense_aware_candidates(
    *,
    best_attacks: np.ndarray,
    best_l2: np.ndarray,
    clean_images: np.ndarray,
    labels: np.ndarray,
    candidates: np.ndarray,
    transform_fn: TransformFn,
    predict_fn: PredictFn,
    targeted: bool = False,
    confidence: float = 0.0,
    scores: Optional[np.ndarray] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Update best candidates using the defense-aware criterion.

    This helper is intentionally separate from the TensorFlow optimizer so tests
    can assert that intermediate candidates, not only final CW outputs, are
    accepted according to ``C(x) != y`` and ``C(x) == C(T(x))``.
    """
    clean_array = np.asarray(clean_images, dtype=np.float32)
    candidate_array = np.asarray(candidates, dtype=np.float32)
    label_ints = (
        np.asarray(labels).astype(np.int64)
        if np.asarray(labels).ndim == 1
        else np.argmax(np.asarray(labels), axis=1).astype(np.int64)
    ).reshape(-1)
    updated_attacks = np.asarray(best_attacks, dtype=np.float32).copy()
    updated_l2 = np.asarray(best_l2, dtype=np.float64).copy()
    score_array = None if scores is None else np.asarray(scores)

    for index, candidate in enumerate(candidate_array):
        diagnostic_record = diagnostics[index] if diagnostics is not None else None
        if diagnostic_record is not None:
            diagnostic_record["total_candidates"] += 1

        if score_array is None:
            candidate_pred = int(_predict_ints(predict_fn, _as_batch(candidate))[0])
            adversarial_success = _cw_compare_label(
                candidate_pred,
                int(label_ints[index]),
                targeted=bool(targeted),
            )
        else:
            score = score_array[index]
            candidate_pred = int(np.argmax(score))
            adversarial_success = _cw_compare_scores(
                score,
                int(label_ints[index]),
                targeted=bool(targeted),
                confidence=float(confidence),
            )

        if not adversarial_success:
            continue

        l2_distance = float(
            np.sum(
                np.square(
                    candidate.astype(np.float32) - clean_array[index].astype(np.float32)
                )
            )
        )
        if diagnostic_record is not None:
            diagnostic_record["adversarial_candidates"] += 1
            _record_best_l2(diagnostic_record, "adversarial", l2_distance)

        transformed = np.asarray(transform_fn(candidate), dtype=np.float32).reshape(
            candidate.shape
        )
        transformed_pred = int(_predict_ints(predict_fn, _as_batch(transformed))[0])
        if candidate_pred != transformed_pred:
            if diagnostic_record is not None:
                diagnostic_record["detected_adversarial_candidates"] += 1
                _record_best_l2(diagnostic_record, "detected_adversarial", l2_distance)
            continue

        if diagnostic_record is not None:
            diagnostic_record["evading_adversarial_candidates"] += 1
            _record_best_l2(diagnostic_record, "defense_aware", l2_distance)

        if l2_distance < float(updated_l2[index]):
            updated_l2[index] = l2_distance
            updated_attacks[index] = candidate

    return updated_attacks.astype(np.float32), updated_l2


def generate_native_adaptive_cw_l2_attack(
    model: Any,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    transform_fn: Optional[TransformFn] = None,
    detector: Any = None,
    predict_fn: Optional[PredictFn] = None,
    confidence: float = 0.0,
    batch_size: int = 1,
    max_iterations: int = 2000,
    learning_rate: float = 0.1,
    binary_search_steps: int = 5,
    initial_const: float = 1.0,
    abort_early: bool = True,
    targeted: bool = False,
    input_range: Any = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    diagnostic_context: Any = None,
    **_: Any,
) -> np.ndarray:
    """Generate defense-aware CW-L2 examples with native candidate tracking.

    This function does not run CW-L2 once and post-filter the final result. It
    evaluates every optimizer candidate and keeps the lowest-L2 image satisfying
    both adversarial success and detector evasion.
    """
    if transform_fn is None:
        transform_fn = getattr(detector, "transform", None)
    if transform_fn is None:
        raise ValueError("native adaptive CW-L2 requires transform_fn or detector.transform.")
    if predict_fn is None:
        predict_fn = getattr(model, "predict_label", None)
    if predict_fn is None:
        raise ValueError("native adaptive CW-L2 requires predict_fn or model.predict_label.")

    image_array = np.asarray(images, dtype=np.float32)
    if image_array.ndim != 4:
        raise ValueError("native adaptive CW-L2 requires image batches.")
    if int(batch_size) != len(image_array):
        # The experiment invokes this attack one sample at a time. Keeping the
        # native implementation batch-exact avoids padding candidates that would
        # never be evaluated.
        batch_size = len(image_array)

    input_min, input_max = _bounds(input_range, clip_min, clip_max)
    input_span = input_max - input_min
    if input_span <= 0.0:
        raise ValueError("native adaptive CW-L2 requires a valid input_range.")

    clipped_images = np.clip(image_array, input_min, input_max).astype(np.float32)
    label_array = np.asarray(labels)
    num_labels = int(getattr(model, "num_labels", 10))
    one_hot_labels = _one_hot(label_array, num_labels)
    label_ints = np.argmax(one_hot_labels, axis=1).astype(np.int64)
    diagnostic_records = [
        _empty_diagnostic_record(
            batch_index=index,
            true_label=int(label_ints[index]),
            context=diagnostic_context,
        )
        for index in range(len(clipped_images))
    ]
    if diagnostics is not None:
        diagnostics.extend(diagnostic_records)

    patch_tensorflow_v1_symbols()
    import tensorflow as tf

    compat_v1 = getattr(getattr(tf, "compat", None), "v1", tf)
    session = _model_session(model)
    shape = tuple(int(value) for value in clipped_images.shape)
    lower_bound = np.zeros(len(clipped_images), dtype=np.float32)
    const = np.ones(len(clipped_images), dtype=np.float32) * float(initial_const)
    upper_bound = np.ones(len(clipped_images), dtype=np.float32) * 1e10
    repeat = int(binary_search_steps) >= 10
    best_l2 = np.ones(len(clipped_images), dtype=np.float64) * np.inf
    best_attacks = clipped_images.copy()

    scaled = (clipped_images - input_min) / input_span
    tanh_images = np.arctanh((scaled * 2.0 - 1.0) * 0.999999).astype(np.float32)

    with _session_graph_context(session):
        start_vars = set(variable.name for variable in compat_v1.global_variables())
        modifier = tf.Variable(
            np.zeros(shape, dtype=np.float32),
            name="native_adaptive_cw_modifier",
        )
        timg = tf.Variable(
            np.zeros(shape, dtype=np.float32),
            name="native_adaptive_cw_timg",
        )
        tlab = tf.Variable(
            np.zeros((len(clipped_images), num_labels), dtype=np.float32),
            name="native_adaptive_cw_tlab",
        )
        tradeoff_const = tf.Variable(
            np.zeros(len(clipped_images), dtype=np.float32),
            name="native_adaptive_cw_const",
        )

        assign_timg = compat_v1.placeholder(tf.float32, shape)
        assign_tlab = compat_v1.placeholder(
            tf.float32,
            (len(clipped_images), num_labels),
        )
        assign_const = compat_v1.placeholder(tf.float32, [len(clipped_images)])
        setup = [
            timg.assign(assign_timg),
            tlab.assign(assign_tlab),
            tradeoff_const.assign(assign_const),
        ]

        newimg = (tf.tanh(modifier + timg) + 1.0) / 2.0 * input_span + input_min
        original = (tf.tanh(timg) + 1.0) / 2.0 * input_span + input_min
        output = _predict_with_model(model, newimg)
        l2dist = tf.reduce_sum(tf.square(newimg - original), axis=[1, 2, 3])
        real = tf.reduce_sum(tlab * output, axis=1)
        other = tf.reduce_max((1.0 - tlab) * output - tlab * 10000.0, axis=1)
        if bool(targeted):
            loss1 = tf.maximum(0.0, other - real + float(confidence))
        else:
            loss1 = tf.maximum(0.0, real - other + float(confidence))
        loss = tf.reduce_sum(tradeoff_const * loss1) + tf.reduce_sum(l2dist)

        optimizer = compat_v1.train.AdamOptimizer(float(learning_rate))
        train = optimizer.minimize(loss, var_list=[modifier])
        end_vars = compat_v1.global_variables()
        new_vars = [variable for variable in end_vars if variable.name not in start_vars]
        init = compat_v1.variables_initializer(var_list=new_vars)

        check_every = max(int(max_iterations) // 10, 1)
        for outer_step in range(int(binary_search_steps)):
            session.run(init)
            if repeat and outer_step == int(binary_search_steps) - 1:
                const = upper_bound.astype(np.float32)
            session.run(
                setup,
                {
                    assign_timg: tanh_images,
                    assign_tlab: one_hot_labels,
                    assign_const: const,
                },
            )

            best_step_l2 = np.ones(len(clipped_images), dtype=np.float64) * np.inf
            best_step_score = np.ones(len(clipped_images), dtype=np.int64) * -1
            previous_loss = 1e6

            for iteration in range(int(max_iterations)):
                _, current_loss, l2s, scores, candidates = session.run(
                    [train, loss, l2dist, output, newimg]
                )

                if bool(abort_early) and iteration % check_every == 0:
                    if float(current_loss) > previous_loss * 0.9999:
                        break
                    previous_loss = float(current_loss)

                for index, (l2_value, score) in enumerate(zip(l2s, scores)):
                    if l2_value < best_step_l2[index] and _cw_compare_scores(
                        score,
                        int(label_ints[index]),
                        targeted=bool(targeted),
                        confidence=float(confidence),
                    ):
                        best_step_l2[index] = float(l2_value)
                        best_step_score[index] = int(np.argmax(score))

                best_attacks, best_l2 = _update_best_defense_aware_candidates(
                    best_attacks=best_attacks,
                    best_l2=best_l2,
                    clean_images=clipped_images,
                    labels=label_ints,
                    candidates=np.clip(candidates, input_min, input_max),
                    transform_fn=transform_fn,
                    predict_fn=predict_fn,
                    targeted=bool(targeted),
                    confidence=float(confidence),
                    scores=scores,
                    diagnostics=diagnostic_records if diagnostics is not None else None,
                )

            for index in range(len(clipped_images)):
                if best_step_score[index] != -1 and _cw_compare_label(
                    int(best_step_score[index]),
                    int(label_ints[index]),
                    targeted=bool(targeted),
                ):
                    upper_bound[index] = min(upper_bound[index], const[index])
                    if upper_bound[index] < 1e9:
                        const[index] = (lower_bound[index] + upper_bound[index]) / 2.0
                else:
                    lower_bound[index] = max(lower_bound[index], const[index])
                    if upper_bound[index] < 1e9:
                        const[index] = (lower_bound[index] + upper_bound[index]) / 2.0
                    else:
                        const[index] *= 10.0

    return np.clip(best_attacks, input_min, input_max).astype(np.float32)
