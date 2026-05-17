"""Adversarial attack evaluation helpers."""

from __future__ import print_function

from typing import Any, Dict

import numpy as np


def evaluate_attack_success(
    sess: Any,
    x: Any,
    predictions: Any,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    Y_true: np.ndarray,
) -> Dict[str, float]:
    """Evaluate clean/adversarial predictions using CleverHans argmax."""
    from cleverhans.utils_tf import model_argmax

    def learning_phase_feed() -> Dict[Any, Any]:
        """Return a feed dict for Keras learning phase when needed."""
        try:
            from keras import backend as K
        except Exception:
            return {}
        if not hasattr(K, "learning_phase"):
            return {}
        phase = K.learning_phase()
        if hasattr(phase, "op"):
            return {phase: 0}
        return {}

    if X_clean.shape != X_adv.shape:
        raise ValueError("X_clean and X_adv must have the same shape.")
    if len(X_clean) != len(Y_true):
        raise ValueError("X_clean and Y_true must contain the same number of examples.")
    if Y_true.ndim != 2:
        raise ValueError("Y_true must be one-hot encoded.")

    true_labels = np.argmax(Y_true, axis=1)
    feed = learning_phase_feed()
    clean_predictions = model_argmax(sess, x, predictions, X_clean, feed=feed)
    adv_predictions = model_argmax(sess, x, predictions, X_adv, feed=feed)

    clean_correct = clean_predictions == true_labels
    adv_correct = adv_predictions == true_labels
    original_classified_wrong_number = int(np.sum(~clean_correct))
    disturbed_failure_number = int(np.sum(clean_correct & adv_correct))
    successful_attacks = int(np.sum(clean_correct & ~adv_correct))
    valid_attack_candidates = int(len(X_clean) - original_classified_wrong_number)

    if valid_attack_candidates:
        attack_success_rate = successful_attacks / float(valid_attack_candidates)
    else:
        attack_success_rate = 0.0

    return {
        "total_examples": int(len(X_clean)),
        "clean_accuracy": float(np.mean(clean_correct)),
        "adversarial_accuracy": float(np.mean(adv_correct)),
        "original_classified_wrong_number": original_classified_wrong_number,
        "disturbed_failure_number": disturbed_failure_number,
        "successful_attacks": successful_attacks,
        "valid_attack_candidates": valid_attack_candidates,
        "attack_success_rate": float(attack_success_rate),
    }
