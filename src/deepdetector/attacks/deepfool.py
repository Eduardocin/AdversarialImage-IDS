"""DeepFool attack generation for differentiable model wrappers."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def _as_image_batch(images: np.ndarray) -> np.ndarray:
    image_array = np.asarray(images, dtype=np.float32)
    if image_array.ndim == 3:
        image_array = image_array.reshape((1,) + image_array.shape)
    if image_array.ndim < 2:
        raise ValueError("images must contain one image or a batch of images.")
    return image_array


def _scores(model: Any, image: np.ndarray) -> np.ndarray:
    batch = image.reshape((1,) + image.shape)
    if hasattr(model, "predict_preprocessed_batch"):
        values = model.predict_preprocessed_batch(batch)
    elif hasattr(model, "predict_batch"):
        values = model.predict_batch(batch)
    elif hasattr(model, "scores"):
        values = model.scores(batch)
    elif callable(model):
        values = model(batch)
    else:
        raise NotImplementedError("DeepFool requires model scores or logits.")

    score_array = np.asarray(values, dtype=np.float32)
    if score_array.ndim == 1:
        return score_array
    if score_array.ndim == 2 and len(score_array) == 1:
        return score_array[0]
    raise ValueError("model scores must have shape (num_classes,) or (1, num_classes).")


def _gradient(model: Any, image: np.ndarray, class_id: int) -> np.ndarray:
    if not hasattr(model, "gradient"):
        raise NotImplementedError("DeepFool requires gradient access on the model.")

    gradient = np.asarray(model.gradient(image, int(class_id)), dtype=np.float32)
    if gradient.shape != image.shape:
        raise ValueError("Gradient shape does not match image shape.")
    return gradient


def _class_candidates(scores: np.ndarray, num_classes: Optional[int]) -> np.ndarray:
    ordered = np.argsort(scores)[::-1]
    if num_classes is None:
        return ordered
    return ordered[: max(1, min(int(num_classes), len(ordered)))]


def _deepfool_one(
    *,
    model: Any,
    image: np.ndarray,
    max_iter: int,
    overshoot: float,
    clip_min: float,
    clip_max: float,
    num_classes: Optional[int],
) -> np.ndarray:
    clean = np.asarray(image, dtype=np.float32)
    clean_scores = _scores(model, clean)
    candidates = _class_candidates(clean_scores, num_classes)
    original_label = int(candidates[0])

    total_perturbation = np.zeros_like(clean, dtype=np.float32)
    adversarial = clean.copy()

    for _ in range(int(max_iter)):
        current_scores = _scores(model, adversarial)
        current_label = int(np.argmax(current_scores))
        if current_label != original_label:
            break

        original_gradient = _gradient(model, adversarial, original_label)
        best_distance = np.inf
        best_direction: Optional[np.ndarray] = None

        for class_id in candidates:
            candidate_label = int(class_id)
            if candidate_label == original_label:
                continue

            candidate_gradient = _gradient(model, adversarial, candidate_label)
            # ImageNet wrappers expose the same attack-gradient convention used
            # by FGSM. This difference recovers the score-gradient direction
            # needed by the DeepFool update without changing wrapper behavior.
            direction = original_gradient - candidate_gradient
            direction_norm = float(np.linalg.norm(direction.reshape(-1)))
            if direction_norm <= 0.0:
                continue

            score_gap = float(current_scores[candidate_label] - current_scores[original_label])
            distance = abs(score_gap) / direction_norm
            if distance < best_distance:
                best_distance = distance
                best_direction = direction

        if best_direction is None:
            break

        direction_norm = float(np.linalg.norm(best_direction.reshape(-1)))
        perturbation = (best_distance + 1e-4) * best_direction / direction_norm
        total_perturbation = total_perturbation + perturbation.astype(np.float32)
        adversarial = clean + (1.0 + float(overshoot)) * total_perturbation
        adversarial = np.clip(adversarial, clip_min, clip_max).astype(np.float32)

    return adversarial.astype(np.float32)


def generate_deepfool(
    model: Any,
    images: np.ndarray,
    labels: Optional[Sequence[Any]] = None,
    *,
    max_iter: int = 50,
    overshoot: float = 0.02,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Generate adversarial examples with DeepFool.

    The implementation follows the original DeepFool update rule while using
    the repository's differentiable wrapper contract: class scores plus
    ``gradient(image, class_id)``. Labels are accepted for dispatcher
    compatibility; the attack uses the model's clean prediction as the class to
    fool, matching the Table 10 failure definition.
    """
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative.")
    if clip_min >= clip_max:
        raise ValueError("clip_min must be smaller than clip_max.")
    if labels is not None and len(labels) != len(_as_image_batch(images)):
        raise ValueError("labels must have the same length as images.")
    if not hasattr(model, "gradient"):
        raise NotImplementedError("DeepFool requires gradient access on the model.")

    image_batch = _as_image_batch(images)
    adversarial = [
        _deepfool_one(
            model=model,
            image=image,
            max_iter=max_iter,
            overshoot=overshoot,
            clip_min=clip_min,
            clip_max=clip_max,
            num_classes=num_classes,
        )
        for image in image_batch
    ]
    return np.asarray(adversarial, dtype=np.float32)
