"""Top-k prediction-change evaluation for ambiguous ImageNet samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from deepdetector.attacks.fgsm_imagenet import is_caffe_input
from deepdetector.io.paths import ensure_dir
from deepdetector.io.result_writers import write_metrics_csv, write_metrics_json


SelectionSample = Tuple[np.ndarray, np.ndarray, float, float]
AttackGenerator = Callable[[Any, np.ndarray, int, Mapping[str, Any]], np.ndarray]
TransformFn = Callable[[np.ndarray], np.ndarray]

SELECTION_FIELDS = (
    "class",
    "total_loaded",
    "ambiguous_candidates",
    "selected_for_experiment",
    "mean_top1_confidence",
    "mean_top1_top2_margin",
)
METRIC_FIELDS = (
    "class",
    "k",
    "selected",
    "test_number",
    "disturbed_failure",
    "TP",
    "FN",
    "FP",
    "TN",
    "recall_percent",
    "precision_percent",
    "f1_percent",
    "false_positive_rate_percent",
    "false_positive_reduction_percent",
    "attack_success_rate_percent",
)


@dataclass(frozen=True)
class TopKDetectionResult:
    """Output rows and JSON payloads for one top-k experiment run."""

    selection_rows: list[dict[str, Any]]
    selection_json: dict[str, dict[str, Any]]
    metric_rows: list[dict[str, Any]]
    metrics_json: dict[str, dict[str, dict[str, Any]]]
    ambiguous_images_by_class: dict[str, list[np.ndarray]]


def softmax(scores: Sequence[float]) -> np.ndarray:
    """Return a stable softmax vector."""
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_array.size == 0:
        raise ValueError("scores must not be empty.")
    shifted = score_array - np.max(score_array)
    exp_scores = np.exp(shifted)
    return (exp_scores / np.sum(exp_scores)).astype(np.float64)


def probabilities_from_scores(scores: Sequence[float]) -> np.ndarray:
    """Return probabilities, preserving already-normalized model outputs."""
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_array.size == 0:
        raise ValueError("scores must not be empty.")
    if np.all(score_array >= 0.0) and np.isclose(np.sum(score_array), 1.0, atol=1e-5):
        return score_array
    return softmax(score_array)


def top_k_indices(probs: Sequence[float], k: int) -> list[int]:
    """Return indices of the k largest probabilities in descending order."""
    prob_array = np.asarray(probs, dtype=np.float64).reshape(-1)
    if prob_array.size == 0:
        raise ValueError("probs must not be empty.")
    if int(k) <= 0:
        raise ValueError("k must be positive.")
    if int(k) > prob_array.size:
        raise ValueError("k cannot exceed the number of classes.")
    ordered = np.argsort(-prob_array, kind="mergesort")
    return [int(index) for index in ordered[: int(k)]]


def is_detected_topk(
    probs_before: Sequence[float],
    probs_after: Sequence[float],
    k: int,
) -> bool:
    """Return whether top-k sets have no class intersection."""
    before = set(top_k_indices(probs_before, k))
    after = set(top_k_indices(probs_after, k))
    return len(before.intersection(after)) == 0


def _predict_scores(model: Any, image: np.ndarray) -> np.ndarray:
    """Return model scores for one image, including preprocessed Caffe tensors."""
    image_array = np.asarray(image, dtype=np.float32)
    batch = image_array.reshape((1,) + image_array.shape)
    if is_caffe_input(batch) and hasattr(model, "predict_preprocessed_batch"):
        return np.asarray(model.predict_preprocessed_batch(batch)[0], dtype=np.float64)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(image_array), dtype=np.float64)
    if hasattr(model, "predict_batch"):
        return np.asarray(model.predict_batch(batch)[0], dtype=np.float64)
    raise ValueError("model must expose predict, predict_batch, or predict_preprocessed_batch.")


def predict_proba(model: Any, image: np.ndarray) -> np.ndarray:
    """Return class probabilities for one image."""
    return probabilities_from_scores(_predict_scores(model, image))


def _class_top_stats(model: Any, image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return probabilities, top-1 confidence, and top1-top2 margin."""
    probs = predict_proba(model, image)
    top_classes = top_k_indices(probs, k=2)
    top1_prob = float(probs[top_classes[0]])
    top2_prob = float(probs[top_classes[1]])
    return probs, top1_prob, float(top1_prob - top2_prob)


def _select_ambiguous_samples(
    *,
    model: Any,
    samples: Sequence[np.ndarray],
    selection_config: Mapping[str, Any],
) -> tuple[list[SelectionSample], dict[str, Any]]:
    """Select ambiguous samples for one class."""
    max_margin = float(selection_config.get("max_margin", 0.15))
    min_confidence = float(selection_config.get("min_top1_confidence", 0.20))
    max_confidence = float(selection_config.get("max_top1_confidence", 0.60))
    max_samples = int(selection_config.get("max_samples_per_class", len(samples)))

    candidates: list[SelectionSample] = []
    for image in samples:
        probs, top1_prob, margin = _class_top_stats(model, image)
        if margin <= max_margin and min_confidence <= top1_prob <= max_confidence:
            candidates.append((np.asarray(image, dtype=np.float32), probs, top1_prob, margin))

    candidates.sort(key=lambda item: item[3])
    selected = candidates[:max_samples]
    summary = {
        "total_loaded": int(len(samples)),
        "ambiguous_candidates": int(len(candidates)),
        "selected_for_experiment": int(len(selected)),
        "mean_top1_confidence": _mean([item[2] for item in selected]),
        "mean_top1_top2_margin": _mean([item[3] for item in selected]),
    }
    return selected, summary


def _mean(values: Sequence[float]) -> float:
    """Return a plain float mean with zero for empty input."""
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _zero_counts(selected: int) -> dict[str, int]:
    """Return initialized counters for one k value."""
    return {
        "selected": int(selected),
        "test_number": 0,
        "disturbed_failure": 0,
        "TP": 0,
        "FN": 0,
        "FP": 0,
        "TN": 0,
    }


def _safe_rate(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or zero for an empty denominator."""
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _metric_payload(counts: Mapping[str, int], baseline_fp: int) -> dict[str, Any]:
    """Return counts plus percentage metrics for one class/k bucket."""
    tp = int(counts["TP"])
    fn = int(counts["FN"])
    fp = int(counts["FP"])
    tn = int(counts["TN"])
    selected = int(counts["selected"])
    test_number = int(counts["test_number"])

    recall = _safe_rate(tp, tp + fn)
    precision = _safe_rate(tp, tp + fp)
    f1 = _safe_rate(2.0 * recall * precision, recall + precision)
    false_positive_rate = _safe_rate(fp, fp + tn)
    false_positive_reduction = _safe_rate(baseline_fp - fp, baseline_fp)
    attack_success_rate = _safe_rate(test_number, selected)

    return {
        "selected": selected,
        "test_number": test_number,
        "disturbed_failure": int(counts["disturbed_failure"]),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "recall_percent": recall * 100.0,
        "precision_percent": precision * 100.0,
        "f1_percent": f1 * 100.0,
        "false_positive_rate_percent": false_positive_rate * 100.0,
        "false_positive_reduction_percent": false_positive_reduction * 100.0,
        "attack_success_rate_percent": attack_success_rate * 100.0,
    }


def _configured_k_values(config: Mapping[str, Any]) -> list[int]:
    """Return the configured baseline plus candidate k values."""
    baseline = int(config.get("baseline_k", 1))
    candidates = [int(value) for value in config.get("candidate_k_values", [2, 3, 5])]
    values: list[int] = []
    for value in [baseline] + candidates:
        if value not in values:
            values.append(value)
    return values


def evaluate_topk_detection(
    *,
    samples_by_class: Mapping[str, Sequence[np.ndarray]],
    model: Any,
    transform: TransformFn,
    attack_generator: AttackGenerator,
    config: Mapping[str, Any],
) -> TopKDetectionResult:
    """Evaluate top-k detection and return CSV/JSON-ready outputs."""
    selection_config = config.get("ambiguity_selection", {})
    detection_config = config.get("topk_detection", {})
    attack_config = config.get("attack", {})
    k_values = _configured_k_values(detection_config)

    selected_by_class: dict[str, list[SelectionSample]] = {}
    selection_json: dict[str, dict[str, Any]] = {}
    selection_rows: list[dict[str, Any]] = []

    for class_name, samples in samples_by_class.items():
        selected, summary = _select_ambiguous_samples(
            model=model,
            samples=samples,
            selection_config=selection_config,
        )
        selected_by_class[str(class_name)] = selected
        selection_json[str(class_name)] = summary
        row = {"class": str(class_name)}
        row.update(summary)
        selection_rows.append(row)

    global_selection = _global_selection_summary(selection_json)
    selection_json["global"] = global_selection
    global_row = {"class": "global"}
    global_row.update(global_selection)
    selection_rows.append(global_row)

    counts_by_class = {
        class_name: {k: _zero_counts(len(selected)) for k in k_values}
        for class_name, selected in selected_by_class.items()
    }

    for class_name, selected in selected_by_class.items():
        for image, clean_probs, _, _ in selected:
            filtered_clean = transform(image)
            filtered_clean_probs = predict_proba(model, filtered_clean)
            clean_pred = int(np.argmax(clean_probs))

            adversarial = attack_generator(model, image, clean_pred, attack_config)
            adv_probs = predict_proba(model, adversarial)
            attack_success = int(np.argmax(adv_probs)) != clean_pred
            filtered_adv_probs = None
            if attack_success:
                filtered_adv_probs = predict_proba(model, transform(adversarial))

            for k in k_values:
                class_counts = counts_by_class[class_name][k]
                if is_detected_topk(clean_probs, filtered_clean_probs, k):
                    class_counts["FP"] += 1
                else:
                    class_counts["TN"] += 1

                if not attack_success:
                    class_counts["disturbed_failure"] += 1
                    continue

                class_counts["test_number"] += 1
                assert filtered_adv_probs is not None
                if is_detected_topk(adv_probs, filtered_adv_probs, k):
                    class_counts["TP"] += 1
                else:
                    class_counts["FN"] += 1

    global_counts = _global_counts(counts_by_class, k_values)
    metrics_json: dict[str, dict[str, dict[str, Any]]] = {}
    metric_rows: list[dict[str, Any]] = []
    for class_name, class_counts in counts_by_class.items():
        metrics_json[class_name] = {}
        baseline_fp = int(class_counts[1]["FP"])
        for k in k_values:
            payload = _metric_payload(class_counts[k], baseline_fp=baseline_fp)
            metrics_json[class_name][str(k)] = payload
            row = {"class": class_name, "k": k}
            row.update(payload)
            metric_rows.append(row)

    metrics_json["global"] = {}
    global_baseline_fp = int(global_counts[1]["FP"])
    for k in k_values:
        payload = _metric_payload(global_counts[k], baseline_fp=global_baseline_fp)
        metrics_json["global"][str(k)] = payload
        row = {"class": "global", "k": k}
        row.update(payload)
        metric_rows.append(row)

    return TopKDetectionResult(
        selection_rows=selection_rows,
        selection_json=selection_json,
        metric_rows=metric_rows,
        metrics_json=metrics_json,
        ambiguous_images_by_class={
            class_name: [item[0] for item in selected]
            for class_name, selected in selected_by_class.items()
        },
    )


def _global_selection_summary(
    selection_by_class: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate selection counters by summing counts and weighting means."""
    total_loaded = sum(int(row["total_loaded"]) for row in selection_by_class.values())
    ambiguous = sum(int(row["ambiguous_candidates"]) for row in selection_by_class.values())
    selected = sum(int(row["selected_for_experiment"]) for row in selection_by_class.values())
    if selected:
        mean_confidence = sum(
            float(row["mean_top1_confidence"]) * int(row["selected_for_experiment"])
            for row in selection_by_class.values()
        ) / selected
        mean_margin = sum(
            float(row["mean_top1_top2_margin"]) * int(row["selected_for_experiment"])
            for row in selection_by_class.values()
        ) / selected
    else:
        mean_confidence = 0.0
        mean_margin = 0.0
    return {
        "total_loaded": int(total_loaded),
        "ambiguous_candidates": int(ambiguous),
        "selected_for_experiment": int(selected),
        "mean_top1_confidence": float(mean_confidence),
        "mean_top1_top2_margin": float(mean_margin),
    }


def _global_counts(
    counts_by_class: Mapping[str, Mapping[int, Mapping[str, int]]],
    k_values: Sequence[int],
) -> dict[int, dict[str, int]]:
    """Aggregate counters by summing class counters for every k."""
    global_counts = {k: _zero_counts(0) for k in k_values}
    for class_counts in counts_by_class.values():
        for k in k_values:
            for field in ("selected", "test_number", "disturbed_failure", "TP", "FN", "FP", "TN"):
                global_counts[k][field] += int(class_counts[k][field])
    return global_counts


def write_topk_detection_outputs(
    *,
    output_dir: Path,
    result: TopKDetectionResult,
) -> dict[str, Path]:
    """Write exactly the official top-k selection and metric artifacts."""
    output_path = Path(output_dir)
    ambiguous_images_dir = write_ambiguous_images(
        output_path / "ambiguous_images",
        result.ambiguous_images_by_class,
    )
    selection_csv = write_metrics_csv(
        output_path / "selection.csv",
        result.selection_rows,
        SELECTION_FIELDS,
    )
    selection_json = write_metrics_json(output_path / "selection.json", result.selection_json)
    metrics_csv = write_metrics_csv(
        output_path / "metrics.csv",
        result.metric_rows,
        METRIC_FIELDS,
    )
    metrics_json = write_metrics_json(output_path / "metrics.json", result.metrics_json)
    return {
        "ambiguous_images": ambiguous_images_dir,
        "selection_csv": selection_csv,
        "selection_json": selection_json,
        "metrics_csv": metrics_csv,
        "metrics_json": metrics_json,
    }


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Return one image as RGB uint8 data for saved ambiguous artifacts."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 2:
        rgb = np.stack([image_array, image_array, image_array], axis=-1)
    elif image_array.ndim == 3 and image_array.shape[0] in (1, 3) and image_array.shape[-1] not in (1, 3):
        rgb = np.transpose(image_array, (1, 2, 0))
        if image_array.shape[0] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        elif is_caffe_input(image_array.reshape((1,) + image_array.shape)):
            rgb = rgb[:, :, ::-1]
    elif image_array.ndim == 3 and image_array.shape[-1] in (1, 3):
        rgb = image_array
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
    else:
        raise ValueError("ambiguous image must have shape HxW, HxWxC, or CxHxW.")

    min_value = float(np.nanmin(rgb)) if rgb.size else 0.0
    max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
    if min_value < 0.0 and min_value >= -0.5 and max_value <= 0.5:
        unit = np.clip(rgb + 0.5, 0.0, 1.0)
    elif max_value > 1.0:
        unit = np.clip(rgb, 0.0, 255.0) / 255.0
    else:
        unit = np.clip(rgb, 0.0, 1.0)
    return np.rint(unit * 255.0).astype(np.uint8)


def _clear_png_files(directory: Path) -> None:
    """Remove stale PNGs from a generated ambiguous-image class directory."""
    if not directory.is_dir():
        return
    for path in directory.glob("*.png"):
        if path.is_file():
            path.unlink()


def write_ambiguous_images(
    output_dir: Path,
    images_by_class: Mapping[str, Sequence[np.ndarray]],
) -> Path:
    """Write selected clean ambiguous images as deterministic per-class PNGs."""
    from PIL import Image

    root = ensure_dir(Path(output_dir))
    for class_name, images in images_by_class.items():
        class_dir = ensure_dir(root / str(class_name))
        _clear_png_files(class_dir)
        for index, image in enumerate(images, start=1):
            image_path = class_dir / "{0:06d}.png".format(index)
            Image.fromarray(_as_uint8_rgb(image), mode="RGB").save(str(image_path))
    return root
