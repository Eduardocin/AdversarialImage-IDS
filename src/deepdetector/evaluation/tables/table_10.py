"""Materialize official Table 10 model-group outputs."""

from __future__ import annotations

try:
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
except Exception:
    pass

import logging
from pathlib import Path
from typing import Any

import numpy as np

from deepdetector.attacks.fgsm_imagenet import generate_fgsm_caffe_image, predict_caffe_label
from deepdetector.attacks.registry import generate_attack
from deepdetector.data.fashion_mnist import load_fashion_mnist_evaluation_split
from deepdetector.data.imagenet import resize_normalized_image
from deepdetector.evaluation.article_reproduction import (
    apply_filter_batch,
    create_restored_mnist_graph,
    predict_labels,
)
from deepdetector.evaluation.defense_aware import create_restored_mnist_m2_graph
from deepdetector.evaluation.detector_metrics import (
    compute_detector_counts,
    compute_precision_recall,
)
from deepdetector.filters.factory import build_filter_from_config
from deepdetector.io.paths import ensure_dir, resolve_project_path
from deepdetector.io.result_writers import write_metrics_csv, write_metrics_json
from deepdetector.models.imagenet_wrappers import (
    CaffeNetCaffeWrapper,
    GoogLeNetCaffeWrapper,
    InceptionV3TensorFlowWrapper,
)


logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


TABLE_10_SCHEMA: list[str] = [
    "no",
    "attack_model",
    "dataset",
    "num_failures",
    "tp",
    "fn",
    "fp",
    "rtp",
    "rtp_percent",
    "recall",
    "precision",
    "f1",
]

TABLE_10_METRIC_FIELDS: list[str] = [
    "num_failures",
    "tp",
    "fn",
    "fp",
    "rtp",
    "rtp_percent",
    "recall",
    "precision",
    "f1",
]


def build_pending_table_10_row(
    *,
    no: int,
    attack_model: str,
    dataset: str,
) -> dict[str, Any]:
    """Build a Table 10 row whose experiment metrics are not available yet."""
    row: dict[str, Any] = {
        "no": no,
        "attack_model": attack_model,
        "dataset": dataset,
    }
    for field in TABLE_10_METRIC_FIELDS:
        row[field] = None
    return row


def normalize_table_10_result(
    *,
    no: int,
    attack_model: str,
    dataset: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Convert a computed row result to the official Table 10 schema."""
    row = build_pending_table_10_row(
        no=no,
        attack_model=attack_model,
        dataset=dataset,
    )
    metrics = result.get("metrics", {})
    source = metrics if isinstance(metrics, dict) else result
    for field in TABLE_10_METRIC_FIELDS:
        if field in source:
            row[field] = source[field]
    return row


def _output_dir(config: dict[str, Any]) -> Path:
    output_config = config.get("output", {})
    configured_dir = output_config.get("dir") or config.get("output_dir")
    output_dir = resolve_project_path(configured_dir)
    if output_dir is None:
        raise ValueError("Table 10 group must define output.dir or output_dir.")
    return output_dir


def _path_text(path_value: Any, field_name: str) -> str:
    path = resolve_project_path(path_value)
    if path is None:
        raise ValueError("Table 10 ImageNet config must define {0}.".format(field_name))
    return str(path)


def build_table_10_googlenet_model(config: dict[str, Any]) -> Any:
    """Instantiate the configured GoogLeNet model wrapper."""
    model_config = config.get("model", {})
    return GoogLeNetCaffeWrapper(
        model_dir=_path_text(model_config.get("model_dir"), "model.model_dir"),
        deploy_prototxt=_path_text(model_config.get("deploy_proto"), "model.deploy_proto"),
        caffemodel=_path_text(model_config.get("caffemodel"), "model.caffemodel"),
        attack_deploy_prototxt=(
            _path_text(model_config.get("attack_deploy_proto"), "model.attack_deploy_proto")
            if model_config.get("attack_deploy_proto")
            else None
        ),
        mean_file=(
            str(resolve_project_path(model_config.get("mean_file")))
            if model_config.get("mean_file")
            else None
        ),
        use_gpu=bool(model_config.get("use_gpu", False)),
        batch_size=int(model_config.get("batch_size", 32)),
    )


def build_table_10_caffenet_model(config: dict[str, Any]) -> Any:
    """Instantiate the configured CaffeNet model wrapper."""
    model_config = config.get("model", {})
    return CaffeNetCaffeWrapper(
        model_dir=_path_text(model_config.get("model_dir"), "model.model_dir"),
        deploy_prototxt=_path_text(model_config.get("deploy_proto"), "model.deploy_proto"),
        caffemodel=_path_text(model_config.get("caffemodel"), "model.caffemodel"),
        attack_deploy_prototxt=(
            _path_text(model_config.get("attack_deploy_proto"), "model.attack_deploy_proto")
            if model_config.get("attack_deploy_proto")
            else None
        ),
        mean_file=(
            str(resolve_project_path(model_config.get("mean_file")))
            if model_config.get("mean_file")
            else None
        ),
        use_gpu=bool(model_config.get("use_gpu", False)),
        batch_size=int(model_config.get("batch_size", 32)),
    )


def build_table_10_inception_v3_model(config: dict[str, Any]) -> Any:
    """Instantiate the configured Inception v3 TensorFlow wrapper."""
    model_config = config.get("model", {})
    return InceptionV3TensorFlowWrapper(
        graph_path=_path_text(model_config.get("graph_path"), "model.graph_path"),
        input_map_name=str(model_config.get("input_map_name", "Mul:0")),
        output_tensor_name=str(model_config.get("output_tensor", "softmax/logits:0")),
        batch_size=int(model_config.get("batch_size", 32)),
    )


def _build_table_10_imagenet_model(config: dict[str, Any]) -> Any:
    model_group = str(config.get("model_group", "")).lower()
    if not model_group:
        model_name = str(config.get("model", {}).get("name", "")).lower()
        if model_name == "googlenet_caffe":
            model_group = "googlenet"
        elif model_name == "caffenet":
            model_group = "caffenet"
        elif model_name == "inception_v3":
            model_group = "inception_v3"
    if model_group == "googlenet":
        return build_table_10_googlenet_model(config)
    if model_group == "caffenet":
        return build_table_10_caffenet_model(config)
    if model_group == "inception_v3":
        return build_table_10_inception_v3_model(config)
    raise ValueError("Unsupported Table 10 ImageNet model group: {0}".format(model_group))


def _validate_table_10_model(model: Any) -> None:
    if not (hasattr(model, "predict_preprocessed_batch") or hasattr(model, "predict_batch")):
        raise NotImplementedError("Table 10 ImageNet requires model scores or predictions.")


def _predict_one(model: Any, image: np.ndarray) -> int:
    image_array = np.asarray(image, dtype=np.float32)

    if isinstance(model, InceptionV3TensorFlowWrapper):
        label = model.predict_preprocessed_label(
            image_array.reshape((1,) + image_array.shape)
        )
        return int(np.asarray(label).reshape(-1)[0])

    return predict_caffe_label(model, image_array)


def _configured_n_samples(config: dict[str, Any]) -> int | None:
    dataset_config = config.get("dataset", {})
    evaluation_config = config.get("evaluation", {})
    value = evaluation_config.get("n_samples", dataset_config.get("n_samples"))
    if value in (None, "", "all"):
        return None
    n_samples = int(value)
    if n_samples <= 0:
        raise ValueError("Table 10 ImageNet n_samples must be positive or 'all'.")
    return n_samples


def _preprocess_table_10_image(model: Any, image_size: int):
    def preprocess(image: np.ndarray) -> np.ndarray:
        resized = resize_normalized_image(image, image_size=image_size)
        if hasattr(model, "preprocess"):
            return np.asarray(model.preprocess(resized)[0], dtype=np.float32)
        return resized.astype(np.float32)

    return preprocess


def _read_rgb_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(str(path)) as image:
        rgb_image = image.convert("RGB")
        return (np.asarray(rgb_image, dtype=np.float32) / 255.0).astype(np.float32)


def _ordered_class_names(
    class_indices: dict[str, Any],
    class_order: list[str] | None = None,
) -> list[str]:
    if class_order is None:
        return sorted(str(class_name) for class_name in class_indices)
    return [str(class_name) for class_name in class_order]


def _class_folder_rows_by_class(
    images_dir: Path,
    class_indices: dict[str, Any],
    class_order: list[str] | None = None,
) -> dict[str, list[tuple[Path, int]]]:
    rows_by_class: dict[str, list[tuple[Path, int]]] = {}
    ordered_classes = _ordered_class_names(class_indices, class_order=class_order)

    for class_name in ordered_classes:
        if class_name not in class_indices:
            raise ValueError(
                "Table 10 ImageNet class_order references unknown class: {0}".format(
                    class_name
                )
            )
        label_index = class_indices[class_name]
        class_dir = images_dir / str(class_name)
        if not class_dir.is_dir():
            raise ValueError("Missing ImageNet class directory: {0}".format(class_dir))
        rows_by_class[class_name] = [
            (path, int(label_index))
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return rows_by_class


def _class_folder_rows(
    images_dir: Path,
    class_indices: dict[str, Any],
    class_order: list[str] | None = None,
    class_quotas: dict[str, Any] | None = None,
) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    rows_by_class = _class_folder_rows_by_class(
        images_dir,
        class_indices,
        class_order=class_order,
    )
    ordered_classes = _ordered_class_names(class_indices, class_order=class_order)

    for class_name in ordered_classes:
        class_rows = rows_by_class[class_name]
        if class_quotas is not None:
            if class_name not in class_quotas:
                raise ValueError(
                    "Table 10 ImageNet class_quotas must define class: {0}".format(
                        class_name
                    )
                )
            quota = int(class_quotas[class_name])
            if quota < 0:
                raise ValueError("Table 10 ImageNet class quota must be non-negative.")
            if len(class_rows) < quota:
                raise ValueError(
                    "Table 10 ImageNet class {0} has {1} samples; quota requires {2}.".format(
                        class_name,
                        len(class_rows),
                        quota,
                    )
                )
            class_rows = class_rows[:quota]
        rows.extend(class_rows)
    return rows


def _requires_clean_correct_selection(config: dict[str, Any]) -> bool:
    dataset_config = config.get("dataset", {})
    evaluation_config = config.get("evaluation", {})
    if "require_clean_correct" in dataset_config:
        return bool(dataset_config["require_clean_correct"])
    if "require_clean_correct" in evaluation_config:
        return bool(evaluation_config["require_clean_correct"])
    return (
        str(config.get("model_group", "")).lower() == "inception_v3"
        and isinstance(dataset_config.get("class_quotas"), dict)
    )


def _append_processed_image(
    *,
    images: list[np.ndarray],
    labels: list[int],
    processed: np.ndarray,
    label_index: int,
    expected_shape: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if expected_shape is None:
        expected_shape = processed.shape
    elif processed.shape != expected_shape:
        raise ValueError(
            "Table 10 ImageNet preprocessing returned inconsistent shapes: {0} and {1}".format(
                expected_shape,
                processed.shape,
            )
        )
    images.append(processed)
    labels.append(int(label_index))
    return expected_shape


def _load_clean_correct_table_10_imagenet_class_folders(
    *,
    config: dict[str, Any],
    model: Any,
    n_samples: int | None,
    image_size: int,
    images_dir: Path,
    class_indices: dict[str, Any],
    class_order: list[str] | None,
    class_quotas: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    rows_by_class = _class_folder_rows_by_class(
        images_dir,
        class_indices,
        class_order=class_order,
    )
    ordered_classes = _ordered_class_names(class_indices, class_order=class_order)
    preprocess = _preprocess_table_10_image(model, image_size=image_size)
    images: list[np.ndarray] = []
    labels: list[int] = []
    expected_shape: tuple[int, ...] | None = None
    clean_errors = 0
    summary: dict[str, Any] = {
        "classes": ordered_classes,
        "quotas": {class_name: int(class_quotas[class_name]) for class_name in ordered_classes},
        "candidate_counts": {
            class_name: len(rows_by_class[class_name]) for class_name in ordered_classes
        },
        "candidates_read": {},
        "clean_errors": {},
        "clean_correct": {},
    }
    config["_table_10_dataset_summary"] = summary

    for class_name in ordered_classes:
        if class_name not in class_quotas:
            raise ValueError(
                "Table 10 ImageNet class_quotas must define class: {0}".format(
                    class_name
                )
            )
        quota = int(class_quotas[class_name])
        if quota < 0:
            raise ValueError("Table 10 ImageNet class quota must be non-negative.")
        if len(rows_by_class[class_name]) < quota:
            raise ValueError(
                "Insufficient ImageNet candidates for class {0}: required at least {1}, found {2}.".format(
                    class_name,
                    quota,
                    len(rows_by_class[class_name]),
                )
            )

        selected_for_class = 0
        read_for_class = 0
        clean_errors_for_class = 0
        for path, label_index in rows_by_class[class_name]:
            read_for_class += 1
            processed = np.asarray(preprocess(_read_rgb_image(path)), dtype=np.float32)
            clean_pred = _predict_one(model, processed)
            if clean_pred != int(label_index):
                clean_errors += 1
                clean_errors_for_class += 1
                continue

            expected_shape = _append_processed_image(
                images=images,
                labels=labels,
                processed=processed,
                label_index=int(label_index),
                expected_shape=expected_shape,
            )
            selected_for_class += 1
            if selected_for_class == quota:
                break

        summary["candidates_read"][class_name] = read_for_class
        summary["clean_errors"][class_name] = clean_errors_for_class
        summary["clean_correct"][class_name] = selected_for_class
        config["_table_10_dataset_summary"] = summary
        if selected_for_class < quota:
            raise ValueError(
                "Insufficient clean-correct ImageNet samples for class {0} and model {1}: required {2}, found {3}.".format(
                    class_name,
                    str(config.get("model_group", "")).lower() or "unknown",
                    quota,
                    selected_for_class,
                )
            )

    if n_samples is not None:
        images = images[:n_samples]
        labels = labels[:n_samples]

    logger.info(
        "Selected %d clean-correct ImageNet samples for Table 10 %s; discarded %d clean errors.",
        len(images),
        config.get("model_group"),
        clean_errors,
    )
    if not images:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _dataset_summary_for_manifest(config: dict[str, Any]) -> dict[str, Any] | None:
    summary = config.get("_table_10_dataset_summary")
    if isinstance(summary, dict):
        if summary.get("name") == "fashion_mnist":
            return {
                "name": "fashion_mnist",
                "domain": summary.get("domain", "mnist_compatible"),
                "split": summary.get("split", "test"),
                "csv_path": summary.get("csv_path"),
                "split_strategy": dict(summary.get("split_strategy", {})),
                "training_sample": dict(summary.get("training_sample", {})),
                "evaluation_sample": dict(summary.get("evaluation_sample", {})),
                "checkpoint_training": dict(config.get("checkpoint_training", {})),
                "image_shape": list(summary.get("image_shape", [28, 28, 1])),
                "value_range": summary.get(
                    "value_range",
                    {"min": 0.0, "max": 1.0},
                ),
                "class_order": list(summary.get("class_order", [])),
                "class_quotas": dict(summary.get("class_quotas", {})),
                "candidate_counts": dict(summary.get("candidate_counts", {})),
                "candidates_read": dict(summary.get("candidates_read", {})),
                "clean_errors": dict(summary.get("clean_errors", {})),
                "clean_correct": dict(summary.get("clean_correct", {})),
                "selected_clean_correct": dict(summary.get("selected_clean_correct", {})),
                "selection_policy": summary.get("selection_policy"),
            }
        return {
            "classes": list(summary.get("classes", [])),
            "quotas": dict(summary.get("quotas", {})),
            "candidate_counts": dict(summary.get("candidate_counts", {})),
            "candidates_read": dict(summary.get("candidates_read", {})),
            "clean_errors": dict(summary.get("clean_errors", {})),
            "clean_correct": dict(summary.get("clean_correct", {})),
        }
    return None


def _load_table_10_imagenet_class_folders(
    *,
    config: dict[str, Any],
    model: Any,
    n_samples: int | None,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset_config = config.get("dataset", {})
    images_dir = _path_text(dataset_config.get("images_dir"), "dataset.images_dir")
    class_indices = dataset_config.get("class_indices", {})
    if not isinstance(class_indices, dict) or not class_indices:
        raise ValueError("Table 10 ImageNet dataset must define class_indices.")

    class_order_config = dataset_config.get("class_order")
    class_order = (
        [str(class_name) for class_name in class_order_config]
        if class_order_config is not None
        else None
    )
    class_quotas_config = dataset_config.get("class_quotas")
    class_quotas = (
        {str(class_name): int(quota) for class_name, quota in class_quotas_config.items()}
        if isinstance(class_quotas_config, dict)
        else None
    )
    if _requires_clean_correct_selection(config):
        if class_quotas is None:
            raise ValueError(
                "Table 10 ImageNet clean-correct selection requires class_quotas."
            )
        return _load_clean_correct_table_10_imagenet_class_folders(
            config=config,
            model=model,
            n_samples=n_samples,
            image_size=image_size,
            images_dir=Path(images_dir),
            class_indices=class_indices,
            class_order=class_order,
            class_quotas=class_quotas,
        )

    rows = _class_folder_rows(
        Path(images_dir),
        class_indices,
        class_order=class_order,
        class_quotas=class_quotas,
    )
    if bool(dataset_config.get("shuffle", False)):
        rng = np.random.RandomState(int(config.get("evaluation", {}).get("seed", 20170830)))
        order = rng.permutation(len(rows))
        rows = [rows[int(index)] for index in order]
    if n_samples is not None:
        rows = rows[:n_samples]

    preprocess = _preprocess_table_10_image(model, image_size=image_size)
    images = []
    labels = []
    expected_shape = None
    for path, label_index in rows:
        processed = np.asarray(preprocess(_read_rgb_image(path)), dtype=np.float32)
        expected_shape = _append_processed_image(
            images=images,
            labels=labels,
            processed=processed,
            label_index=int(label_index),
            expected_shape=expected_shape,
        )

    if not images:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _load_table_10_imagenet_images(
    config: dict[str, Any],
    model: Any,
) -> tuple[np.ndarray, np.ndarray]:
    dataset_config = config.get("dataset", {})
    images_dir = dataset_config.get("images_dir")
    if not images_dir:
        raise ValueError("Table 10 ImageNet dataset must define images_dir.")

    n_samples = _configured_n_samples(config)
    image_size = int(dataset_config.get("image_size", 224))
    images, labels = _load_table_10_imagenet_class_folders(
        config=config,
        model=model,
        n_samples=n_samples,
        image_size=image_size,
    )

    if len(images) == 0:
        raise ValueError("Table 10 ImageNet dataset is empty.")
    logger.info("Loaded %d ImageNet samples for Table 10 %s.", len(images), config.get("model_group"))
    return images.astype(np.float32), labels.astype(np.int32)


def _load_table_10_googlenet_images(
    config: dict[str, Any],
    model: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for existing GoogLeNet tests and callers."""
    return _load_table_10_imagenet_images(config, model)


def _table_10_filter(config: dict[str, Any]):
    filter_config = config.get("filter") or {
        "name": "proposed_detection_filter",
        "type": "proposed_detection_filter",
    }
    _, filter_fn, _ = build_filter_from_config(filter_config)
    return filter_fn


def _attack_kwargs(row_config: dict[str, Any]) -> dict[str, Any]:
    attack_config = dict(row_config.get("attack", {}))
    attack_config.pop("name", None)
    return attack_config


def _fgsm_epsilon_255(attack_config: dict[str, Any]) -> float:
    if "epsilon_255" in attack_config:
        return float(attack_config["epsilon_255"])
    if "epsilon" in attack_config:
        return float(attack_config["epsilon"]) * 255.0
    return 1.0


def _generate_table_10_adversarial(
    *,
    attack_name: str,
    row_config: dict[str, Any],
    model: Any,
    clean_image: np.ndarray,
    true_label: int,
    clean_pred: int,
) -> np.ndarray:
    attack_config = row_config.get("attack", {})
    if attack_name == "fgsm":
        return generate_fgsm_caffe_image(
            model=model,
            image=clean_image,
            class_id=clean_pred,
            epsilon_255=_fgsm_epsilon_255(attack_config),
            clip_min=float(attack_config.get("clip_min", 0.0)),
            clip_max=float(attack_config.get("clip_max", 255.0)),
        )

    adversarial_batch = generate_attack(
        attack_name,
        model=model,
        images=clean_image.reshape((1,) + clean_image.shape),
        labels=np.asarray([true_label], dtype=np.int32),
        **_attack_kwargs(row_config),
    )
    return np.asarray(adversarial_batch[0], dtype=np.float32)


def _is_gradient_attack(attack_name: str) -> bool:
    return str(attack_name).lower() == "deepfool"


def _table_10_metrics_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = compute_detector_counts(records)
    rates = compute_precision_recall(counts)
    return {
        "num_failures": int(counts["n_discarded_attack_failed"]),
        "tp": int(counts["TP"]),
        "fn": int(counts["FN"]),
        "fp": int(counts["FP"]),
        "rtp": int(counts["TTP"]),
        "rtp_percent": float(rates["ttp_rate"] * 100.0),
        "recall": float(rates["recall"] * 100.0),
        "precision": float(rates["precision"] * 100.0),
        "f1": float(rates["f1"] * 100.0),
    }


def _progress_interval(total: int) -> int:
    if total <= 20:
        return 1
    return 25


def evaluate_table_10_imagenet_row(
    group_config: dict[str, Any],
    row_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one implemented ImageNet Table 10 row."""
    attack_config = row_config.get("attack", {})
    attack_name = str(attack_config.get("name", "")).strip()
    if not attack_name:
        raise ValueError("Implemented Table 10 rows must define attack.name.")

    logger.info(
        "Evaluating Table 10 row %s with %s.",
        row_config.get("no", ""),
        attack_name,
    )
    model = _build_table_10_imagenet_model(group_config)
    _validate_table_10_model(model)
    if _is_gradient_attack(attack_name) and not hasattr(model, "gradient"):
        raise NotImplementedError("Table 10 DeepFool requires gradient access on the model.")
    if str(group_config.get("model_group", "")).lower() == "googlenet":
        images, labels = _load_table_10_googlenet_images(group_config, model)
    else:
        images, labels = _load_table_10_imagenet_images(group_config, model)
    filter_fn = _table_10_filter(group_config)

    records: list[dict[str, Any]] = []
    total_images = len(images)
    progress_every = _progress_interval(total_images)
    for sample_index, clean_image in enumerate(images):
        true_label = int(labels[sample_index])
        clean_pred = _predict_one(model, clean_image)
        if clean_pred != true_label:
            records.append(
                {
                    "sample_index": int(sample_index),
                    "true_label": true_label,
                    "clean_pred": int(clean_pred),
                    "discarded_clean_error": True,
                }
            )
            if (sample_index + 1) == 1 or (sample_index + 1) == total_images or (sample_index + 1) % progress_every == 0:
                counts = compute_detector_counts(records)
                logger.info(
                    "Table 10 row %s progress %d/%d | clean_errors=%d attack_failures=%d tp=%d fn=%d fp=%d.",
                    row_config.get("no", ""),
                    sample_index + 1,
                    total_images,
                    counts["n_discarded_clean_error"],
                    counts["n_discarded_attack_failed"],
                    counts["TP"],
                    counts["FN"],
                    counts["FP"],
                )
            continue

        logger.info(
            "Table 10 row %s sample %d/%d: clean_pred=%d true_label=%d; generating %s.",
            row_config.get("no", ""),
            sample_index + 1,
            total_images,
            clean_pred,
            true_label,
            attack_name,
        )
        adversarial_image = _generate_table_10_adversarial(
            attack_name=attack_name,
            row_config=row_config,
            model=model,
            clean_image=clean_image,
            true_label=true_label,
            clean_pred=clean_pred,
        )
        if adversarial_image.shape != clean_image.shape:
            raise ValueError("Adversarial image shape does not match clean image shape.")

        adv_pred = _predict_one(model, adversarial_image)
        if adv_pred == clean_pred:
            records.append(
                {
                    "sample_index": int(sample_index),
                    "true_label": true_label,
                    "clean_pred": int(clean_pred),
                    "adv_pred": int(adv_pred),
                    "discarded_attack_failed": True,
                }
            )
            if (sample_index + 1) == 1 or (sample_index + 1) == total_images or (sample_index + 1) % progress_every == 0:
                counts = compute_detector_counts(records)
                logger.info(
                    "Table 10 row %s progress %d/%d | clean_errors=%d attack_failures=%d tp=%d fn=%d fp=%d.",
                    row_config.get("no", ""),
                    sample_index + 1,
                    total_images,
                    counts["n_discarded_clean_error"],
                    counts["n_discarded_attack_failed"],
                    counts["TP"],
                    counts["FN"],
                    counts["FP"],
                )
            continue

        filtered_clean = np.asarray(filter_fn(clean_image), dtype=np.float32).reshape(
            clean_image.shape
        )
        filtered_adv = np.asarray(filter_fn(adversarial_image), dtype=np.float32).reshape(
            adversarial_image.shape
        )
        filtered_clean_pred = _predict_one(model, filtered_clean)
        filtered_adv_pred = _predict_one(model, filtered_adv)

        records.append(
            {
                "sample_index": int(sample_index),
                "true_label": true_label,
                "clean_pred": int(clean_pred),
                "adv_pred": int(adv_pred),
                "filtered_clean_pred": int(filtered_clean_pred),
                "filtered_adv_pred": int(filtered_adv_pred),
                "detected": bool(filtered_adv_pred != adv_pred),
                "corrected": bool(filtered_adv_pred == true_label),
                "false_positive": bool(filtered_clean_pred != clean_pred),
            }
        )
        if (sample_index + 1) == 1 or (sample_index + 1) == total_images or (sample_index + 1) % progress_every == 0:
            counts = compute_detector_counts(records)
            logger.info(
                "Table 10 row %s progress %d/%d | clean_errors=%d attack_failures=%d tp=%d fn=%d fp=%d.",
                row_config.get("no", ""),
                sample_index + 1,
                total_images,
                counts["n_discarded_clean_error"],
                counts["n_discarded_attack_failed"],
                counts["TP"],
                counts["FN"],
                counts["FP"],
            )

    metrics = _table_10_metrics_from_records(records)
    logger.info(
        "Table 10 row %s complete: num_failures=%s tp=%s fn=%s fp=%s rtp=%s.",
        row_config.get("no", ""),
        metrics["num_failures"],
        metrics["tp"],
        metrics["fn"],
        metrics["fp"],
        metrics["rtp"],
    )
    return {"metrics": metrics}


def evaluate_table_10_googlenet_row(
    group_config: dict[str, Any],
    row_config: dict[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper for the GoogLeNet Table 10 evaluator."""
    config = dict(group_config)
    config.setdefault("model_group", "googlenet")
    return evaluate_table_10_imagenet_row(config, row_config)


def _table_10_checkpoint_dir(config: dict[str, Any]) -> str:
    checkpoint_dir = resolve_project_path(config.get("model", {}).get("checkpoint_dir"))
    if checkpoint_dir is None:
        raise ValueError("Table 10 Fashion-MNIST model.checkpoint_dir is required.")
    if not checkpoint_dir.is_dir():
        raise IOError("Fashion-MNIST checkpoint_dir not found: {0}".format(checkpoint_dir))
    return str(checkpoint_dir)


def _build_table_10_fashion_mnist_graph(config: dict[str, Any]) -> dict[str, Any]:
    model_group = str(config.get("model_group", "")).lower()
    model_config = config.get("model", {})
    if str(model_config.get("family", "")).lower() != "mnist":
        raise ValueError("Fashion-MNIST Table 10 requires model.family=mnist.")
    if str(model_config.get("dataset_name", "")).lower() != "fashion_mnist":
        raise ValueError("Fashion-MNIST model.dataset_name must be fashion_mnist.")
    if list(model_config.get("input_shape", [28, 28, 1])) != [28, 28, 1]:
        raise ValueError("Fashion-MNIST model.input_shape must be [28, 28, 1].")
    if int(model_config.get("num_classes", 10)) != 10:
        raise ValueError("Fashion-MNIST model.num_classes must be 10.")

    checkpoint_dir = _table_10_checkpoint_dir(config)
    if model_group == "m1":
        return create_restored_mnist_graph(checkpoint_dir)
    if model_group == "m2":
        return create_restored_mnist_m2_graph(checkpoint_dir)
    raise ValueError("Fashion-MNIST Table 10 supports only model_group m1 or m2.")


def _mnist_graph_predict(
    graph: dict[str, Any],
    images: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    return predict_labels(
        graph["sess"],
        graph["x"],
        graph["predictions"],
        images,
        batch_size=batch_size,
    )


def _fashion_mnist_quotas(config: dict[str, Any]) -> dict[str, int]:
    dataset_config = config.get("dataset", {})
    quotas = dict(dataset_config.get("class_quotas", {}))
    if not quotas:
        class_order = list(dataset_config.get("class_order", []))
        quotas = {str(class_name): 100 for class_name in class_order}
    return {str(key): int(value) for key, value in quotas.items()}


def _fashion_mnist_class_indices(config: dict[str, Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in dict(config.get("dataset", {}).get("class_indices", {})).items()
    }


def _select_fashion_mnist_clean_correct(
    *,
    config: dict[str, Any],
    images: np.ndarray,
    labels: np.ndarray,
    clean_predictions: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_config = config.get("dataset", {})
    class_order = [str(class_name) for class_name in dataset_config.get("class_order", [])]
    class_indices = _fashion_mnist_class_indices(config)
    quotas = _fashion_mnist_quotas(config)
    require_clean_correct = bool(dataset_config.get("require_clean_correct", True))
    selected_indices: list[int] = []
    clean_errors: dict[str, int] = {}
    clean_correct: dict[str, int] = {}
    candidates_read: dict[str, int] = {}

    for class_name in class_order:
        if class_name not in class_indices:
            raise ValueError("Fashion-MNIST class_indices missing class: {0}".format(class_name))
        quota = int(quotas[class_name])
        label = int(class_indices[class_name])
        class_candidates = np.flatnonzero(labels == label).astype(np.int64)
        candidates_read[class_name] = int(len(class_candidates))
        clean_correct_mask = clean_predictions[class_candidates] == label
        correct_indices = class_candidates[clean_correct_mask]
        clean_errors[class_name] = int(len(class_candidates) - len(correct_indices))
        clean_correct[class_name] = int(len(correct_indices))
        if require_clean_correct:
            selected_indices.extend(correct_indices[:quota].tolist())
        else:
            if len(class_candidates) < quota:
                raise ValueError(
                    "Insufficient Fashion-MNIST samples for class {0}: required {1}, found {2}.".format(
                        class_name,
                        quota,
                        len(class_candidates),
                    )
                )
            selected_indices.extend(class_candidates[:quota].tolist())

    metadata.update(
        {
            "classes": class_order,
            "quotas": quotas,
            "candidates_read": candidates_read,
            "clean_errors": clean_errors,
            "clean_correct": clean_correct,
            "selected_clean_correct": clean_correct if require_clean_correct else candidates_read,
            "selection_policy": (
                "discard_clean_errors" if require_clean_correct else "include_all_candidates"
            ),
        }
    )
    selected = np.asarray(selected_indices, dtype=np.int64)
    if require_clean_correct and len(selected) == 0:
        raise ValueError("Fashion-MNIST evaluation has no clean-correct samples.")
    return images[selected], labels[selected], clean_predictions[selected]


def _generate_fashion_mnist_adversarial(
    *,
    graph: dict[str, Any],
    row_config: dict[str, Any],
    config: dict[str, Any],
    images: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    attack_config = dict(row_config.get("attack", {}))
    attack_name = str(attack_config.pop("name", "")).strip().lower()
    if attack_name == "fgsm":
        return generate_attack(
            "fgsm",
            sess=graph["sess"],
            model=graph["model"],
            x_placeholder=graph["x"],
            images=images,
            eps=float(attack_config.pop("epsilon", attack_config.pop("eps", 0.2))),
            clip_min=float(attack_config.pop("clip_min", 0.0)),
            clip_max=float(attack_config.pop("clip_max", 1.0)),
            **attack_config,
        )
    if attack_name == "cw_l2_nn_robust":
        robust_root = (
            attack_config.pop("nn_robust_attacks_root", None)
            or config.get("evaluation", {}).get("nn_robust_attacks_root")
            or "nn_robust_attacks"
        )
        return generate_attack(
            "cw_l2_nn_robust",
            graph=graph,
            images=images,
            labels=labels,
            nn_robust_attacks_root=robust_root,
            **attack_config,
        )
    raise ValueError("Unsupported Fashion-MNIST Table 10 attack: {0}".format(attack_name))


def evaluate_table_10_fashion_mnist_row(
    group_config: dict[str, Any],
    row_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one Fashion-MNIST Table 10 extension row."""
    attack_name = str(row_config.get("attack", {}).get("name", "")).strip().lower()
    if attack_name not in {"fgsm", "cw_l2_nn_robust"}:
        raise ValueError("Unsupported Fashion-MNIST Table 10 attack: {0}".format(attack_name))

    graph = _build_table_10_fashion_mnist_graph(group_config)
    batch_size = int(group_config.get("evaluation", {}).get("batch_size", 256))
    filter_fn = _table_10_filter(group_config)

    try:
        images, labels, metadata = load_fashion_mnist_evaluation_split(
            group_config.get("dataset", {})
        )
        clean_predictions_all = _mnist_graph_predict(graph, images, batch_size)
        selected_images, selected_labels, clean_predictions = _select_fashion_mnist_clean_correct(
            config=group_config,
            images=images,
            labels=labels,
            clean_predictions=clean_predictions_all,
            metadata=metadata,
        )
        group_config["_table_10_dataset_summary"] = metadata

        adversarial_images = _generate_fashion_mnist_adversarial(
            graph=graph,
            row_config=row_config,
            config=group_config,
            images=selected_images,
            labels=selected_labels,
        )
        adversarial_images = np.asarray(adversarial_images, dtype=np.float32)
        if adversarial_images.shape != selected_images.shape:
            raise ValueError("Adversarial image shape does not match clean image shape.")

        adversarial_predictions = _mnist_graph_predict(
            graph,
            adversarial_images,
            batch_size,
        )
        filtered_clean = apply_filter_batch(filter_fn, selected_images)
        filtered_adv = apply_filter_batch(filter_fn, adversarial_images)
        filtered_clean_predictions = _mnist_graph_predict(graph, filtered_clean, batch_size)
        filtered_adv_predictions = _mnist_graph_predict(graph, filtered_adv, batch_size)

        records: list[dict[str, Any]] = []
        for sample_index, true_label in enumerate(selected_labels):
            clean_pred = int(clean_predictions[sample_index])
            adv_pred = int(adversarial_predictions[sample_index])
            if adv_pred == clean_pred:
                records.append(
                    {
                        "sample_index": int(sample_index),
                        "true_label": int(true_label),
                        "clean_pred": clean_pred,
                        "adv_pred": adv_pred,
                        "discarded_attack_failed": True,
                    }
                )
                continue
            filtered_adv_pred = int(filtered_adv_predictions[sample_index])
            records.append(
                {
                    "sample_index": int(sample_index),
                    "true_label": int(true_label),
                    "clean_pred": clean_pred,
                    "adv_pred": adv_pred,
                    "filtered_clean_pred": int(filtered_clean_predictions[sample_index]),
                    "filtered_adv_pred": filtered_adv_pred,
                    "detected": bool(filtered_adv_pred != adv_pred),
                    "corrected": bool(filtered_adv_pred == int(true_label)),
                    "false_positive": bool(
                        int(filtered_clean_predictions[sample_index]) != clean_pred
                    ),
                }
            )
        return {"metrics": _table_10_metrics_from_records(records)}
    finally:
        graph["sess"].close()


def _is_table_10_imagenet_attack(
    group_config: dict[str, Any],
    row_config: dict[str, Any],
) -> bool:
    model_group = str(group_config.get("model_group", "")).lower()
    attack_name = str(row_config.get("attack", {}).get("name", "")).lower()
    return (
        str(group_config.get("dataset", {}).get("name", "")).lower() == "imagenet"
        and (
            (model_group == "googlenet" and attack_name in {"fgsm", "deepfool"})
            or (model_group == "caffenet" and attack_name == "deepfool")
            or (model_group == "inception_v3" and attack_name in {"cw_l2", "cw_linf"})
        )
    )


def _is_table_10_fashion_mnist_attack(
    group_config: dict[str, Any],
    row_config: dict[str, Any],
) -> bool:
    model_group = str(group_config.get("model_group", "")).lower()
    attack_name = str(row_config.get("attack", {}).get("name", "")).lower()
    return (
        str(group_config.get("dataset", {}).get("name", "")).lower() == "fashion_mnist"
        and (
            (model_group == "m1" and attack_name == "fgsm")
            or (model_group == "m2" and attack_name == "cw_l2_nn_robust")
        )
    )


def _row_result(group_config: dict[str, Any], row_config: dict[str, Any]) -> dict[str, Any]:
    metrics = row_config.get("metrics")
    if isinstance(metrics, dict):
        return {"metrics": metrics}
    if _is_table_10_fashion_mnist_attack(group_config, row_config):
        return evaluate_table_10_fashion_mnist_row(group_config, row_config)
    if _is_table_10_imagenet_attack(group_config, row_config):
        if str(group_config.get("model_group", "")).lower() == "googlenet":
            return evaluate_table_10_googlenet_row(group_config, row_config)
        return evaluate_table_10_imagenet_row(group_config, row_config)
    return row_config


def save_table_10_outputs(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    dataset_group: str,
    model_group: str,
    experiment_id: str | None = None,
    dataset_summary: dict[str, Any] | None = None,
    manifest_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write the official CSV and JSON outputs for one Table 10 group."""
    output_path = ensure_dir(output_dir)
    csv_path = write_metrics_csv(output_path / "metrics.csv", rows, TABLE_10_SCHEMA)
    json_path = write_metrics_json(
        output_path / "metrics.json",
        {
            "table": 10,
            "dataset_group": dataset_group,
            "model_group": model_group,
            "rows": rows,
        },
    )
    outputs = {"csv": csv_path, "json": json_path}
    if manifest_entries is not None:
        manifest_payload: dict[str, Any] = {
            "table": 10,
            "dataset_group": dataset_group,
            "model_group": model_group,
            "rows": manifest_entries,
        }
        if experiment_id:
            manifest_payload["experiment_id"] = experiment_id
        if dataset_summary is not None:
            manifest_payload["dataset"] = dataset_summary
        outputs["manifest"] = write_metrics_json(
            output_path / "manifest.json",
            manifest_payload,
        )
    return outputs


def run_table_10_group(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize one configured Table 10 model group."""
    rows_config = list(config.get("rows", []))
    if not rows_config:
        raise ValueError("Table 10 group must define rows.")

    model_group = str(config.get("model_group", "")).strip()
    if not model_group:
        raise ValueError("Table 10 group must define model_group.")

    dataset_label = str(config.get("dataset_label", "")).strip()
    if not dataset_label:
        raise ValueError("Table 10 group must define dataset_label.")

    dataset_group = str(
        config.get("dataset_group") or config.get("dataset", {}).get("name", "")
    ).strip()
    if not dataset_group:
        raise ValueError("Table 10 group must define dataset.name or dataset_group.")

    rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    write_manifest = bool(config.get("output", {}).get("manifest", False)) or model_group in {
        "caffenet",
        "inception_v3",
    }
    for row_config in rows_config:
        no = int(row_config["no"])
        attack_model = str(row_config["attack_model"])
        status = str(row_config.get("status", "planned"))
        if status != "implemented":
            manifest_entry = {
                "no": no,
                "attack_model": attack_model,
                "status": status,
            }
            if row_config.get("blocked_reason"):
                manifest_entry["blocked_reason"] = str(row_config["blocked_reason"])
                if model_group in {"caffenet", "inception_v3"}:
                    write_manifest = True
            manifest_entries.append(manifest_entry)
            rows.append(
                build_pending_table_10_row(
                    no=no,
                    attack_model=attack_model,
                    dataset=dataset_label,
                )
            )
            continue

        try:
            result = _row_result(config, row_config)
        except Exception as exc:
            if model_group != "inception_v3":
                raise
            rows.append(
                build_pending_table_10_row(
                    no=no,
                    attack_model=attack_model,
                    dataset=dataset_label,
                )
            )
            manifest_entries.append(
                {
                    "no": no,
                    "attack_model": attack_model,
                    "status": "blocked",
                    "blocked_reason": str(exc),
                }
            )
            write_manifest = True
            continue

        rows.append(
            normalize_table_10_result(
                no=no,
                attack_model=attack_model,
                dataset=dataset_label,
                result=result,
            )
        )
        manifest_entries.append(
            {
                "no": no,
                "attack_model": attack_model,
                "status": "completed",
            }
        )
        if str(config.get("dataset", {}).get("name", "")).lower() == "fashion_mnist":
            manifest_entries[-1]["attack"] = dict(row_config.get("attack", {}))

    save_table_10_outputs(
        rows=rows,
        output_dir=_output_dir(config),
        dataset_group=dataset_group,
        model_group=model_group,
        experiment_id=str(config.get("experiment_id", "")).strip() or None,
        dataset_summary=_dataset_summary_for_manifest(config),
        manifest_entries=manifest_entries if write_manifest else None,
    )
    return rows
