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
from deepdetector.data.imagenet import resize_normalized_image
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


def _class_folder_rows(
    images_dir: Path,
    class_indices: dict[str, Any],
    class_order: list[str] | None = None,
    class_quotas: dict[str, Any] | None = None,
) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    if class_order is None:
        ordered_classes = sorted(str(class_name) for class_name in class_indices)
    else:
        ordered_classes = [str(class_name) for class_name in class_order]

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
            raise ValueError("Missing Table 10 ImageNet class directory: {0}".format(class_dir))
        class_rows = [
            (path, int(label_index))
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
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
        if expected_shape is None:
            expected_shape = processed.shape
        elif processed.shape != expected_shape:
            raise ValueError(
                "Table 10 GoogLeNet preprocessing returned inconsistent shapes: {0} and {1}".format(
                    expected_shape,
                    processed.shape,
                )
            )
        images.append(processed)
        labels.append(int(label_index))

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


def _row_result(group_config: dict[str, Any], row_config: dict[str, Any]) -> dict[str, Any]:
    metrics = row_config.get("metrics")
    if isinstance(metrics, dict):
        return {"metrics": metrics}
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
        outputs["manifest"] = write_metrics_json(
            output_path / "manifest.json",
            {
                "table": 10,
                "dataset_group": dataset_group,
                "model_group": model_group,
                "rows": manifest_entries,
            },
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

    dataset_group = str(config.get("dataset", {}).get("name", "")).strip()
    if not dataset_group:
        raise ValueError("Table 10 group must define dataset.name.")

    rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    write_manifest = model_group in {"caffenet", "inception_v3"}
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

    save_table_10_outputs(
        rows=rows,
        output_dir=_output_dir(config),
        dataset_group=dataset_group,
        model_group=model_group,
        manifest_entries=manifest_entries if write_manifest else None,
    )
    return rows
