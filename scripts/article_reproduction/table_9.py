"""Reproduce Table 9 with the final FGSM detector filter."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepdetector.attacks.fgsm_imagenet import (  # noqa: E402
    predict_caffe_label,
    preprocess_caffe_inputs,
)
from deepdetector.data.imagenet import resize_normalized_image  # noqa: E402
from deepdetector.evaluation.article_reproduction import (  # noqa: E402
    close_graph,
    create_restored_mnist_graph,
    evaluate_filter_on_images,
    load_mnist_test_slice,
)
from deepdetector.evaluation.table4_imagenet import (  # noqa: E402
    Table4Sample,
    generate_fgsm_from_gradient,
)
from deepdetector.filters.registry import FILTER_REGISTRY  # noqa: E402
from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper  # noqa: E402


logger = logging.getLogger(__name__)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "article_reproduction" / "table_9.yaml"
OUTPUT_FIELDS: Tuple[str, ...] = (
    "split",
    "TP",
    "FN",
    "FP",
    "recall_percent",
    "precision_percent",
    "f1_percent",
)
SPLIT_ORDER: Tuple[str, ...] = ("Training", "Validation")
EXPECTED_FLOWS: Tuple[str, ...] = ("mnist_m1_fgsm", "imagenet_googlenet_fgsm")
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpeg", ".jpg", ".png")

Counters = Dict[str, int]
SplitCounters = Dict[str, Counters]


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve a path relative to the project root."""
    if path_value in (None, ""):
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    """Return a stable project-relative path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def load_config(path: Path) -> Dict[str, Any]:
    """Load the Table 9 YAML config."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")
    return config


def output_dir_from_config(config: Mapping[str, Any], override: Optional[str]) -> Path:
    """Return the configured output directory."""
    output_config = config.get("output", {})
    if not isinstance(output_config, Mapping):
        raise ValueError("Config output must be a mapping.")
    output_dir = _resolve_path(override or output_config.get("results_dir"))
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "article_reproduction" / "table_9"
    return output_dir


def _output_name(config: Mapping[str, Any], key: str, default: str) -> str:
    """Return one configured output filename."""
    output_config = config.get("output", {})
    if not isinstance(output_config, Mapping):
        return default
    return str(output_config.get(key, default))


def _empty_counters() -> Counters:
    """Return empty detection counters."""
    return {"TP": 0, "FN": 0, "FP": 0}


def _empty_split_counters(splits: Sequence[str] = SPLIT_ORDER) -> SplitCounters:
    """Return empty counters for every output split."""
    return {str(split): _empty_counters() for split in splits}


def _add_counters(left: Counters, right: Mapping[str, Any]) -> None:
    """Add TP/FN/FP counters into ``left`` in place."""
    for field in ("TP", "FN", "FP"):
        left[field] = int(left.get(field, 0)) + int(right.get(field, 0))


def _metrics(tp: int, fn: int, fp: int) -> Tuple[float, float, float]:
    """Return recall, precision, and F1 percentages after counter aggregation."""
    recall = tp / float(tp + fn) if tp + fn else 0.0
    precision = tp / float(tp + fp) if tp + fp else 0.0
    f1 = 2.0 * recall * precision / float(recall + precision) if recall + precision else 0.0
    return round(recall * 100.0, 2), round(precision * 100.0, 2), round(f1 * 100.0, 2)


def aggregate_counters(per_flow_counters: Mapping[str, SplitCounters]) -> SplitCounters:
    """Aggregate TP/FN/FP by split before calculating derived metrics."""
    aggregate = _empty_split_counters()
    for flow_counters in per_flow_counters.values():
        for split in SPLIT_ORDER:
            _add_counters(aggregate[split], flow_counters.get(split, {}))
    return aggregate


def rows_from_counters(counters_by_split: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build final CSV rows from aggregated counters."""
    rows: List[Dict[str, Any]] = []
    for split in SPLIT_ORDER:
        counters = counters_by_split.get(split, {})
        tp = int(counters.get("TP", 0))
        fn = int(counters.get("FN", 0))
        fp = int(counters.get("FP", 0))
        recall, precision, f1 = _metrics(tp=tp, fn=fn, fp=fp)
        rows.append(
            {
                "split": split,
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "recall_percent": recall,
                "precision_percent": precision,
                "f1_percent": f1,
            }
        )
    return rows


def write_table9_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write the final Table 9 CSV with the exact spec schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = {}
            for field in OUTPUT_FIELDS:
                value = row.get(field, "")
                if field.endswith("_percent") and value != "":
                    value = "{0:.2f}".format(float(value))
                csv_row[field] = value
            writer.writerow(csv_row)
    return path


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    """Return a simple Markdown table."""
    lines = ["| {0} |".format(" | ".join(headers))]
    lines.append("| {0} |".format(" | ".join(["---"] * len(headers))))
    for row in rows:
        lines.append("| {0} |".format(" | ".join(str(value) for value in row)))
    return lines


def write_table9_markdown(
    path: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    status_payload: Mapping[str, Any],
    csv_path: Path,
) -> Path:
    """Write a short Table 9 reproduction report."""
    reference = config.get("reference", {})
    lines = [
        "# Table 9 Final FGSM Detector",
        "",
        str(config.get("experiment", {}).get("objective", "Table 9 reproduction.")),
        "",
        "## Local Result",
    ]
    lines.extend(
        _markdown_table(
            ["Split", "TP", "FN", "FP", "Recall", "Precision", "F1"],
            [
                [
                    row["split"],
                    row["TP"],
                    row["FN"],
                    row["FP"],
                    "{0:.2f}%".format(float(row["recall_percent"])),
                    "{0:.2f}%".format(float(row["precision_percent"])),
                    "{0:.2f}%".format(float(row["f1_percent"])),
                ]
                for row in rows
            ],
        )
    )
    lines.extend(["", "## Article Reference"])
    lines.extend(
        _markdown_table(
            ["Split", "TP", "FN", "FP", "Recall", "Precision", "F1"],
            [
                [
                    split,
                    reference.get(split, {}).get("TP", ""),
                    reference.get(split, {}).get("FN", ""),
                    reference.get(split, {}).get("FP", ""),
                    "{0:.2f}%".format(float(reference.get(split, {}).get("recall_percent", 0.0))),
                    "{0:.2f}%".format(
                        float(reference.get(split, {}).get("precision_percent", 0.0))
                    ),
                    "{0:.2f}%".format(float(reference.get(split, {}).get("f1_percent", 0.0))),
                ]
                for split in SPLIT_ORDER
            ],
        )
    )
    lines.extend(["", "## Execution Notes"])
    lines.append("- Status: {0}".format(status_payload.get("status", "")))
    for warning in status_payload.get("warnings", []):
        lines.append("- {0}".format(warning))
    if not status_payload.get("warnings"):
        lines.append("- No blocked flows reported.")
    lines.append("- CSV: {0}".format(_display_path(csv_path)))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_status(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write the Table 9 status JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the Table 9 config contract used by the script."""
    splits = list(config.get("splits", []))
    if splits != list(SPLIT_ORDER):
        raise ValueError("Table 9 config must define splits: Training, Validation.")

    flows = list(config.get("orchestration", {}).get("flows", []))
    if flows != list(EXPECTED_FLOWS):
        raise ValueError("Table 9 config must define the MNIST and ImageNet FGSM flows.")

    filter_name = str(config.get("detection", {}).get("filter_name", ""))
    if filter_name not in FILTER_REGISTRY:
        raise ValueError("Detection filter is not registered: {0}".format(filter_name))


def _flow_enabled(config: Mapping[str, Any], dataset_key: str) -> bool:
    """Return whether one dataset flow is enabled."""
    dataset_config = config.get("datasets", {}).get(dataset_key, {})
    if not isinstance(dataset_config, Mapping):
        return False
    return bool(dataset_config.get("enabled", True))


def _status_base(
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    sample_size: Optional[int],
) -> Dict[str, Any]:
    """Build the common status JSON payload."""
    csv_path = output_dir / _output_name(config, "csv", "table_9.csv")
    markdown_path = output_dir / _output_name(config, "markdown", "table_9.md")
    return {
        "status": "pending",
        "config_path": _display_path(config_path),
        "results_csv": _display_path(csv_path),
        "markdown": _display_path(markdown_path),
        "enabled_flows": list(config.get("orchestration", {}).get("flows", [])),
        "completed_flows": [],
        "skipped_flows": [],
        "sample_size": sample_size,
        "warnings": [],
    }


def _check_mnist_dry_run(config: Mapping[str, Any]) -> Tuple[str, str]:
    """Check MNIST dependencies without running the full detector."""
    if not _flow_enabled(config, "mnist"):
        return "skipped", "mnist_m1_fgsm disabled in config"

    mnist_config = config.get("datasets", {}).get("mnist", {})
    checkpoint_dir = _resolve_path(mnist_config.get("checkpoint_dir"))
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return "blocked", "MNIST checkpoint directory is missing: {0}".format(checkpoint_dir)

    try:
        import cleverhans  # noqa: F401
        import tensorflow  # noqa: F401
    except ImportError as exc:
        return "blocked", "MNIST dependencies are unavailable: {0}".format(exc)

    try:
        slices = mnist_config.get("slices", {})
        training_slice = slices.get("Training", {})
        start = int(training_slice.get("start", 0))
        sample_size = int(config.get("evaluation", {}).get("dry_run_sample_size", 4))
        load_mnist_test_slice(start, start + sample_size)
    except Exception as exc:  # pragma: no cover - environment-dependent data availability
        return "partial", "MNIST sample load was not available: {0}".format(exc)

    return "available", "MNIST M1 FGSM prerequisites are available"


def _imagenet_asset_paths(config: Mapping[str, Any]) -> List[Path]:
    """Return configured ImageNet model asset paths."""
    assets = config.get("datasets", {}).get("imagenet", {}).get("model_assets", {})
    paths = []
    for key in ("deploy_proto", "caffemodel"):
        path = _resolve_path(assets.get(key))
        if path is not None:
            paths.append(path)
    return paths


def _check_imagenet_dry_run(config: Mapping[str, Any]) -> Tuple[str, str]:
    """Check ImageNet/Caffe prerequisites without requiring success."""
    if not _flow_enabled(config, "imagenet"):
        return "skipped", "imagenet_googlenet_fgsm disabled in config"

    missing_assets = [path for path in _imagenet_asset_paths(config) if not path.is_file()]
    if missing_assets:
        return "blocked", "blocked_imagenet_caffe: missing GoogLeNet asset(s): {0}".format(
            ", ".join(_display_path(path) for path in missing_assets)
        )

    try:
        build_imagenet_model(config)
    except ImportError as exc:
        return "blocked", "blocked_imagenet_caffe: {0}".format(exc)
    except OSError as exc:
        return "blocked", "blocked_imagenet_caffe: {0}".format(exc)

    return "available", "ImageNet GoogLeNet/Caffe prerequisites are available"


def run_dry_run(config: Mapping[str, Any], config_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Validate configuration and write dry-run status."""
    validate_config(config)
    payload = _status_base(config, config_path, output_dir, sample_size=None)

    checks = {
        "mnist_m1_fgsm": _check_mnist_dry_run(config),
        "imagenet_googlenet_fgsm": _check_imagenet_dry_run(config),
    }
    for flow_name, (state, message) in checks.items():
        if state == "available":
            payload["completed_flows"].append(flow_name)
        else:
            payload["skipped_flows"].append(flow_name)
            payload["warnings"].append(message)

    payload["status"] = "completed" if not payload["skipped_flows"] else "partial"
    if len(payload["skipped_flows"]) == len(payload["enabled_flows"]):
        payload["status"] = "blocked"

    status_path = output_dir / _output_name(config, "status_json", "status.json")
    write_status(status_path, payload)
    return payload


def _mnist_split_bounds(
    mnist_config: Mapping[str, Any],
    split: str,
    sample_size: Optional[int],
) -> Tuple[int, int]:
    """Return the configured MNIST slice bounds for one split."""
    split_config = mnist_config.get("slices", {}).get(split)
    if not isinstance(split_config, Mapping):
        raise ValueError("Missing MNIST slice config for split: {0}".format(split))
    start = int(split_config["start"])
    end = int(split_config["end"])
    if sample_size is not None:
        end = min(end, start + int(sample_size))
    return start, end


def run_mnist_flow(
    config: Mapping[str, Any],
    filter_fn: Any,
    sample_size: Optional[int],
) -> SplitCounters:
    """Run the MNIST M1 FGSM portion of Table 9."""
    mnist_config = config.get("datasets", {}).get("mnist", {})
    if not _flow_enabled(config, "mnist"):
        raise RuntimeError("mnist_m1_fgsm is disabled in config.")

    checkpoint_dir = _resolve_path(mnist_config.get("checkpoint_dir"))
    if checkpoint_dir is None:
        raise ValueError("MNIST checkpoint_dir is required.")

    attack_config = mnist_config.get("attack", {})
    evaluation_config = config.get("evaluation", {})
    exclude_invalid_pairs = bool(evaluation_config.get("exclude_clean_errors", True)) and bool(
        evaluation_config.get("exclude_failed_attacks", True)
    )
    counters_by_split = _empty_split_counters()
    graph = create_restored_mnist_graph(str(checkpoint_dir))

    try:
        for split in SPLIT_ORDER:
            start, end = _mnist_split_bounds(mnist_config, split, sample_size)
            images, labels = load_mnist_test_slice(start, end)
            logger.info("Table 9 MNIST %s evaluating %d samples", split, len(images))
            metrics = evaluate_filter_on_images(
                graph=graph,
                images=images,
                labels=labels,
                epsilon=float(attack_config.get("epsilon", 0.2)),
                filter_fn=filter_fn,
                clip_min=float(attack_config.get("clip_min", 0.0)),
                clip_max=float(attack_config.get("clip_max", 1.0)),
                exclude_invalid_pairs=exclude_invalid_pairs,
            )
            counters_by_split[split] = {
                "TP": int(metrics.get("TP", 0)),
                "FN": int(metrics.get("FN", 0)),
                "FP": int(metrics.get("FP", 0)),
            }
    finally:
        close_graph(graph)

    return counters_by_split


def build_imagenet_model(config: Mapping[str, Any]) -> GoogLeNetCaffeWrapper:
    """Instantiate the configured GoogLeNet Caffe wrapper."""
    assets = config.get("datasets", {}).get("imagenet", {}).get("model_assets", {})
    return GoogLeNetCaffeWrapper(
        model_dir=str(_resolve_path(assets.get("model_dir"))),
        deploy_prototxt=str(_resolve_path(assets.get("deploy_proto"))),
        caffemodel=str(_resolve_path(assets.get("caffemodel"))),
        mean_file=(str(_resolve_path(assets.get("mean_file"))) if assets.get("mean_file") else None),
        use_gpu=bool(assets.get("use_gpu", False)),
        batch_size=int(assets.get("batch_size", 32)),
    )


def _read_rgb_image(path: Path) -> np.ndarray:
    """Load one image as normalized RGB float32 data."""
    from PIL import Image

    with Image.open(str(path)) as image:
        rgb_image = image.convert("RGB")
        return (np.asarray(rgb_image, dtype=np.float32) / 255.0).astype(np.float32)


def _imagenet_class_samples(
    class_name: str,
    class_config: Mapping[str, Any],
    image_size: int,
    sample_size: Optional[int],
) -> List[Table4Sample]:
    """Load up to ``sample_size`` samples for one configured ImageNet class."""
    class_dir = _resolve_path(class_config.get("path"))
    if class_dir is None or not class_dir.is_dir():
        raise IOError("Missing ImageNet class directory: {0}".format(class_dir))

    image_paths = [
        path
        for path in sorted(class_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if sample_size is not None:
        image_paths = image_paths[: int(sample_size)]

    samples = []
    for image_path in image_paths:
        image = _read_rgb_image(image_path)
        samples.append(
            Table4Sample(
                image=resize_normalized_image(image, image_size=image_size),
                true_label=int(class_config["label_id"]),
                image_id=image_path.stem,
                class_name=str(class_name),
            )
        )
    return samples


def load_imagenet_split_samples(
    config: Mapping[str, Any],
    sample_size: Optional[int],
) -> Dict[str, List[Table4Sample]]:
    """Load configured ImageNet class folders for Training and Validation."""
    imagenet_config = config.get("datasets", {}).get("imagenet", {})
    image_size = int(imagenet_config.get("image_size", 224))
    classes_by_split = imagenet_config.get("classes", {})
    if not isinstance(classes_by_split, Mapping):
        raise ValueError("ImageNet classes must be grouped by split.")

    samples_by_split: Dict[str, List[Table4Sample]] = {}
    for split in SPLIT_ORDER:
        class_configs = classes_by_split.get(split, {})
        if not isinstance(class_configs, Mapping) or not class_configs:
            raise ValueError("Missing ImageNet classes for split: {0}".format(split))

        split_samples: List[Table4Sample] = []
        for class_name, class_config in class_configs.items():
            split_samples.extend(
                _imagenet_class_samples(
                    class_name=str(class_name),
                    class_config=class_config,
                    image_size=image_size,
                    sample_size=sample_size,
                )
            )
        samples_by_split[split] = split_samples
    return samples_by_split


def _evaluate_imagenet_split(
    model: Any,
    samples: Sequence[Table4Sample],
    filter_fn: Any,
    epsilon_255: float,
    clip_min: float,
    clip_max: float,
    exclude_clean_errors: bool,
    exclude_failed_attacks: bool,
) -> Counters:
    """Evaluate final detector counts for one ImageNet split."""
    tp = 0
    fn = 0
    fp = 0

    for sample_index, sample in enumerate(samples, start=1):
        clean_image = preprocess_caffe_inputs(model, sample.image)[0]
        clean_pred = predict_caffe_label(model, clean_image)
        clean_correct = clean_pred == int(sample.true_label)
        if exclude_clean_errors and not clean_correct:
            continue

        adversarial_image = generate_fgsm_from_gradient(
            model=model,
            image=clean_image,
            class_id=clean_pred,
            epsilon_255=epsilon_255,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        adv_pred = predict_caffe_label(model, adversarial_image)
        if exclude_failed_attacks and adv_pred == clean_pred:
            continue

        filtered_clean = filter_fn(clean_image)
        filtered_adv = filter_fn(adversarial_image)
        filtered_clean_pred = predict_caffe_label(model, filtered_clean)
        filtered_adv_pred = predict_caffe_label(model, filtered_adv)

        fp += int(filtered_clean_pred != clean_pred)
        if filtered_adv_pred != adv_pred:
            tp += 1
        else:
            fn += 1

        if sample_index == 1 or sample_index == len(samples) or sample_index % 20 == 0:
            logger.info(
                "Table 9 ImageNet progress %d/%d | tp=%d fn=%d fp=%d",
                sample_index,
                len(samples),
                tp,
                fn,
                fp,
            )

    return {"TP": int(tp), "FN": int(fn), "FP": int(fp)}


def run_imagenet_flow(
    config: Mapping[str, Any],
    filter_fn: Any,
    sample_size: Optional[int],
) -> SplitCounters:
    """Run the ImageNet GoogLeNet FGSM portion of Table 9."""
    if not _flow_enabled(config, "imagenet"):
        raise RuntimeError("imagenet_googlenet_fgsm is disabled in config.")

    imagenet_config = config.get("datasets", {}).get("imagenet", {})
    attack_config = imagenet_config.get("attack", {})
    evaluation_config = config.get("evaluation", {})
    model = build_imagenet_model(config)
    samples_by_split = load_imagenet_split_samples(config, sample_size=sample_size)

    counters_by_split = _empty_split_counters()
    for split in SPLIT_ORDER:
        samples = samples_by_split.get(split, [])
        if not samples:
            raise ValueError("Configured ImageNet split has no images: {0}".format(split))
        logger.info("Table 9 ImageNet %s evaluating %d samples", split, len(samples))
        counters_by_split[split] = _evaluate_imagenet_split(
            model=model,
            samples=samples,
            filter_fn=filter_fn,
            epsilon_255=float(attack_config.get("epsilon_255", 1.0)),
            clip_min=float(attack_config.get("clip_min", 0.0)),
            clip_max=float(attack_config.get("clip_max", 255.0)),
            exclude_clean_errors=bool(evaluation_config.get("exclude_clean_errors", True)),
            exclude_failed_attacks=bool(evaluation_config.get("exclude_failed_attacks", True)),
        )
    return counters_by_split


def _flow_failure_message(flow_name: str, exc: BaseException) -> str:
    """Return a concise warning for a skipped flow."""
    return "{0} skipped: {1}".format(flow_name, exc)


def run_table9(
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    sample_size: Optional[int],
) -> Dict[str, Any]:
    """Run configured Table 9 flows and write all outputs."""
    validate_config(config)
    filter_name = str(config.get("detection", {}).get("filter_name", "article_final"))
    filter_fn = FILTER_REGISTRY[filter_name]

    payload = _status_base(config, config_path, output_dir, sample_size=sample_size)
    per_flow_counters: Dict[str, SplitCounters] = {}

    flow_runners = {
        "mnist_m1_fgsm": lambda: run_mnist_flow(config, filter_fn, sample_size),
        "imagenet_googlenet_fgsm": lambda: run_imagenet_flow(config, filter_fn, sample_size),
    }
    for flow_name in config.get("orchestration", {}).get("flows", []):
        try:
            counters = flow_runners[str(flow_name)]()
        except (ImportError, IOError, OSError, RuntimeError, ValueError) as exc:
            payload["skipped_flows"].append(str(flow_name))
            payload["warnings"].append(_flow_failure_message(str(flow_name), exc))
            continue

        payload["completed_flows"].append(str(flow_name))
        per_flow_counters[str(flow_name)] = counters

    aggregate = aggregate_counters(per_flow_counters)
    rows = rows_from_counters(aggregate)
    csv_path = output_dir / _output_name(config, "csv", "table_9.csv")
    markdown_path = output_dir / _output_name(config, "markdown", "table_9.md")
    status_path = output_dir / _output_name(config, "status_json", "status.json")

    if payload["completed_flows"] and not payload["skipped_flows"]:
        payload["status"] = "completed"
    elif payload["completed_flows"]:
        payload["status"] = "partial"
    else:
        payload["status"] = "blocked"

    payload["aggregate_counters"] = aggregate
    payload["per_flow_counters"] = per_flow_counters
    write_table9_csv(csv_path, rows)
    payload["results_csv"] = _display_path(csv_path)
    payload["markdown"] = _display_path(markdown_path)
    write_table9_markdown(markdown_path, config, rows, payload, csv_path)
    write_status(status_path, payload)
    return payload


def main() -> int:
    """Run Table 9 dry-run or reproduction execution."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")

    config_path = _resolve_path(args.config) or DEFAULT_CONFIG
    config = load_config(config_path)
    output_dir = output_dir_from_config(config, args.output_dir)

    if args.dry_run:
        payload = run_dry_run(config, config_path=config_path, output_dir=output_dir)
    else:
        payload = run_table9(
            config,
            config_path=config_path,
            output_dir=output_dir,
            sample_size=args.sample_size,
        )

    print("status={0}".format(payload["status"]))
    print("status_json={0}".format(_display_path(output_dir / _output_name(config, "status_json", "status.json"))))
    if not args.dry_run:
        print("results_csv={0}".format(payload["results_csv"]))
        print("markdown={0}".format(payload["markdown"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
