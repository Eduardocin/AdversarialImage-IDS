"""Official combined Table 6 runner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from deepdetector.data.imagenet import resize_normalized_image
from deepdetector.evaluation.article_reproduction import (
    close_graph,
    create_restored_mnist_graph,
    evaluate_filter_on_existing_adversarial,
)
from deepdetector.evaluation.table4_imagenet import Table4Sample
from deepdetector.evaluation.table6_imagenet import (
    DEFAULT_SPLIT_ORDER,
    TABLE6_OUTPUT_FIELDS,
    Table6Evaluation,
    evaluate_table6_imagenet,
    validate_table6_result,
)
from deepdetector.experiments.adversarial_examples import (
    DEFAULT_ADVERSARIAL_CACHE_DIR,
    prepare_mnist_fgsm_adversarial_set,
)
from deepdetector.filters.factory import build_filter_from_config
from deepdetector.io.paths import ensure_dir, resolve_project_path
from deepdetector.io.result_writers import write_metrics_csv, write_metrics_json
from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper
from deepdetector.paths import MNIST_M1_CHECKPOINT_DIR


logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve a project-relative path."""
    return resolve_project_path(path_value)


def _output_dir(config: Dict[str, Any]) -> Path:
    """Return the configured output directory."""
    output_dir = _resolve_path(config.get("output", {}).get("dir"))
    if output_dir is None:
        raise ValueError("Table 6 must define output.dir.")
    return output_dir


def _checkpoint_dir(config: Dict[str, Any]) -> str:
    """Return the configured MNIST checkpoint directory."""
    checkpoint_dir = _resolve_path(config.get("model", {}).get("checkpoint_dir"))
    return str(checkpoint_dir or MNIST_M1_CHECKPOINT_DIR)


def _cache_enabled(config: Dict[str, Any]) -> bool:
    """Return whether adversarial example cache is enabled."""
    return bool(config.get("attack", {}).get("cache", True))


def _cache_root(config: Dict[str, Any]) -> Path:
    """Return the configured adversarial cache root."""
    attack_config = config.get("attack", {})
    cache_dir = attack_config.get("cache_dir") or DEFAULT_ADVERSARIAL_CACHE_DIR
    resolved = _resolve_path(cache_dir)
    return resolved or Path(str(cache_dir))


def imagenet_fgsm_cache_path(config: Dict[str, Any], split: str) -> Path:
    """Return the default ImageNet FGSM cache path for one split."""
    attack_config = config.get("attack", {})
    cache_paths = attack_config.get("cache_paths", {})
    if isinstance(cache_paths, dict) and split in cache_paths:
        resolved = _resolve_path(cache_paths[split])
        return resolved or Path(str(cache_paths[split]))
    return _cache_root(config) / "imagenet" / "fgsm" / split / "adversarial_examples.npy"


def _read_imagenet_fgsm_cache(cache_path: Path) -> Optional[np.ndarray]:
    """Load cached ImageNet FGSM examples when the archive is readable."""
    if not cache_path.is_file():
        return None
    try:
        adversarial_images = np.asarray(np.load(str(cache_path)), dtype=np.float32)
    except (OSError, ValueError):
        return None
    if adversarial_images.ndim != 4:
        return None
    return adversarial_images


def _write_imagenet_fgsm_cache(cache_path: Path, adversarial_images: np.ndarray) -> None:
    """Persist ImageNet FGSM examples for one split."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), np.asarray(adversarial_images, dtype=np.float32))


def _load_imagenet_fgsm_caches(
    config: Dict[str, Any],
    split_order: Sequence[str],
    table_label: str = "Table 6",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Path]]:
    """Return cached ImageNet FGSM examples by split."""
    if not _cache_enabled(config):
        return {}, {}

    cached_by_split: Dict[str, np.ndarray] = {}
    paths_by_split: Dict[str, Path] = {}
    for split in split_order:
        split_key = _split_key(split)
        cache_path = imagenet_fgsm_cache_path(config, split_key)
        paths_by_split[split_key] = cache_path
        cached = _read_imagenet_fgsm_cache(cache_path)
        if cached is not None:
            logger.info(
                "Loaded %s ImageNet FGSM cache for split %s: %s (%d examples)",
                table_label,
                split_key,
                cache_path,
                len(cached),
            )
            cached_by_split[split_key] = cached
        else:
            logger.info(
                "%s ImageNet FGSM cache miss for split %s: %s; FGSM will be generated",
                table_label,
                split_key,
                cache_path,
            )
    return cached_by_split, paths_by_split


def _write_missing_imagenet_fgsm_caches(
    result: Table6Evaluation,
    cached_by_split: Mapping[str, np.ndarray],
    paths_by_split: Mapping[str, Path],
    table_label: str = "Table 6",
) -> None:
    """Write ImageNet FGSM caches that were not loaded before evaluation."""
    for split, adversarial_images in result.adversarial_by_split.items():
        if split in cached_by_split or split not in paths_by_split:
            continue
        _write_imagenet_fgsm_cache(paths_by_split[split], adversarial_images)
        logger.info(
            "Wrote %s ImageNet FGSM cache for split %s: %s (%d examples)",
            table_label,
            split,
            paths_by_split[split],
            len(adversarial_images),
        )


def _split_key(split_name: Any) -> str:
    """Normalize split names for aggregation."""
    text = str(split_name).strip().lower()
    if text == "training":
        return "train"
    if text == "val":
        return "validation"
    return text


def _configured_slices(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield configured MNIST Table 6 slices."""
    slices = config.get("dataset", {}).get("slices", [])
    if not slices:
        raise ValueError("Table 6 MNIST config must define dataset.slices.")
    for split_config in slices:
        yield split_config


def _mnist_split_config(config: Dict[str, Any], split_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a single-split MNIST materialization config."""
    dataset_config = dict(config.get("dataset", {}))
    dataset_config.pop("slices", None)
    dataset_config.update(
        {
            "start": split_config["start"],
            "end": split_config["end"],
            "slice_name": _split_key(split_config["name"]),
        }
    )
    materialization_config = dict(config)
    materialization_config["dataset"] = dataset_config
    return materialization_config


def evaluate_mnist_table6(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate Table 6 adaptive quantization on MNIST splits."""
    filter_name, filter_fn, _ = build_filter_from_config(config.get("filter", {}))
    if filter_name != "adaptive_quantization":
        raise ValueError("Table 6 MNIST must use adaptive_quantization.")

    evaluation_config = config.get("evaluation", {})
    rows: List[Dict[str, Any]] = []
    graph = create_restored_mnist_graph(_checkpoint_dir(config))
    try:
        for split_config in _configured_slices(config):
            split_name = _split_key(split_config["name"])
            adversarial_set = prepare_mnist_fgsm_adversarial_set(
                _mnist_split_config(config, split_config),
                graph=graph,
            )
            metrics = evaluate_filter_on_existing_adversarial(
                graph=adversarial_set.graph,
                images=adversarial_set.images,
                labels=adversarial_set.labels,
                adv_images=adversarial_set.adversarial_images,
                clean_pred=adversarial_set.clean_predictions,
                adv_pred=adversarial_set.adversarial_predictions,
                filter_fn=filter_fn,
                batch_size=int(evaluation_config.get("batch_size", 256)),
                exclude_invalid_pairs=bool(
                    evaluation_config.get("exclude_invalid_pairs", False)
                ),
            )
            row = {"split": split_name}
            row.update(metrics)
            rows.append(row)
    finally:
        close_graph(graph)
    return rows


def build_imagenet_model(config: Dict[str, Any]) -> GoogLeNetCaffeWrapper:
    """Instantiate the configured GoogLeNet Caffe wrapper."""
    model_config = config.get("model", {})
    return GoogLeNetCaffeWrapper(
        model_dir=str(_resolve_path(model_config.get("model_dir"))),
        deploy_prototxt=str(_resolve_path(model_config.get("deploy_proto"))),
        caffemodel=str(_resolve_path(model_config.get("caffemodel"))),
        mean_file=(
            str(_resolve_path(model_config.get("mean_file")))
            if model_config.get("mean_file")
            else None
        ),
        use_gpu=bool(model_config.get("use_gpu", False)),
        batch_size=int(model_config.get("batch_size", 32)),
    )


def _read_rgb_image(path: Path) -> np.ndarray:
    """Load one image as normalized RGB float32 data."""
    from PIL import Image

    with Image.open(str(path)) as image:
        rgb_image = image.convert("RGB")
        return (np.asarray(rgb_image, dtype=np.float32) / 255.0).astype(np.float32)


def _class_samples(class_config: Mapping[str, Any], image_size: int) -> List[Table4Sample]:
    """Load all supported images for one configured ImageNet class directory."""
    class_dir = _resolve_path(class_config.get("path"))
    if class_dir is None or not class_dir.is_dir():
        raise IOError("Missing ImageNet class directory: {0}".format(class_dir))

    class_name = str(class_config["name"])
    true_label = int(class_config["label"])
    samples = []
    for path in sorted(class_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = _read_rgb_image(path)
        samples.append(
            Table4Sample(
                image=resize_normalized_image(image, image_size=image_size),
                true_label=true_label,
                image_id=path.stem,
                class_name=class_name,
            )
        )
    return samples


def load_imagenet_split_samples(config: Dict[str, Any]) -> Dict[str, List[Table4Sample]]:
    """Load configured ImageNet Table 6 splits from local class folders."""
    dataset_config = config.get("dataset", {})
    split_configs = dataset_config.get("splits", {})
    if not isinstance(split_configs, dict) or not split_configs:
        raise ValueError("Table 6 ImageNet config must define dataset.splits.")

    image_size = int(dataset_config.get("image_size", 224))
    samples_by_split: Dict[str, List[Table4Sample]] = {}
    for split_name, class_configs in split_configs.items():
        if not class_configs:
            raise ValueError("Configured ImageNet split has no classes: {0}".format(split_name))
        split_samples: List[Table4Sample] = []
        for class_config in class_configs:
            split_samples.extend(_class_samples(class_config, image_size=image_size))

        if bool(dataset_config.get("shuffle", False)):
            seed = int(config.get("experiment", {}).get("seed", 20170830))
            rng = np.random.RandomState(seed)
            split_samples = [
                split_samples[int(index)] for index in rng.permutation(len(split_samples))
            ]

        configured_limit = dataset_config.get("n_samples")
        if configured_limit not in (None, "", "all"):
            split_samples = split_samples[: int(configured_limit)]
        if not split_samples:
            raise ValueError("Configured ImageNet split has no images: {0}".format(split_name))

        samples_by_split[_split_key(split_name)] = split_samples
    return samples_by_split


def _epsilon_255(config: Dict[str, Any]) -> float:
    """Return FGSM epsilon in 0-255 Caffe scale."""
    attack_config = config.get("attack", {})
    if "epsilon_255" in attack_config:
        return float(attack_config["epsilon_255"])
    if "epsilon" in attack_config:
        return float(attack_config["epsilon"]) * 255.0
    return 1.0


def evaluate_imagenet_table6(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate Table 6 adaptive quantization on ImageNet splits."""
    model = build_imagenet_model(config)
    samples_by_split = load_imagenet_split_samples(config)
    attack_config = config.get("attack", {})
    split_order = tuple(
        _split_key(split)
        for split in config.get("split_order", DEFAULT_SPLIT_ORDER)
    )
    cached_by_split, paths_by_split = _load_imagenet_fgsm_caches(config, split_order)
    result = evaluate_table6_imagenet(
        model=model,
        samples_by_split=samples_by_split,
        epsilon_255=_epsilon_255(config),
        clip_min=float(attack_config.get("clip_min", 0.0)),
        clip_max=float(attack_config.get("clip_max", 255.0)),
        split_order=split_order,
        adversarial_by_split=cached_by_split,
    )
    _write_missing_imagenet_fgsm_caches(result, cached_by_split, paths_by_split)
    validate_table6_result(result)
    return result.rows


def _counts(row: Dict[str, Any]) -> Tuple[int, int, int]:
    """Return TP, FN, and FP from one metrics row."""
    return int(row.get("TP", 0)), int(row.get("FN", 0)), int(row.get("FP", 0))


def _metric_row(split: str, tp: int, fn: int, fp: int) -> Dict[str, Any]:
    """Return one aggregated Table 6 metrics row."""
    recall = tp / float(tp + fn) if tp + fn else 0.0
    precision = tp / float(tp + fp) if tp + fp else 0.0
    f1 = 2.0 * recall * precision / float(recall + precision) if recall + precision else 0.0
    return {
        "split": split,
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "recall_percent": float(recall * 100.0),
        "precision_percent": float(precision * 100.0),
        "f1_percent": float(f1 * 100.0),
    }


def aggregate_table6_rows(
    mnist_rows: Sequence[Dict[str, Any]],
    imagenet_rows: Sequence[Dict[str, Any]],
    split_order: Sequence[str] = DEFAULT_SPLIT_ORDER,
) -> List[Dict[str, Any]]:
    """Aggregate MNIST and ImageNet Table 6 counters by split."""
    rows_by_split: Dict[str, Dict[str, int]] = {}
    for row in list(mnist_rows) + list(imagenet_rows):
        split = _split_key(row["split"])
        tp, fn, fp = _counts(row)
        current = rows_by_split.setdefault(split, {"TP": 0, "FN": 0, "FP": 0})
        current["TP"] += tp
        current["FN"] += fn
        current["FP"] += fp

    output_rows = []
    for split in split_order:
        split_key = _split_key(split)
        counts = rows_by_split.get(split_key, {"TP": 0, "FN": 0, "FP": 0})
        output_rows.append(
            _metric_row(
                split=split_key,
                tp=counts["TP"],
                fn=counts["FN"],
                fp=counts["FP"],
            )
        )
    return output_rows


def _json_payload(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return the compact Table 6 JSON payload."""
    payload: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        payload[str(row["split"])] = {
            "tp": int(row["TP"]),
            "fn": int(row["FN"]),
            "fp": int(row["FP"]),
            "recall_percent": float(row["recall_percent"]),
            "precision_percent": float(row["precision_percent"]),
            "f1_percent": float(row["f1_percent"]),
        }
    return payload


def run_table6_experiment(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run the official combined MNIST + ImageNet Table 6 experiment."""
    mnist_config = dict(config.get("mnist", {}))
    imagenet_config = dict(config.get("imagenet", {}))
    if not mnist_config or not imagenet_config:
        raise ValueError("Table 6 must define internal mnist and imagenet configs.")

    mnist_rows = evaluate_mnist_table6(mnist_config)
    imagenet_rows = evaluate_imagenet_table6(imagenet_config)
    rows = aggregate_table6_rows(
        mnist_rows=mnist_rows,
        imagenet_rows=imagenet_rows,
        split_order=config.get("split_order", DEFAULT_SPLIT_ORDER),
    )

    output_dir = ensure_dir(_output_dir(config))
    output_config = config.get("output", {})
    write_metrics_csv(
        output_dir / str(output_config.get("csv", "metrics.csv")),
        rows,
        TABLE6_OUTPUT_FIELDS,
    )
    write_metrics_json(
        output_dir / str(output_config.get("json", "metrics.json")),
        _json_payload(rows),
    )
    return rows
