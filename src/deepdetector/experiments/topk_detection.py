"""Official top-k prediction detection experiment runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from deepdetector.attacks.fgsm_imagenet import generate_fgsm_caffe_image
from deepdetector.data.imagenet import IMAGE_EXTENSIONS, read_rgb_image, resize_normalized_image
from deepdetector.evaluation.imagenet_utils import article_model_inputs
from deepdetector.evaluation.topk_detection import (
    TopKDetectionResult,
    evaluate_topk_detection,
    write_topk_detection_outputs,
)
from deepdetector.filters.factory import build_filter_from_config
from deepdetector.io.paths import resolve_project_path
from deepdetector.models.imagenet_wrappers import build_googlenet_caffe_model


def _output_dir(config: Dict[str, Any]) -> Path:
    """Return the configured output directory."""
    output_config = config.get("output", {})
    output_dir = resolve_project_path(output_config.get("dir"))
    if output_dir is None:
        raise ValueError("topk_detection must define output.dir.")
    return output_dir


def _ordered_class_names(class_indices: Dict[str, Any]) -> list[str]:
    """Return class names in configured order."""
    return [str(class_name) for class_name in class_indices]


def _class_image_paths(images_dir: Path, class_name: str) -> list[Path]:
    """Validate one class folder and return sorted image paths."""
    class_dir = images_dir / class_name
    if not class_dir.is_dir():
        raise ValueError("Missing ImageNet class directory: {0}".format(class_dir))

    image_paths: list[Path] = []
    for path in sorted(class_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Unsupported ImageNet image extension: {0}".format(path))
        image_paths.append(path)

    if not image_paths:
        raise ValueError("ImageNet class directory has no images: {0}".format(class_dir))
    return image_paths


def load_topk_imagenet_samples_by_class(config: Dict[str, Any]) -> dict[str, list[np.ndarray]]:
    """Load configured ImageNet test class folders as normalized images by class."""
    dataset_config = config.get("dataset", {})
    images_dir = resolve_project_path(dataset_config.get("images_dir"))
    if images_dir is None:
        raise ValueError("topk_detection dataset must define images_dir.")
    if not images_dir.is_dir():
        raise ValueError("Missing ImageNet directory: {0}".format(images_dir))

    class_indices = dict(dataset_config.get("class_indices", {}))
    if not class_indices:
        raise ValueError("topk_detection dataset must define class_indices.")

    image_size = int(dataset_config.get("image_size", 224))
    n_samples = dataset_config.get("n_samples", "all")
    samples_by_class: dict[str, list[np.ndarray]] = {}
    for class_name in _ordered_class_names(class_indices):
        image_paths = _class_image_paths(images_dir, class_name)
        if n_samples not in (None, "", "all"):
            image_paths = image_paths[: int(n_samples)]
        samples_by_class[class_name] = [
            resize_normalized_image(read_rgb_image(path), image_size=image_size)
            for path in image_paths
        ]
    return samples_by_class


def _to_article_inputs_by_class(
    model: Any,
    samples_by_class: dict[str, list[np.ndarray]],
) -> dict[str, list[np.ndarray]]:
    """Convert normalized images to the GoogLeNet/Caffe article input space."""
    converted: dict[str, list[np.ndarray]] = {}
    for class_name, samples in samples_by_class.items():
        if not samples:
            converted[class_name] = []
            continue
        batch = np.asarray(samples, dtype=np.float32)
        converted[class_name] = [
            np.asarray(image, dtype=np.float32)
            for image in article_model_inputs(model, batch)
        ]
    return converted


def _epsilon_255(attack_config: Dict[str, Any]) -> float:
    """Return FGSM epsilon in 0-255 units."""
    if "epsilon_255" in attack_config:
        return float(attack_config["epsilon_255"])
    return float(attack_config.get("epsilon", 1.0 / 255.0)) * 255.0


def _generate_fgsm_googlenet(
    model: Any,
    image: np.ndarray,
    class_id: int,
    attack_config: Dict[str, Any],
) -> np.ndarray:
    """Generate one FGSM/GoogLeNet adversarial image through the shared helper."""
    return generate_fgsm_caffe_image(
        model=model,
        image=image,
        class_id=int(class_id),
        epsilon_255=_epsilon_255(attack_config),
        clip_min=float(attack_config.get("clip_min", 0.0)),
        clip_max=float(attack_config.get("clip_max", 255.0)),
    )


def build_topk_googlenet_model(config: Dict[str, Any]) -> Any:
    """Instantiate the configured GoogLeNet/Caffe model."""
    return build_googlenet_caffe_model(config)


def run_topk_detection_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run top-k detection and write the official output artifacts."""
    model = build_topk_googlenet_model(config)
    samples = load_topk_imagenet_samples_by_class(config)
    samples = _to_article_inputs_by_class(model, samples)
    _, transform, _ = build_filter_from_config(dict(config.get("filter", {})))
    result: TopKDetectionResult = evaluate_topk_detection(
        samples_by_class=samples,
        model=model,
        transform=transform,
        attack_generator=_generate_fgsm_googlenet,
        config=config,
    )
    outputs = write_topk_detection_outputs(output_dir=_output_dir(config), result=result)
    return {name: str(path) for name, path in outputs.items()}
