"""Generate FGSM adversarial examples for ImageNet Table 7 using Caffe gradients."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepdetector.attacks.fgsm_imagenet import (  # noqa: E402
    generate_fgsm_imagenet,
    predict_caffe_label,
    preprocess_caffe_inputs,
)
from deepdetector.data.imagenet import resize_normalized_image  # noqa: E402
from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper  # noqa: E402


logger = logging.getLogger(__name__)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "article_reproduction" / "imagenet_table_7.yaml"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "adversarial_examples"
    / "imagenet"
    / "googlenet"
    / "fgsm"
)
IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the adversarial .npy array.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config, model, and images, then stop before generation.",
    )
    return parser


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve a config path relative to the project root."""
    if path_value in (None, ""):
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: Path) -> Dict[str, Any]:
    """Load the experiment YAML config."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")
    return config


def build_model(config: Dict[str, Any]) -> GoogLeNetCaffeWrapper:
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


def _class_image_rows(class_configs: Sequence[Dict[str, Any]]) -> List[Tuple[Path, int]]:
    """Return sorted image paths and labels for configured class directories."""
    rows: List[Tuple[Path, int]] = []
    for class_config in class_configs:
        class_dir = _resolve_path(class_config.get("path"))
        if class_dir is None or not class_dir.is_dir():
            raise IOError("Missing ImageNet class directory: {0}".format(class_dir))

        label = int(class_config["label"])
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((path, label))
    return rows


def load_subset_images(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load configured ImageNet class folders as normalized NHWC images."""
    dataset_config = config.get("dataset", {})
    class_configs = dataset_config.get("classes", [])
    if not class_configs:
        raise ValueError("Config must define dataset.classes.")

    rows = _class_image_rows(class_configs)
    if bool(dataset_config.get("shuffle", False)):
        seed = int(config.get("experiment", {}).get("seed", 20170830))
        rng = np.random.RandomState(seed)
        rows = [rows[int(index)] for index in rng.permutation(len(rows))]

    configured_n_samples = dataset_config.get("n_samples")
    if configured_n_samples not in (None, "", "all"):
        rows = rows[: int(configured_n_samples)]

    image_size = int(dataset_config.get("image_size", 224))
    images = []
    labels = []
    for path, label in rows:
        image = _read_rgb_image(path)
        images.append(resize_normalized_image(image, image_size=image_size))
        labels.append(label)

    if not images:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _epsilon_normalized(config: Dict[str, Any]) -> float:
    """Return FGSM epsilon in normalized [0, 1] image scale."""
    attack_config = config.get("attack", {})
    if "epsilon_255" in attack_config:
        return float(attack_config["epsilon_255"]) / 255.0
    return float(attack_config.get("epsilon", attack_config.get("eps", 1.0 / 255.0)))


def _article_model_inputs(model: GoogLeNetCaffeWrapper, images: np.ndarray) -> np.ndarray:
    """Return images in the Caffe input space used by the source article."""
    return preprocess_caffe_inputs(model, images)


def _epsilon_255(config: Dict[str, Any]) -> float:
    """Return FGSM epsilon in 0-255 image scale for naming."""
    attack_config = config.get("attack", {})
    if "epsilon_255" in attack_config:
        return float(attack_config["epsilon_255"])
    return float(_epsilon_normalized(config) * 255.0)


def _format_eps_255(value: float) -> str:
    """Format epsilon_255 for path naming."""
    if float(value).is_integer():
        return "eps_{0}_255".format(int(value))
    formatted = "{0:.5f}".format(float(value)).rstrip("0").rstrip(".")
    return "eps_{0}_255".format(formatted.replace(".", "p"))


def default_output_path(config: Dict[str, Any]) -> Path:
    """Return a default output path for the adversarial array."""
    eps_label = _format_eps_255(_epsilon_255(config))
    return DEFAULT_OUTPUT_DIR / eps_label / "adversarial_examples.npy"


def _predict_label(model: GoogLeNetCaffeWrapper, image: np.ndarray) -> int:
    """Predict the top-1 label for a single image."""
    return predict_caffe_label(model, image)


def main() -> int:
    """Generate and save FGSM adversarial examples for Table 7."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    config = load_config(_resolve_path(args.config) or DEFAULT_CONFIG)

    model = build_model(config)
    images, labels = load_subset_images(config)
    print("images_shape={0}".format(images.shape))
    print("labels_shape={0}".format(labels.shape))

    if args.dry_run:
        return 0
    if len(images) == 0:
        raise ValueError("No images loaded from the configured dataset.")

    keep_indices = []
    for index, image in enumerate(images):
        clean_pred = _predict_label(model, image)
        if clean_pred == int(labels[index]):
            keep_indices.append(index)
    print("total_images={0}".format(len(images)))
    print("clean_correct={0}".format(len(keep_indices)))
    print("skipped_wrong_baseline={0}".format(len(images) - len(keep_indices)))
    images = images[np.asarray(keep_indices, dtype=np.int64)]
    labels = labels[np.asarray(keep_indices, dtype=np.int64)]
    if len(images) == 0:
        raise ValueError("No clean-correct images loaded from the configured dataset.")

    images = _article_model_inputs(model, images)
    output_path = _resolve_path(args.output) or default_output_path(config)
    if output_path.exists() and not args.overwrite:
        raise IOError("Output already exists: {0}".format(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = generate_fgsm_imagenet(
        model=model,
        images=images,
        labels=None,
        epsilon_255=_epsilon_255(config),
        skip_wrong_baseline=False,
        clip_min=0.0,
        clip_max=255.0,
    )
    adv_images = result.adversarial_images

    np.save(str(output_path), adv_images.astype(np.float32))
    print("saved={0}".format(output_path))
    print("adv_shape={0}".format(adv_images.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
