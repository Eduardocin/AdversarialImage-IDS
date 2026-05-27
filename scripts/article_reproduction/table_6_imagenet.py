"""Reproduce ImageNet Table 6 adaptive quantization metrics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import yaml


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "article_reproduction" / "imagenet_table_6.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "imagenet" / "article_reproduction"
IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")

from deepdetector.data.imagenet import resize_normalized_image  # noqa: E402
from deepdetector.evaluation.table4_imagenet import Table4Sample  # noqa: E402
from deepdetector.evaluation.table6_imagenet import (  # noqa: E402
    Table6Evaluation,
    evaluate_table6_imagenet,
    validate_table6_result,
    write_table6_outputs,
)
from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper  # noqa: E402


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of images to evaluate per split.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="FGSM epsilon in 0-255 Caffe scale. Defaults to config or 1.0.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for table_6_imagenet.csv and diagnostics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the config, model, and images, then stop before inference.",
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


def output_dir_from_config(config: Dict[str, Any], override: Optional[str]) -> Path:
    """Return the configured output directory."""
    output_config = config.get("output", {})
    output_dir = _resolve_path(override or output_config.get("results_dir"))
    return output_dir or DEFAULT_OUTPUT_DIR


def write_status(output_dir: Path, status: str, **fields: Any) -> Path:
    """Write a status JSON file for partial ImageNet runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": status}
    payload.update(fields)
    path = output_dir / "table_6_status.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


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


def load_split_samples(
    config: Dict[str, Any],
    limit_override: Optional[int] = None,
) -> Dict[str, List[Table4Sample]]:
    """Load configured ImageNet splits from local class folders."""
    dataset_config = config.get("dataset", {})
    split_configs = dataset_config.get("splits", {})
    if not isinstance(split_configs, dict) or not split_configs:
        raise ValueError("Config must define dataset.splits.")

    image_size = int(dataset_config.get("image_size", 224))
    samples_by_split: Dict[str, List[Table4Sample]] = {}
    for split_name, class_configs in split_configs.items():
        if not class_configs:
            raise ValueError("Configured split has no classes: {0}".format(split_name))
        split_samples: List[Table4Sample] = []
        for class_config in class_configs:
            split_samples.extend(_class_samples(class_config, image_size=image_size))

        if bool(dataset_config.get("shuffle", False)):
            seed = int(config.get("experiment", {}).get("seed", 20170830))
            rng = np.random.RandomState(seed)
            split_samples = [split_samples[int(index)] for index in rng.permutation(len(split_samples))]

        configured_limit = limit_override if limit_override is not None else dataset_config.get("n_samples")
        if configured_limit not in (None, "", "all"):
            split_samples = split_samples[: int(configured_limit)]
        if not split_samples:
            raise ValueError("Configured split has no images: {0}".format(split_name))

        samples_by_split[str(split_name)] = split_samples

    return samples_by_split


def epsilon_255_from_config(config: Dict[str, Any], override: Optional[float]) -> float:
    """Return FGSM epsilon in 0-255 Caffe scale."""
    if override is not None:
        return float(override)
    attack_config = config.get("attack", {})
    if "epsilon_255" in attack_config:
        return float(attack_config["epsilon_255"])
    if "epsilon" in attack_config:
        return float(attack_config["epsilon"]) * 255.0
    return 1.0


def _print_split_summary(result: Table6Evaluation) -> None:
    """Print the required per-split experiment summary."""
    for summary in result.summaries:
        print("Split: {0}".format(summary.split))
        print("Total images: {0}".format(summary.total_images))
        print("Clean correct: {0}".format(summary.clean_correct))
        print("Skipped wrong baseline: {0}".format(summary.skipped_wrong_baseline))
        print("FGSM success: {0}".format(summary.fgsm_success))
        print("Disturbed failure: {0}".format(summary.disturbed_failure))
        print("TP: {0}".format(summary.tp))
        print("FN: {0}".format(summary.fn))
        print("FP: {0}".format(summary.fp))
        print("Recall: {0}".format(summary.recall))
        print("Precision: {0}".format(summary.precision))
        print("F1: {0}".format(summary.f1))


def main() -> int:
    """Run Table 6 ImageNet and write CSV outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    config = load_config(_resolve_path(args.config) or DEFAULT_CONFIG)
    output_dir = output_dir_from_config(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = build_model(config)
    except ImportError as exc:
        logger.warning("%s", exc)
        write_status(output_dir, "bloqueado_caffe", limitation="caffe_indisponivel", message=str(exc))
        return 0
    except OSError as exc:
        logger.warning("%s", exc)
        write_status(
            output_dir,
            "bloqueado_modelo_googlenet",
            limitation="assets_googlenet_indisponiveis",
            message=str(exc),
            setup_command="python scripts/imagenet/download_caffe_imagenet_assets.py --model googlenet",
        )
        return 0

    try:
        samples_by_split = load_split_samples(config, limit_override=args.limit)
    except (IOError, ValueError) as exc:
        write_status(output_dir, "falhou_dataset", message=str(exc))
        print(str(exc), file=sys.stderr)
        return 1

    print(
        "n_images={0}".format(
            sum(len(samples) for samples in samples_by_split.values())
        )
    )

    if args.dry_run:
        write_status(
            output_dir,
            "parcial",
            limitation="dry_run",
            n_loaded=sum(len(samples) for samples in samples_by_split.values()),
        )
        return 0

    attack_config = config.get("attack", {})
    result = evaluate_table6_imagenet(
        model=model,
        samples_by_split=samples_by_split,
        epsilon_255=epsilon_255_from_config(config, args.epsilon),
        clip_min=float(attack_config.get("clip_min", 0.0)),
        clip_max=float(attack_config.get("clip_max", 255.0)),
    )
    _print_split_summary(result)

    output_config = config.get("output", {})
    csv_path, diagnostics_path = write_table6_outputs(
        output_dir=output_dir,
        result=result,
        csv_name=str(output_config.get("csv", "table_6_imagenet.csv")),
        diagnostics_name=str(
            output_config.get("diagnostics_csv", "table_6_imagenet_diagnostics.csv")
        ),
    )

    try:
        validate_table6_result(result)
    except RuntimeError as exc:
        write_status(
            output_dir,
            "falhou_validacao",
            n_loaded=result.n_clean_total,
            n_clean_correct=result.n_clean_correct,
            n_attack_success=result.n_attack_success,
            results_csv=str(csv_path),
            diagnostics_csv=str(diagnostics_path),
            message=str(exc),
        )
        print(str(exc), file=sys.stderr)
        return 1

    write_status(
        output_dir,
        "completo",
        n_loaded=result.n_clean_total,
        n_clean_correct=result.n_clean_correct,
        n_attack_success=result.n_attack_success,
        results_csv=str(csv_path),
        diagnostics_csv=str(diagnostics_path),
    )
    print("results_csv={0}".format(csv_path))
    print("diagnostics_csv={0}".format(diagnostics_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
