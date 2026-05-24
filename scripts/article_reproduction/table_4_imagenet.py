"""Reproduce ImageNet Table 4 scalar quantization metrics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "article_reproduction" / "imagenet_table_4.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "imagenet" / "article_reproduction"
IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")

from deepdetector.data.imagenet import resize_normalized_image  # noqa: E402
from deepdetector.evaluation.table4_imagenet import (  # noqa: E402
    TABLE4_INTERVALS,
    Table4Sample,
    evaluate_table4_imagenet,
    validate_attack_success,
    write_table4_outputs,
)
from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper  # noqa: E402


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory containing ImageNet class folders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of images to evaluate for a quick run.",
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
        help="Directory for table_4_imagenet.csv and diagnostics.",
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


def configured_intervals(config: Dict[str, Any]) -> Iterable[int]:
    """Return scalar quantization interval counts from config."""
    intervals = config.get("quantization", {}).get("intervals", TABLE4_INTERVALS)
    if not intervals:
        raise ValueError("Config must define quantization.intervals.")
    for value in intervals:
        yield int(value)


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
    path = output_dir / "table_4_status.json"
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


def _class_image_rows(images_dir: Path, class_indices: Dict[str, Any]) -> List[Tuple[Path, str, int]]:
    """Return sorted image paths and labels for a class-folder ImageNet subset."""
    rows = []
    for class_name, label_index in sorted(class_indices.items()):
        class_dir = images_dir / class_name
        if not class_dir.is_dir():
            raise IOError("Missing ImageNet class directory: {0}".format(class_dir))
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((path, str(class_name), int(label_index)))
    return rows


def load_subset_samples(
    config: Dict[str, Any],
    data_root_override: Optional[str] = None,
    limit_override: Optional[int] = None,
) -> List[Table4Sample]:
    """Load the configured local ImageNet subset from class folders."""
    dataset_config = config.get("dataset", {})
    images_dir = _resolve_path(data_root_override or dataset_config.get("images_dir"))
    if images_dir is None:
        raise ValueError("Config must define dataset.images_dir.")

    class_indices = dataset_config.get("class_indices", {})
    if not isinstance(class_indices, dict) or not class_indices:
        raise ValueError("Config must define dataset.class_indices.")

    rows = _class_image_rows(images_dir, class_indices)
    if bool(dataset_config.get("shuffle", False)):
        rng = np.random.RandomState(config.get("experiment", {}).get("seed", 20170830))
        order = rng.permutation(len(rows))
        rows = [rows[int(index)] for index in order]

    configured_limit = limit_override if limit_override is not None else dataset_config.get("n_samples")
    if configured_limit in (None, "", "all"):
        n_samples = len(rows)
    else:
        n_samples = int(configured_limit)
    rows = rows[:n_samples]
    image_size = int(dataset_config.get("image_size", 224))

    samples = []
    for path, class_name, label_index in rows:
        image = _read_rgb_image(path)
        samples.append(
            Table4Sample(
                image=resize_normalized_image(image, image_size=image_size),
                true_label=label_index,
                image_id=path.stem,
                class_name=class_name,
            )
        )

    return samples


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


def main() -> int:
    """Run the interval comparison and write CSV and Markdown outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    config = load_config(_resolve_path(args.config) or DEFAULT_CONFIG)
    output_dir = output_dir_from_config(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = build_model(config)
    except ImportError as exc:
        logger.warning("%s", exc)
        write_status(
            output_dir,
            "bloqueado_caffe",
            limitation="caffe_indisponivel",
            message=str(exc),
        )
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

    samples = load_subset_samples(
        config,
        data_root_override=args.data_root,
        limit_override=args.limit,
    )
    print("n_images={0}".format(len(samples)))

    if args.dry_run:
        write_status(output_dir, "parcial", limitation="dry_run", n_loaded=int(len(samples)))
        return 0
    if len(samples) == 0:
        write_status(output_dir, "parcial", limitation="nenhuma_imagem_carregada", n_loaded=0)
        return 0

    attack_config = config.get("attack", {})
    result = evaluate_table4_imagenet(
        model=model,
        samples=samples,
        intervals=configured_intervals(config),
        epsilon_255=epsilon_255_from_config(config, args.epsilon),
        clip_min=float(attack_config.get("clip_min", 0.0)),
        clip_max=float(attack_config.get("clip_max", 1.0)),
    )
    print("total_images={0}".format(result.n_clean_total))
    print("clean_correct={0}".format(result.n_clean_correct))
    print("skipped_wrong_baseline={0}".format(result.skipped_wrong_baseline))
    output_config = config.get("output", {})
    csv_path, diagnostics_path = write_table4_outputs(
        output_dir=output_dir,
        result=result,
        csv_name=str(output_config.get("csv", "table_4_imagenet.csv")),
        diagnostics_name=str(
            output_config.get("diagnostics_csv", "table_4_imagenet_diagnostics.csv")
        ),
    )
    try:
        validate_attack_success(result)
    except RuntimeError as exc:
        write_status(
            output_dir,
            "falhou_fgsm_sem_sucesso",
            n_loaded=int(len(samples)),
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
        n_loaded=int(len(samples)),
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
