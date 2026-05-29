"""Runner for the ImageNet half of Table 4."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from deepdetector.data.imagenet import resize_normalized_image
from deepdetector.experiments.adversarial_examples import DEFAULT_ADVERSARIAL_CACHE_DIR
from deepdetector.evaluation.table4_imagenet import (
    TABLE4_INTERVALS,
    Table4Sample,
    evaluate_table4_imagenet,
    validate_attack_success,
    write_table4_outputs,
)
from deepdetector.io.paths import resolve_project_path
from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper


logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve a config path relative to the project root."""
    return resolve_project_path(path_value)


def _json_key(data: Dict[str, Any]) -> str:
    """Return deterministic JSON text for cache key material."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _cache_digest(data: Dict[str, Any]) -> str:
    """Return a short stable cache digest."""
    return hashlib.sha256(_json_key(data).encode("utf-8")).hexdigest()[:20]


def _attack_cache_enabled(config: Dict[str, Any]) -> bool:
    """Return whether the configured attack cache is enabled."""
    return bool(config.get("attack", {}).get("cache", True))


def _attack_cache_root(config: Dict[str, Any]) -> Path:
    """Return the configured adversarial example cache root."""
    cache_dir = config.get("attack", {}).get("cache_dir") or DEFAULT_ADVERSARIAL_CACHE_DIR
    return _resolve_path(cache_dir) or Path(str(cache_dir))


def configured_intervals(config: Dict[str, Any]) -> Iterable[int]:
    """Return scalar quantization interval counts from config."""
    intervals = config.get("quantization", {}).get("intervals", TABLE4_INTERVALS)
    if not intervals:
        raise ValueError("Config must define quantization.intervals.")
    for value in intervals:
        yield int(value)


def write_status(output_dir: Path, status: str, **fields: Any) -> Path:
    """Write a status JSON file for partial ImageNet runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": status}
    payload.update(fields)
    status_name = str(
        config_status_name(fields.get("config"))
        if isinstance(fields.get("config"), dict)
        else "table_4_status.json"
    )
    path = output_dir / status_name
    payload.pop("config", None)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def config_status_name(config: Dict[str, Any]) -> str:
    """Return the configured status JSON filename."""
    return str(config.get("output", {}).get("status_json", "table_4_status.json"))


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


def _class_image_rows(
    images_dir: Path,
    class_indices: Dict[str, Any],
) -> List[Tuple[Path, str, int]]:
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

    configured_limit = (
        limit_override if limit_override is not None else dataset_config.get("n_samples")
    )
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


def _table4_cache_key(
    config: Dict[str, Any],
    samples: List[Table4Sample],
    epsilon_255: float,
    clip_min: float,
    clip_max: float,
) -> Dict[str, Any]:
    """Return cache key material for ImageNet Table 4 FGSM attacks."""
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    return {
        "version": 1,
        "kind": "imagenet_table4_fgsm",
        "dataset": {
            "name": "imagenet",
            "split": dataset_config.get("split"),
            "images_dir": str(_resolve_path(dataset_config.get("images_dir"))),
            "image_size": int(dataset_config.get("image_size", 224)),
            "samples": [
                {
                    "image_id": sample.image_id,
                    "class_name": sample.class_name,
                    "true_label": int(sample.true_label),
                }
                for sample in samples
            ],
        },
        "model": {
            "name": model_config.get("name"),
            "family": model_config.get("family"),
            "deploy_proto": str(_resolve_path(model_config.get("deploy_proto"))),
            "caffemodel": str(_resolve_path(model_config.get("caffemodel"))),
            "mean_file": str(_resolve_path(model_config.get("mean_file"))),
        },
        "attack": {
            "name": config.get("attack", {}).get("name", "fgsm"),
            "epsilon_255": float(epsilon_255),
            "clip_min": float(clip_min),
            "clip_max": float(clip_max),
        },
    }


def table4_adversarial_cache_path(
    config: Dict[str, Any],
    samples: List[Table4Sample],
    epsilon_255: float,
    clip_min: float,
    clip_max: float,
) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Return the configured ImageNet Table 4 attack cache path and key."""
    if not _attack_cache_enabled(config):
        return None
    attack_config = config.get("attack", {})
    explicit_path = attack_config.get("cache_path") or attack_config.get("adversarial_path")
    cache_key = _table4_cache_key(
        config=config,
        samples=samples,
        epsilon_255=epsilon_255,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    if explicit_path:
        cache_path = _resolve_path(explicit_path) or Path(str(explicit_path))
    else:
        cache_path = (
            _attack_cache_root(config)
            / "imagenet"
            / "fgsm"
            / "table_4"
            / "{0}.npz".format(_cache_digest(cache_key))
        )
    return cache_path, cache_key


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


def output_dir_from_config(config: Dict[str, Any], override: Optional[str] = None) -> Path:
    """Return the configured output directory."""
    output_config = config.get("output", {})
    output_dir = _resolve_path(override or output_config.get("dir"))
    if output_dir is None:
        raise ValueError("Config must define output.dir.")
    return output_dir


def run_table4_imagenet_experiment(
    config: Dict[str, Any],
    data_root_override: Optional[str] = None,
    limit_override: Optional[int] = None,
    epsilon_override: Optional[float] = None,
    output_dir_override: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run ImageNet Table 4 and write its CSV and status files."""
    output_dir = output_dir_from_config(config, output_dir_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Table 4 ImageNet output directory: %s", output_dir)

    try:
        logger.info("Loading GoogLeNet/Caffe model for Table 4 ImageNet.")
        model = build_model(config)
        logger.info("GoogLeNet/Caffe model loaded.")
    except ImportError as exc:
        logger.warning("%s", exc)
        status_path = write_status(
            output_dir,
            "bloqueado_caffe",
            limitation="caffe_indisponivel",
            message=str(exc),
            config=config,
        )
        return {"status": "bloqueado_caffe", "status_json": str(status_path)}
    except OSError as exc:
        logger.warning("%s", exc)
        status_path = write_status(
            output_dir,
            "bloqueado_modelo_googlenet",
            limitation="assets_googlenet_indisponiveis",
            message=str(exc),
            setup_command=(
                "python scripts/imagenet/download_caffe_imagenet_assets.py --model googlenet"
            ),
            config=config,
        )
        return {
            "status": "bloqueado_modelo_googlenet",
            "status_json": str(status_path),
        }

    try:
        dataset_config = config.get("dataset", {})
        logger.info(
            "Loading ImageNet Table 4 samples from %s.",
            data_root_override or dataset_config.get("images_dir"),
        )
        samples = load_subset_samples(
            config,
            data_root_override=data_root_override,
            limit_override=limit_override,
        )
    except (IOError, ValueError) as exc:
        status_path = write_status(
            output_dir,
            "falhou_dataset",
            message=str(exc),
            config=config,
        )
        raise ValueError(str(exc)) from exc

    logger.info("Loaded %d ImageNet Table 4 samples.", len(samples))
    if dry_run:
        status_path = write_status(
            output_dir,
            "parcial",
            limitation="dry_run",
            n_loaded=int(len(samples)),
            config=config,
        )
        return {"status": "parcial", "status_json": str(status_path)}
    if len(samples) == 0:
        status_path = write_status(
            output_dir,
            "parcial",
            limitation="nenhuma_imagem_carregada",
            n_loaded=0,
            config=config,
        )
        return {"status": "parcial", "status_json": str(status_path)}

    attack_config = config.get("attack", {})
    epsilon_255 = epsilon_255_from_config(config, epsilon_override)
    clip_min = float(attack_config.get("clip_min", 0.0))
    clip_max = float(attack_config.get("clip_max", 1.0))
    cache_info = table4_adversarial_cache_path(
        config=config,
        samples=samples,
        epsilon_255=epsilon_255,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    cache_path = cache_info[0] if cache_info else None
    cache_metadata = (
        {"cache_key": cache_info[1], "cache_path": str(cache_info[0])}
        if cache_info
        else None
    )
    logger.info(
        "Evaluating ImageNet Table 4 with epsilon_255=%s and intervals=%s.",
        epsilon_255,
        ", ".join(str(value) for value in configured_intervals(config)),
    )
    result = evaluate_table4_imagenet(
        model=model,
        samples=samples,
        intervals=configured_intervals(config),
        epsilon_255=epsilon_255,
        clip_min=clip_min,
        clip_max=clip_max,
        cache_path=cache_path,
        cache_metadata=cache_metadata,
    )
    output_config = config.get("output", {})
    logger.info("Writing ImageNet Table 4 CSV output.")
    csv_path = write_table4_outputs(
        output_dir=output_dir,
        result=result,
        csv_name=str(output_config.get("csv", "table_4_imagenet.csv")),
    )

    try:
        validate_attack_success(result)
    except RuntimeError as exc:
        status_path = write_status(
            output_dir,
            "falhou_fgsm_sem_sucesso",
            n_loaded=int(len(samples)),
            n_clean_correct=result.n_clean_correct,
            n_attack_success=result.n_attack_success,
            results_csv=str(csv_path),
            message=str(exc),
            config=config,
        )
        raise ValueError(str(exc)) from exc

    status_path = write_status(
        output_dir,
        "completo",
        n_loaded=int(len(samples)),
        n_clean_correct=result.n_clean_correct,
        n_attack_success=result.n_attack_success,
        results_csv=str(csv_path),
        config=config,
    )
    logger.info("ImageNet Table 4 completed: %s", csv_path)
    return {
        "status": "completo",
        "csv": str(csv_path),
        "status_json": str(status_path),
    }
