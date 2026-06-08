"""Train or restore Fashion-MNIST checkpoints for the M2 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepdetector.data.fashion_mnist import split_fashion_mnist_balanced
from deepdetector.io.config import load_yaml_config
from deepdetector.io.paths import resolve_project_path
from deepdetector.models.mnist_cnn import create_tf_session
from deepdetector.models.mnist_m2 import build_mnist_m2_model
from deepdetector.training.train_mnist_m2 import train_or_load_mnist_m2_model


DEFAULT_CONFIG = resolve_project_path("configs/experiments.yaml")
DEFAULT_EXPERIMENT_BY_MODEL = {
    "m2": "fashion_mnist_cw_l2_m2",
}
DEFAULT_FILENAME_BY_MODEL = {
    "m2": "mnist_m2.ckpt",
}
DEFAULT_EPOCHS_BY_MODEL = {
    "m2": 10,
}


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(DEFAULT_EXPERIMENT_BY_MODEL), required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--experiment",
        help="Experiment config to read. Defaults to the Fashion-MNIST experiment for --model.",
    )
    parser.add_argument("--train-dir", help="Override model.checkpoint_dir.")
    parser.add_argument("--filename", help="Checkpoint filename inside the training directory.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Restore an existing checkpoint from train-dir when available.",
    )
    return parser


def build_training_graph(model_name: str) -> Dict[str, Any]:
    """Create the TensorFlow graph for a Fashion-MNIST-compatible model."""
    import tensorflow as tf

    tf.compat.v1.reset_default_graph()
    sess = create_tf_session()
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 28, 28, 1), name="x")
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name="y")

    if model_name == "m2":
        model, predictions = build_mnist_m2_model(x)
    else:
        raise ValueError("Unsupported Fashion-MNIST model: {0}".format(model_name))

    return {
        "sess": sess,
        "x": x,
        "y": y,
        "model": model,
        "predictions": predictions,
    }


def _experiment_id_for_model(model_name: str, experiment_id: Optional[str]) -> str:
    if experiment_id:
        return experiment_id
    return DEFAULT_EXPERIMENT_BY_MODEL[model_name]


def _experiment_config(
    model_name: str,
    consolidated_config: Dict[str, Any],
    experiment_id: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    resolved_experiment_id = _experiment_id_for_model(model_name, experiment_id)
    experiments = consolidated_config.get("experiments", {})
    if resolved_experiment_id not in experiments:
        raise ValueError("Unknown experiment: {0}".format(resolved_experiment_id))

    experiment = dict(experiments[resolved_experiment_id])
    configured_model = str(
        experiment.get("model_group", experiment.get("model", {}).get("name", ""))
    ).lower()
    if configured_model != model_name:
        raise ValueError(
            "Experiment {0} is configured for model {1}, not {2}.".format(
                resolved_experiment_id,
                configured_model or "<missing>",
                model_name,
            )
        )
    if str(experiment.get("dataset", {}).get("name", "")).lower() != "fashion_mnist":
        raise ValueError("Fashion-MNIST checkpoint training requires dataset.name=fashion_mnist.")
    return resolved_experiment_id, experiment


def _training_config(
    model_name: str,
    experiment: Dict[str, Any],
    train_dir_override: Optional[str],
    filename: Optional[str],
    epochs: Optional[int],
    batch_size: int,
    learning_rate: float,
    label_smoothing: float,
    load_model: bool,
) -> Dict[str, Any]:
    configured_train_dir = train_dir_override or experiment.get("model", {}).get("checkpoint_dir")
    train_dir = resolve_project_path(configured_train_dir)
    if train_dir is None:
        raise ValueError("Fashion-MNIST model.checkpoint_dir is required.")

    return {
        "train_dir": str(train_dir),
        "filename": filename or DEFAULT_FILENAME_BY_MODEL[model_name],
        "epochs": int(epochs if epochs is not None else DEFAULT_EPOCHS_BY_MODEL[model_name]),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "label_smoothing": float(label_smoothing),
        "load_model": bool(load_model),
    }


def _expected_checkpoint_training(dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    strategy = dict(dataset_config.get("split_strategy", {}))
    train_quota = int(strategy.get("train_class_quota", 900))
    eval_quota = int(strategy.get("evaluation_class_quota", 100))
    return {
        "source_csv": dataset_config.get("csv_path"),
        "split_name": "train",
        "selection": {
            "method": "first_n_per_class",
            "per_class_start": 0,
            "per_class_end": train_quota,
            "class_quota": train_quota,
            "total_samples": int(strategy.get("train_samples", train_quota * 10)),
        },
        "validation_sample": {
            "split_name": "evaluation",
            "method": "next_n_per_class",
            "per_class_start": train_quota,
            "per_class_end": train_quota + eval_quota,
            "class_quota": eval_quota,
            "total_samples": int(strategy.get("evaluation_samples", eval_quota * 10)),
            "excludes_training": True,
        },
    }


def _validate_mapping(
    *,
    name: str,
    configured: Dict[str, Any],
    expected: Dict[str, Any],
) -> None:
    for key, expected_value in expected.items():
        if isinstance(expected_value, dict):
            configured_value = configured.get(key)
            if not isinstance(configured_value, dict):
                raise ValueError(
                    "Fashion-MNIST {0}.{1} must be configured.".format(name, key)
                )
            _validate_mapping(
                name="{0}.{1}".format(name, key),
                configured=configured_value,
                expected=expected_value,
            )
            continue

        configured_value = configured.get(key)
        if configured_value != expected_value:
            raise ValueError(
                "Fashion-MNIST {0}.{1} must be {2!r}; got {3!r}.".format(
                    name,
                    key,
                    expected_value,
                    configured_value,
                )
            )


def _validate_checkpoint_training(
    experiment: Dict[str, Any],
    train_images: np.ndarray,
    eval_images: np.ndarray,
) -> Dict[str, Any]:
    configured = experiment.get("checkpoint_training")
    if not isinstance(configured, dict):
        raise ValueError("Fashion-MNIST checkpoint_training is required.")

    expected = _expected_checkpoint_training(experiment.get("dataset", {}))
    _validate_mapping(
        name="checkpoint_training",
        configured=configured,
        expected=expected,
    )

    selection = dict(configured.get("selection", {}))
    validation_sample = dict(configured.get("validation_sample", {}))
    if int(selection.get("total_samples", -1)) != int(len(train_images)):
        raise ValueError(
            "Fashion-MNIST checkpoint_training.selection.total_samples must match loaded train samples."
        )
    if int(validation_sample.get("total_samples", -1)) != int(len(eval_images)):
        raise ValueError(
            "Fashion-MNIST checkpoint_training.validation_sample.total_samples must match loaded evaluation samples."
        )
    if not bool(validation_sample.get("excludes_training", False)):
        raise ValueError(
            "Fashion-MNIST checkpoint_training.validation_sample.excludes_training must be true."
        )
    return dict(configured)


def _trainer_for_model(model_name: str) -> Any:
    if model_name == "m2":
        return train_or_load_mnist_m2_model
    raise ValueError("Unsupported Fashion-MNIST model: {0}".format(model_name))


def run_training(
    model_name: str,
    consolidated_config: Dict[str, Any],
    experiment_id: Optional[str] = None,
    train_dir_override: Optional[str] = None,
    filename: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    label_smoothing: float = 0.1,
    load_model: bool = False,
) -> Dict[str, Any]:
    """Train or restore the configured Fashion-MNIST checkpoint."""
    model_key = model_name.lower()
    experiment_name, experiment = _experiment_config(
        model_key,
        consolidated_config,
        experiment_id,
    )
    split = split_fashion_mnist_balanced(experiment.get("dataset", {}))
    train_images = np.asarray(split["train_images"], dtype=np.float32)
    train_labels = np.asarray(split["train_labels_one_hot"], dtype=np.float32)
    eval_images = np.asarray(split["evaluation_images"], dtype=np.float32)
    eval_labels = np.asarray(split["evaluation_labels_one_hot"], dtype=np.float32)
    metadata = dict(split["metadata"])
    strategy = dict(metadata.get("split_strategy", {}))
    checkpoint_training = _validate_checkpoint_training(
        experiment,
        train_images,
        eval_images,
    )
    training_config = _training_config(
        model_key,
        experiment,
        train_dir_override,
        filename,
        epochs,
        batch_size,
        learning_rate,
        label_smoothing,
        load_model,
    )

    graph = build_training_graph(model_key)
    try:
        result = _trainer_for_model(model_key)(
            graph["sess"],
            graph["x"],
            graph["y"],
            graph["predictions"],
            train_images,
            train_labels,
            eval_images,
            eval_labels,
            training_config,
        )
    finally:
        close = getattr(graph.get("sess"), "close", None)
        if close is not None:
            close()

    result = dict(result)
    result.update(
        {
            "experiment_id": experiment_name,
            "dataset": "fashion_mnist",
            "model": model_key,
            "csv_path": metadata.get("csv_path"),
            "train_samples": int(len(train_images)),
            "evaluation_samples": int(len(eval_images)),
            "train_class_quota": int(strategy.get("train_class_quota", 0)),
            "evaluation_class_quota": int(strategy.get("evaluation_class_quota", 0)),
            "checkpoint_training": checkpoint_training,
        }
    )
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Object of type {0} is not JSON serializable".format(type(value).__name__))


def main() -> int:
    """Load config, train or restore a checkpoint, and print a JSON summary."""
    args = build_parser().parse_args()
    config_path = resolve_project_path(args.config) or DEFAULT_CONFIG
    if config_path is None:
        raise ValueError("A config path is required.")

    try:
        result = run_training(
            model_name=args.model,
            consolidated_config=load_yaml_config(config_path),
            experiment_id=args.experiment,
            train_dir_override=args.train_dir,
            filename=args.filename,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            label_smoothing=args.label_smoothing,
            load_model=args.load_model,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))

    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
