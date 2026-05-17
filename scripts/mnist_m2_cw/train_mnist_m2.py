"""Train or load the MNIST M2 clean baseline for CW experiments."""

from __future__ import print_function

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np


SCRIPTS_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "_project_root.py").is_file()
)
sys.path.insert(0, str(SCRIPTS_ROOT))
from _project_root import configure_project_paths

PROJECT_ROOT = configure_project_paths(__file__)

from deepdetector.data.mnist import load_mnist_data
from deepdetector.models.mnist_cnn import create_tf_session
from deepdetector.models.mnist_m2 import build_mnist_m2_model
from deepdetector.training.train_mnist_m2 import train_or_load_mnist_m2_model


SEED_TF = 1234
SEED_NUMPY = 20170830
SUMMARY_DIR = PROJECT_ROOT / "results" / "mnist" / "m2_cw" / "clean_baseline"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--train-dir",
        default=str(SUMMARY_DIR / "checkpoints"),
        help="Directory used for TensorFlow checkpoint files.",
    )
    parser.add_argument("--filename", default="mnist_m2.ckpt")
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Restore an existing M2 checkpoint from --train-dir when available.",
    )
    return parser


def _summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return the stable CSV row for the M2 clean baseline."""
    return {
        "experiment": "mnist_m2_clean_baseline",
        "dataset": "mnist",
        "model": "M2",
        "epochs": result["epochs"],
        "batch_size": result["batch_size"],
        "learning_rate": result["learning_rate"],
        "test_accuracy_clean": result["clean_accuracy"],
        "checkpoint_path": result["checkpoint_path"],
        "seed_tf": SEED_TF,
        "seed_numpy": SEED_NUMPY,
        "notes": (
            "Separate M2 CNN checkpoint for MNIST CW reproduction; "
            "architecture aligned to the base M2 definition."
        ),
    }


def write_summary_csv(result: Dict[str, Any], output_dir: Path) -> Path:
    """Write the clean baseline summary CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"
    fieldnames = [
        "experiment",
        "dataset",
        "model",
        "epochs",
        "batch_size",
        "learning_rate",
        "test_accuracy_clean",
        "checkpoint_path",
        "seed_tf",
        "seed_numpy",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(_summary_row(result))
    return path


def write_summary_md(result: Dict[str, Any], output_dir: Path) -> Path:
    """Write the clean baseline summary Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.md"
    lines = [
        "# MNIST M2 Clean Baseline",
        "",
        "## Configuration",
        "",
        "- dataset: MNIST",
        "- model: M2 CNN base architecture",
        "- epochs: {0}".format(result["epochs"]),
        "- batch_size: {0}".format(result["batch_size"]),
        "- learning_rate: {0}".format(result["learning_rate"]),
        "- train_dir: `{0}`".format(result["train_dir"]),
        "- filename: `{0}`".format(result["filename"]),
        "- trained_from_scratch: {0}".format(result["trained_from_scratch"]),
        "- seed_tf: {0}".format(SEED_TF),
        "- seed_numpy: {0}".format(SEED_NUMPY),
        "",
        "## Metrics",
        "",
        "- test_accuracy_clean: {0:.6f}".format(result["clean_accuracy"]),
        "",
        "## Checkpoint",
        "",
        "`{0}`".format(result["checkpoint_path"]),
        "",
        "## Notes",
        "",
        "The exact M2 architecture from reference [36] is not present in this "
        "repository. This checkpoint uses the base M2 CNN definition aligned "
        "in `src/deepdetector/models/mnist_m2.py`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    """Run the M2 training or restore path."""
    args = build_parser().parse_args()

    import tensorflow as tf

    tf.compat.v1.set_random_seed(SEED_TF)
    rng = np.random.RandomState([2017, 8, 30])
    sess = create_tf_session()

    X_train, Y_train, X_test, Y_test = load_mnist_data(rng=rng)
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 28, 28, 1), name="x")
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name="y")
    _, predictions = build_mnist_m2_model(x)

    result = train_or_load_mnist_m2_model(
        sess=sess,
        x=x,
        y=y,
        predictions=predictions,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        config={
            "nb_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_dir": args.train_dir,
            "filename": args.filename,
            "load_model": args.load_model,
            "rng": rng,
        },
    )

    csv_path = write_summary_csv(result, SUMMARY_DIR)
    md_path = write_summary_md(result, SUMMARY_DIR)
    print("test_accuracy_clean={0:.6f}".format(result["clean_accuracy"]))
    print("checkpoint_path={0}".format(result["checkpoint_path"]))
    print("summary_csv={0}".format(csv_path))
    print("summary_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
