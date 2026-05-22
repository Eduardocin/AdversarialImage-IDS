"""Train or load the MNIST clean baseline with TF1/Keras/CleverHans."""

from __future__ import print_function

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())

from deepdetector.data.mnist import load_mnist_data
from deepdetector.models.mnist_cnn import build_mnist_model, create_tf_session
from deepdetector.paths import MNIST_M1_CHECKPOINT_DIR, MNIST_RESULTS_DIR
from deepdetector.training.train_mnist_m1 import train_or_load_mnist_model


SUMMARY_DIR = MNIST_RESULTS_DIR / "clean_baseline"
SUMMARY_PATH = SUMMARY_DIR / "summary.md"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--train-dir",
        default=str(MNIST_M1_CHECKPOINT_DIR),
        help="Directory used for TensorFlow checkpoint files.",
    )
    parser.add_argument("--filename", default="mnist.ckpt")
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Restore an existing checkpoint from --train-dir when available.",
    )
    return parser


def write_summary(result: Dict[str, Any]) -> Path:
    """Write a Markdown summary for the clean MNIST baseline."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MNIST Clean Baseline",
        "",
        "## Hyperparameters",
        "",
        "- epochs: {0}".format(result["epochs"]),
        "- batch_size: {0}".format(result["batch_size"]),
        "- learning_rate: {0}".format(result["learning_rate"]),
        "- label_smoothing: {0}".format(result["label_smoothing"]),
        "- train_dir: `{0}`".format(result["train_dir"]),
        "- filename: `{0}`".format(result["filename"]),
        "- trained_from_scratch: {0}".format(result["trained_from_scratch"]),
        "",
        "## Metrics",
        "",
        "- clean_accuracy: {0:.6f}".format(result["clean_accuracy"]),
        "",
        "## Checkpoint",
        "",
        "`{0}`".format(result["checkpoint_path"]),
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    return SUMMARY_PATH


def main() -> int:
    """Run the clean MNIST training or restore path."""
    args = build_parser().parse_args()

    import tensorflow as tf

    tf.compat.v1.set_random_seed(1234)
    rng = np.random.RandomState([2017, 8, 30])
    sess = create_tf_session()

    X_train, Y_train, X_test, Y_test = load_mnist_data(rng=rng)
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 28, 28, 1), name="x")
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name="y")

    _, predictions = build_mnist_model(x)

    result = train_or_load_mnist_model(
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

    summary_path = write_summary(result)
    print("clean_accuracy={0:.6f}".format(result["clean_accuracy"]))
    print("checkpoint_path={0}".format(result["checkpoint_path"]))
    print("summary_path={0}".format(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

