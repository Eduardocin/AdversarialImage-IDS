"""Train or restore the MNIST M2 clean baseline model."""

from __future__ import print_function

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepdetector.data.mnist import load_mnist_data  # noqa: E402
from deepdetector.io.paths import resolve_project_path  # noqa: E402
from deepdetector.models.mnist_cnn import create_tf_session  # noqa: E402
from deepdetector.models.mnist_m2 import build_mnist_m2_model  # noqa: E402
from deepdetector.training.train_mnist_m2 import train_or_load_mnist_m2_model  # noqa: E402


DEFAULT_TRAIN_DIR = PROJECT_ROOT / "artifacts" / "models" / "mnist" / "m2" / (
    "clean_baseline_tf2"
) / "checkpoints"


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--filename", default="mnist_m2.ckpt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=60000)
    parser.add_argument("--test-start", type=int, default=0)
    parser.add_argument("--test-end", type=int, default=10000)
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Restore an existing checkpoint from --train-dir when available.",
    )
    return parser


def training_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Return the train_or_load_mnist_m2_model config."""
    train_dir = resolve_project_path(args.train_dir, project_root=PROJECT_ROOT)
    return {
        "train_dir": str(train_dir or Path(args.train_dir)),
        "filename": args.filename,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "label_smoothing": args.label_smoothing,
        "load_model": bool(args.load_model),
    }


def main() -> int:
    """Run M2 training and print a JSON summary."""
    args = build_parser().parse_args()

    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    x_train, y_train, x_test, y_test = load_mnist_data(
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    sess = create_tf_session()
    try:
        x_placeholder = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, 28, 28, 1),
            name="x",
        )
        y_placeholder = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, 10),
            name="y",
        )
        _, predictions = build_mnist_m2_model(x_placeholder)
        result = train_or_load_mnist_m2_model(
            sess=sess,
            x=x_placeholder,
            y=y_placeholder,
            predictions=predictions,
            X_train=x_train,
            Y_train=y_train,
            X_test=x_test,
            Y_test=y_test,
            config=training_config(args),
        )
    finally:
        sess.close()

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
