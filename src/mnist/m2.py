"""Carlini MNIST M2 checks and training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from pathlib import Path
import sys
import time
from typing import List, Optional

from .config import MnistExperimentConfig, M2Config
from .data import ensure_carlini_mnist_data


@dataclass(frozen=True)
class M2TrainingConfig:
    """Training configuration for Carlini's MNIST M2 architecture."""

    nn_robust_attacks_root: Path
    output: Path
    epochs: int = 50
    batch_size: int = 128
    keras_verbose: int = 0


@dataclass(frozen=True)
class M2CheckResult:
    """Summary of a Carlini M2 weights check."""

    weights_path: Path
    exists: bool
    file_size: int
    loaded: bool


def check_m2(config: Optional[MnistExperimentConfig] = None) -> M2CheckResult:
    """Check whether the Carlini-compatible M2 model can be loaded."""

    cfg = config or MnistExperimentConfig()
    weights_path = cfg.paths.m2_weights_path
    exists = weights_path.exists()
    file_size = weights_path.stat().st_size if exists else 0

    print("[M2] weights_path={}".format(weights_path), flush=True)
    print("[M2] weights_exists={}".format(exists), flush=True)
    print("[M2] weights_size_bytes={}".format(file_size), flush=True)
    if not exists:
        return M2CheckResult(weights_path=weights_path, exists=False, file_size=0, loaded=False)

    from .models import build_m2_inference_model

    print("[M2] loading Carlini-compatible M2 architecture", flush=True)
    build_m2_inference_model(weights_path)
    print("[M2] load_ok=True", flush=True)
    return M2CheckResult(weights_path=weights_path, exists=True, file_size=file_size, loaded=True)


def _require_existing_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError("{} not found: {}".format(label, resolved))
    return resolved


def train_m2(config: M2TrainingConfig) -> Path:
    """Train and save Carlini's MNIST M2 model."""

    nn_root = _require_existing_path(config.nn_robust_attacks_root, "nn_robust_attacks root")
    output = config.output
    output.parent.mkdir(parents=True, exist_ok=True)

    experiment_config = MnistExperimentConfig()
    paths = experiment_config.paths
    if nn_root != paths.nn_robust_attacks_root:
        paths = replace(paths, nn_robust_attacks_root=nn_root)
    ensure_carlini_mnist_data(paths=paths, source_dir=None)

    sys.path.insert(0, str(nn_root))
    os.chdir(str(nn_root))

    import tensorflow as tf
    from keras.callbacks import Callback
    from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from keras.models import Sequential
    from keras.optimizers import SGD
    from setup_mnist import MNIST

    class EpochLogger(Callback):
        def __init__(self, total_epochs: int) -> None:
            super().__init__()
            self.total_epochs = total_epochs
            self.started_at = 0.0

        def on_epoch_begin(self, epoch: int, logs=None) -> None:
            self.started_at = time.time()
            print("[M2 train] epoch {}/{} started".format(epoch + 1, self.total_epochs), flush=True)

        def on_epoch_end(self, epoch: int, logs=None) -> None:
            elapsed = time.time() - self.started_at
            metrics = logs or {}
            metric_text = " ".join(
                "{}={:.4f}".format(name, value)
                for name, value in sorted(metrics.items())
                if isinstance(value, (float, int))
            )
            print(
                "[M2 train] epoch {}/{} finished in {:.1f}s {}".format(
                    epoch + 1,
                    self.total_epochs,
                    elapsed,
                    metric_text,
                ),
                flush=True,
            )

    print("[M2 train] nn_robust_attacks_root={}".format(nn_root), flush=True)
    print("[M2 train] output={}".format(output), flush=True)
    print(
        "[M2 train] epochs={} batch_size={} keras_verbose={}".format(
            config.epochs,
            config.batch_size,
            config.keras_verbose,
        ),
        flush=True,
    )
    data = MNIST()
    print(
        "[M2 train] data train={} validation={} test={}".format(
            data.train_data.shape,
            data.validation_data.shape,
            data.test_data.shape,
        ),
        flush=True,
    )

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=data.train_data.shape[1:]))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(Dense(10))

    def loss(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    model.fit(
        data.train_data,
        data.train_labels,
        batch_size=config.batch_size,
        validation_data=(data.validation_data, data.validation_labels),
        epochs=config.epochs,
        shuffle=True,
        callbacks=[EpochLogger(config.epochs)],
        verbose=config.keras_verbose,
    )
    model.save(str(output))
    print("[M2 train] saved_m2_model={}".format(output), flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    experiment_config = MnistExperimentConfig()
    m2_config = M2Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nn-robust-attacks-root",
        type=Path,
        default=experiment_config.paths.nn_robust_attacks_root,
    )
    parser.add_argument("--epochs", type=int, default=m2_config.epochs)
    parser.add_argument("--batch-size", type=int, default=m2_config.batch_size)
    parser.add_argument(
        "--keras-verbose",
        type=int,
        choices=[0, 1, 2],
        default=m2_config.keras_verbose,
        help="Keras fit verbosity. The script always prints epoch start/end logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <nn_robust_attacks_root>/models/mnist.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Train M2 from the command line."""

    args = build_parser().parse_args(argv)
    nn_root = args.nn_robust_attacks_root.resolve()
    output = args.output or (nn_root / "models" / "mnist")
    saved_path = train_m2(
        M2TrainingConfig(
            nn_robust_attacks_root=nn_root,
            output=output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            keras_verbose=args.keras_verbose,
        )
    )
    print("saved_m2_model={}".format(saved_path))
    return 0
