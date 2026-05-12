"""Environment checks for the MNIST DeepDetector implementation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import MnistExperimentConfig


def check_m1_cleverhans() -> List[str]:
    """Check TensorFlow/Keras/CleverHans imports required by M1/FGSM."""

    failures: List[str] = []
    try:
        import keras
        import keras.backend as keras_backend
        import tensorflow as tf
        from cleverhans.attacks import FastGradientMethod
        from cleverhans.utils import AccuracyReport
        from cleverhans.utils_keras import KerasModelWrapper, cnn_model
        from cleverhans.utils_mnist import data_mnist
        from cleverhans.utils_tf import model_argmax, model_eval, model_train
    except Exception as exc:  # pragma: no cover - exercised in legacy env.
        failures.append("M1 CleverHans import failed: {}".format(exc))
        return failures

    required_tf_attrs = ["Session", "app", "placeholder", "set_random_seed"]
    missing_tf_attrs = [name for name in required_tf_attrs if not hasattr(tf, name)]
    if missing_tf_attrs:
        failures.append("TensorFlow is missing TF1 APIs: {}".format(missing_tf_attrs))

    required_keras_attrs = ["set_session", "image_dim_ordering", "set_image_dim_ordering"]
    missing_keras_attrs = [name for name in required_keras_attrs if not hasattr(keras_backend, name)]
    if missing_keras_attrs:
        failures.append("Keras backend is missing legacy APIs: {}".format(missing_keras_attrs))

    print("[OK] tensorflow {}".format(tf.__version__))
    print("[OK] keras {}".format(keras.__version__))
    print("[OK] cleverhans M1 imports")
    print("[OK] M1 symbols: {}, {}".format(data_mnist.__name__, cnn_model.__name__))
    print(
        "[OK] M1 helpers: {}, {}, {}, {}, {}, {}".format(
            model_train.__name__,
            model_eval.__name__,
            model_argmax.__name__,
            FastGradientMethod.__name__,
            KerasModelWrapper.__name__,
            AccuracyReport.__name__,
        )
    )
    return failures


def check_m2_carlini(nn_robust_attacks_root: Path) -> List[str]:
    """Check Carlini imports and M2 weights required by C&W."""

    failures: List[str] = []
    if not nn_robust_attacks_root.exists():
        return ["nn_robust_attacks root not found: {}".format(nn_robust_attacks_root)]

    sys.path.insert(0, str(nn_robust_attacks_root))
    try:
        from l2_attack import CarliniL2
        from li_attack import CarliniLi
        from setup_mnist import MNIST, MNISTModel
    except Exception as exc:  # pragma: no cover - exercised in legacy env.
        failures.append("M2 Carlini import failed: {}".format(exc))
        return failures

    model_path = nn_robust_attacks_root / "models" / "mnist"
    if not model_path.exists():
        failures.append(
            "Carlini M2 weights are missing. Expected {}. "
            "Train or place the original weights before C&W runs.".format(model_path)
        )

    print("[OK] nn_robust_attacks root: {}".format(nn_robust_attacks_root))
    print("[OK] M2 symbols: {}, {}".format(MNIST.__name__, MNISTModel.__name__))
    print("[OK] C&W attacks: {}, {}".format(CarliniL2.__name__, CarliniLi.__name__))
    return failures


def build_parser() -> argparse.ArgumentParser:
    config = MnistExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nn-robust-attacks-root",
        type=Path,
        default=config.paths.nn_robust_attacks_root,
        help="Path to Carlini's nn_robust_attacks checkout used by M2/C&W.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run all MNIST environment checks."""

    args = build_parser().parse_args(argv)
    failures = []
    failures.extend(check_m1_cleverhans())
    failures.extend(check_m2_carlini(args.nn_robust_attacks_root.resolve()))

    if failures:
        print("")
        print("MNIST environment is incomplete:")
        for failure in failures:
            print("- {}".format(failure))
        return 1

    print("")
    print("MNIST M1/M2 environment is ready.")
    return 0

