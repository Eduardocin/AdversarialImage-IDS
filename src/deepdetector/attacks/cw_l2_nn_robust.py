"""CW-L2 attack adapter for carlini/nn_robust_attacks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from deepdetector.io.paths import resolve_project_path


def _patch_tensorflow_v1_symbols() -> None:
    """Expose TF1 symbols expected by nn_robust_attacks when running on TF2."""
    try:
        import tensorflow as tf
    except Exception:
        return

    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass

    names = [
        "placeholder",
        "Session",
        "ConfigProto",
        "global_variables",
        "trainable_variables",
        "variables_initializer",
        "global_variables_initializer",
        "reset_default_graph",
        "get_default_graph",
        "assign",
        "gradients",
    ]
    for name in names:
        if not hasattr(tf, name) and hasattr(tf.compat.v1, name):
            setattr(tf, name, getattr(tf.compat.v1, name))

    try:
        if not hasattr(tf.train, "AdamOptimizer"):
            tf.train.AdamOptimizer = tf.compat.v1.train.AdamOptimizer
    except Exception:
        pass


def _patch_keras_backend_symbols() -> None:
    """Expose old/new Keras image-format helpers expected by legacy code."""
    try:
        from keras import backend as K
    except Exception:
        return

    if not hasattr(K, "set_image_dim_ordering") and hasattr(K, "set_image_data_format"):
        def set_image_dim_ordering(value: str) -> None:
            data_format = "channels_last" if value == "tf" else "channels_first"
            K.set_image_data_format(data_format)

        K.set_image_dim_ordering = set_image_dim_ordering
    if not hasattr(K, "image_dim_ordering") and hasattr(K, "image_data_format"):
        def image_dim_ordering() -> str:
            return "tf" if K.image_data_format() == "channels_last" else "th"

        K.image_dim_ordering = image_dim_ordering
    if not hasattr(K, "set_image_data_format") and hasattr(K, "set_image_dim_ordering"):
        def set_image_data_format(value: str) -> None:
            dim_ordering = "tf" if value == "channels_last" else "th"
            K.set_image_dim_ordering(dim_ordering)

        K.set_image_data_format = set_image_data_format
    if not hasattr(K, "image_data_format") and hasattr(K, "image_dim_ordering"):
        def image_data_format() -> str:
            return "channels_last" if K.image_dim_ordering() == "tf" else "channels_first"

        K.image_data_format = image_data_format


class NnRobustM2Adapter(object):
    """Expose the M2 Keras model through the nn_robust_attacks contract."""

    image_size = 28
    num_channels = 1
    num_labels = 10

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, data: Any) -> Any:
        """Return logits for centered input tensors in [-0.5, 0.5]."""
        return self.model(data + 0.5)


def _one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    label_array = np.asarray(labels)
    if label_array.ndim == 2:
        return label_array.astype(np.float32)
    output = np.zeros((len(label_array), int(num_classes)), dtype=np.float32)
    output[np.arange(len(label_array)), label_array.astype(np.int64)] = 1.0
    return output


def _resolve_nn_robust_root(root: Any) -> Path:
    configured = root or "nn_robust_attacks"
    path = resolve_project_path(configured) or Path(str(configured)).expanduser()
    if not path.is_dir():
        raise ValueError(
            "nn_robust_attacks root not found for cw_l2_nn_robust: {0}".format(path)
        )
    return path


def _load_carlini_l2(root: Path) -> Any:
    _patch_tensorflow_v1_symbols()
    _patch_keras_backend_symbols()
    attack_path = root / "l2_attack.py"
    if not attack_path.is_file():
        raise ImportError("Missing nn_robust_attacks l2_attack.py: {0}".format(attack_path))
    spec = importlib.util.spec_from_file_location(
        "deepdetector_nn_robust_l2_attack",
        str(attack_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load nn_robust_attacks from {0}".format(attack_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CarliniL2


def _set_keras_inference_phase() -> None:
    _patch_keras_backend_symbols()
    try:
        from keras import backend as K

        if hasattr(K, "set_learning_phase"):
            K.set_learning_phase(0)
    except Exception:
        pass


def generate_cw_l2_nn_robust_attack(
    *,
    graph: dict[str, Any],
    images: np.ndarray,
    labels: np.ndarray,
    kappa: float = 0.0,
    nn_robust_attacks_root: Any = None,
    batch_size: int = 1,
    max_iterations: int = 2000,
    binary_search_steps: int = 5,
    initial_const: float = 1.0,
    learning_rate: float = 0.1,
    targeted: bool = False,
    abort_early: bool = True,
    **_: Any,
) -> np.ndarray:
    """Generate CW-L2 examples using the nn_robust_attacks CarliniL2 backend."""
    root = _resolve_nn_robust_root(nn_robust_attacks_root)
    CarliniL2 = _load_carlini_l2(root)
    clean_images = np.asarray(images, dtype=np.float32)
    centered_images = clean_images - 0.5
    _patch_tensorflow_v1_symbols()
    _set_keras_inference_phase()
    attack = CarliniL2(
        graph["sess"],
        NnRobustM2Adapter(graph["model"]),
        batch_size=int(batch_size),
        max_iterations=int(max_iterations),
        confidence=float(kappa),
        binary_search_steps=int(binary_search_steps),
        initial_const=float(initial_const),
        learning_rate=float(learning_rate),
        targeted=bool(targeted),
        abort_early=bool(abort_early),
        boxmin=-0.5,
        boxmax=0.5,
    )
    centered_adv = attack.attack(centered_images, _one_hot(labels))
    return np.clip(np.asarray(centered_adv, dtype=np.float32) + 0.5, 0.0, 1.0)
