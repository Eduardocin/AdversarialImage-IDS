"""CW-L2 attack adapter for carlini/nn_robust_attacks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from deepdetector.io.paths import resolve_project_path


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
