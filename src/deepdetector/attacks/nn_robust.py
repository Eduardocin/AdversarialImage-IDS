"""Registry-compatible adapters for carlini/nn_robust_attacks."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from deepdetector.attacks.tf_compat import patch_tensorflow_v1_symbols
from deepdetector.io.paths import resolve_project_path


def _predict_with_model(model: Any, data: Any) -> Any:
    if hasattr(model, "get_logits"):
        return model.get_logits(data)
    if callable(model):
        return model(data)
    if hasattr(model, "predict"):
        return model.predict(data)
    raise NotImplementedError("nn_robust_attacks requires model logits.")


class _ShiftedPredictModel(object):
    """Expose `.predict` with an input shift for original Carlini helpers."""

    def __init__(self, model: Any, input_shift: float) -> None:
        self._model = model
        self._input_shift = float(input_shift)

    def predict(self, data: Any) -> Any:
        return _predict_with_model(self._model, data + self._input_shift)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class NnRobustModelAdapter(object):
    """Expose a model through the contract expected by nn_robust_attacks."""

    def __init__(
        self,
        model: Any,
        *,
        image_shape: Tuple[int, int, int],
        num_labels: int,
        input_shift: float = 0.0,
    ) -> None:
        if len(image_shape) != 3:
            raise ValueError("nn_robust_attacks requires HWC image_shape.")
        if image_shape[0] != image_shape[1]:
            raise ValueError("nn_robust_attacks requires square images.")
        self._model = model
        self.model = _ShiftedPredictModel(model, input_shift) if input_shift else model
        self.image_size = int(image_shape[0])
        self.num_channels = int(image_shape[2])
        self.num_labels = int(num_labels)
        self.input_shift = float(input_shift)

    def predict(self, data: Any) -> Any:
        """Return pre-softmax logits for centered attack tensors."""
        shifted = data + self.input_shift if self.input_shift else data
        return _predict_with_model(self._model, shifted)


def _load_nn_robust_class(root: str, filename: str, class_name: str) -> Any:
    attack_root = resolve_project_path(root) or Path(str(root)).expanduser()
    attack_path = attack_root / filename
    if not attack_path.is_file():
        raise ImportError("Missing nn_robust_attacks {0}: {1}".format(filename, attack_path))

    patch_tensorflow_v1_symbols()
    spec = importlib.util.spec_from_file_location(
        "deepdetector_nn_robust_{0}".format(attack_path.stem),
        str(attack_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load nn_robust_attacks from {0}".format(attack_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise ImportError(
            "Missing nn_robust_attacks class {0}: {1}".format(class_name, attack_path)
        )
    return getattr(module, class_name)


def _load_nn_robust_carlini_l2(root: str) -> Any:
    return _load_nn_robust_class(root, "l2_attack.py", "CarliniL2")


def _load_nn_robust_carlini_l2_adaptive(root: str) -> Any:
    return _load_nn_robust_class(root, "l2_adaptive_attack.py", "CarliniL2Adaptive")


def _load_nn_robust_carlini_li(root: str) -> Any:
    return _load_nn_robust_class(root, "li_attack.py", "CarliniLi")


def _model_session(model: Any) -> Any:
    session = getattr(model, "sess", None) or getattr(model, "session", None)
    if session is None:
        raise NotImplementedError("nn_robust_attacks requires a model exposing sess or session.")
    return session


@contextmanager
def _session_graph_context(session: Any):
    graph = getattr(session, "graph", None)
    if graph is None or not hasattr(graph, "as_default"):
        yield
        return
    with graph.as_default():
        yield


def _one_hot(labels: np.ndarray, num_labels: int) -> np.ndarray:
    label_array = np.asarray(labels)
    if label_array.ndim == 2:
        if label_array.shape[1] != num_labels:
            raise ValueError("Expected one-hot labels with {0} classes.".format(num_labels))
        return label_array.astype(np.float32)
    label_int = label_array.astype(np.int64).reshape(-1)
    encoded = np.zeros((len(label_int), int(num_labels)), dtype=np.float32)
    encoded[np.arange(len(label_int)), label_int] = 1.0
    return encoded


def _adapter_for_images(
    model: Any,
    images: np.ndarray,
    *,
    input_shift: float = 0.0,
) -> NnRobustModelAdapter:
    image_array = np.asarray(images, dtype=np.float32)
    if image_array.ndim != 4:
        raise ValueError("nn_robust_attacks requires image batches.")
    return NnRobustModelAdapter(
        model,
        image_shape=tuple(int(value) for value in image_array.shape[1:]),
        num_labels=int(getattr(model, "num_labels", 10)),
        input_shift=float(input_shift),
    )


def _require_root(nn_robust_attacks_root: Optional[str]) -> str:
    if not nn_robust_attacks_root:
        raise ValueError("nn_robust_attacks_root is required for nn_robust CW attacks.")
    return str(nn_robust_attacks_root)


def generate_nn_robust_cw_l2_attack(
    model: Any,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    nn_robust_attacks_root: Optional[str] = None,
    kappa: float = 0.0,
    confidence: Optional[float] = None,
    batch_size: int = 1,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    binary_search_steps: int = 9,
    initial_const: float = 1e-3,
    abort_early: bool = True,
    targeted: bool = False,
    clip_min: float = -0.5,
    clip_max: float = 0.5,
    **_: Any,
) -> np.ndarray:
    """Generate CW-L2 examples using carlini/nn_robust_attacks."""
    image_array = np.asarray(images, dtype=np.float32)
    adapter = _adapter_for_images(model, image_array)
    one_hot_labels = _one_hot(np.asarray(labels), adapter.num_labels)
    session = _model_session(model)
    CarliniL2 = _load_nn_robust_carlini_l2(_require_root(nn_robust_attacks_root))

    with _session_graph_context(session):
        attack = CarliniL2(
            session,
            adapter,
            batch_size=int(batch_size),
            confidence=float(kappa if confidence is None else confidence),
            targeted=bool(targeted),
            learning_rate=float(learning_rate),
            binary_search_steps=int(binary_search_steps),
            max_iterations=int(max_iterations),
            abort_early=bool(abort_early),
            initial_const=float(initial_const),
            boxmin=float(clip_min),
            boxmax=float(clip_max),
        )
        adversarial = attack.attack(image_array, one_hot_labels)
    return np.clip(np.asarray(adversarial, dtype=np.float32), clip_min, clip_max)


def generate_nn_robust_cw_linf_attack(
    model: Any,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    nn_robust_attacks_root: Optional[str] = None,
    targeted: bool = False,
    learning_rate: float = 0.005,
    max_iterations: int = 1000,
    abort_early: bool = True,
    initial_const: float = 1e-5,
    largest_const: float = 20.0,
    reduce_const: bool = False,
    decrease_factor: float = 0.9,
    const_factor: float = 2.0,
    clip_min: float = -0.5,
    clip_max: float = 0.5,
    **_: Any,
) -> np.ndarray:
    """Generate CW-Linf examples using carlini/nn_robust_attacks."""
    image_array = np.asarray(images, dtype=np.float32)
    adapter = _adapter_for_images(model, image_array)
    one_hot_labels = _one_hot(np.asarray(labels), adapter.num_labels)
    session = _model_session(model)
    CarliniLi = _load_nn_robust_carlini_li(_require_root(nn_robust_attacks_root))

    with _session_graph_context(session):
        attack = CarliniLi(
            session,
            adapter,
            targeted=bool(targeted),
            learning_rate=float(learning_rate),
            max_iterations=int(max_iterations),
            abort_early=bool(abort_early),
            initial_const=float(initial_const),
            largest_const=float(largest_const),
            reduce_const=bool(reduce_const),
            decrease_factor=float(decrease_factor),
            const_factor=float(const_factor),
        )
        adversarial = attack.attack(image_array, one_hot_labels)
    return np.clip(np.asarray(adversarial, dtype=np.float32), clip_min, clip_max)
