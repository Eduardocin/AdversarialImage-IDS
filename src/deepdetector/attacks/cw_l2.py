"""CW L2 attack generation for TF1/Keras MNIST models."""

from __future__ import print_function

from contextlib import contextmanager
from typing import Any, Callable, Optional, Tuple

import numpy as np


def _patch_tensorflow_v1_symbols() -> None:
    """Expose TF1 symbols expected by CleverHans when running on TF2."""
    try:
        import tensorflow as tf
    except Exception:
        return

    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass

    names = [
        "GraphKeys",
        "get_collection",
        "add_to_collection",
        "variable_scope",
        "get_variable",
        "global_variables",
        "local_variables",
        "trainable_variables",
        "global_variables_initializer",
        "variables_initializer",
        "placeholder",
        "Session",
        "ConfigProto",
        "reset_default_graph",
        "get_default_graph",
        "name_scope",
        "control_dependencies",
        "assign",
        "assign_add",
        "gradients",
    ]

    for name in names:
        if not hasattr(tf, name) and hasattr(tf.compat.v1, name):
            setattr(tf, name, getattr(tf.compat.v1, name))


def _one_hot_to_int(labels: np.ndarray) -> np.ndarray:
    """Convert one-hot labels to integer labels."""
    label_array = np.asarray(labels)
    if label_array.ndim == 1:
        return label_array.astype(np.int64)
    return np.argmax(label_array, axis=1).astype(np.int64)


def _one_hot(labels: np.ndarray, nb_classes: int = 10) -> np.ndarray:
    """Return one-hot labels as float32."""
    label_int = _one_hot_to_int(labels)
    encoded = np.zeros((len(label_int), nb_classes), dtype=np.float32)
    encoded[np.arange(len(label_int)), label_int] = 1.0
    return encoded


def _pad_to_batch(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pad arrays to a full batch size and return the pad count."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    pad = (-len(images)) % batch_size
    if pad == 0:
        return images, labels, 0
    pad_images = np.repeat(images[-1:], pad, axis=0)
    pad_labels = np.repeat(labels[-1:], pad, axis=0)
    return (
        np.concatenate([images, pad_images], axis=0),
        np.concatenate([labels, pad_labels], axis=0),
        pad,
    )


def _load_cwl2() -> Any:
    """Load the CWL2 class without importing optional attack dependencies."""
    _patch_tensorflow_v1_symbols()
    try:
        from cleverhans.attacks.carlini_wagner_l2 import CWL2

        return CWL2
    except Exception:
        pass

    try:
        import importlib.util
        import sys
        import types
        from pathlib import Path

        import cleverhans

        attacks_dir = Path(cleverhans.__file__).resolve().parent / "attacks"
        package_name = "cleverhans.attacks"
        package = sys.modules.get(package_name)
        if package is None or not hasattr(package, "__path__"):
            package = types.ModuleType(package_name)
            package.__path__ = [str(attacks_dir)]
            sys.modules[package_name] = package

        def load_module(name: str, path: Path) -> Any:
            if name in sys.modules:
                return sys.modules[name]
            spec = importlib.util.spec_from_file_location(name, str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        load_module("cleverhans.attacks.attack", attacks_dir / "attack.py")
        cw_module = load_module(
            "cleverhans.attacks.carlini_wagner_l2",
            attacks_dir / "carlini_wagner_l2.py",
        )
        return cw_module.CWL2
    except Exception as exc:
        raise ImportError("CWL2 attack is unavailable in this CleverHans stack.") from exc


def generate_cw_l2_examples(
    sess: Any,
    model: Any,
    x_placeholder: Any,
    images: np.ndarray,
    labels: np.ndarray,
    confidence: float,
    batch_size: int,
    max_iterations: int,
    learning_rate: float,
    binary_search_steps: int,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    initial_const: float = 1e-3,
    abort_early: bool = True,
    targeted: bool = False,
    nb_classes: int = 10,
    image_shape: Optional[Tuple[int, ...]] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> np.ndarray:
    """Generate CW L2 adversarial examples.

    CW L2 optimizes a margin loss and an L2 distortion penalty. ``confidence``
    controls the classification margin used by the optimizer.
    """
    CWL2 = _load_cwl2()

    class LogitsWrapper(object):
        """Minimal wrapper exposing logits for CWL2."""

        def __init__(self, keras_model: Any) -> None:
            self._model = keras_model

        def get_logits(self, x: Any) -> Any:
            return self._model(x)

    image_array = np.asarray(images, dtype=np.float32)
    expected_shape = tuple(image_shape or (28, 28, 1))
    if image_array.ndim != 4 or tuple(image_array.shape[1:]) != expected_shape:
        raise ValueError("Expected images with shape (N, {0}).".format(expected_shape))

    try:
        from keras import backend as K

        if hasattr(K, "set_learning_phase"):
            K.set_learning_phase(0)
    except Exception:
        pass

    label_array = _one_hot(labels, nb_classes=int(nb_classes))
    wrapper = LogitsWrapper(model)
    batch_size = int(batch_size)
    attack = CWL2(
        sess,
        wrapper,
        batch_size,
        float(confidence),
        bool(targeted),
        float(learning_rate),
        int(binary_search_steps),
        int(max_iterations),
        bool(abort_early),
        float(initial_const),
        float(clip_min),
        float(clip_max),
        int(label_array.shape[1]),
        tuple(image_array.shape[1:]),
    )

    adv_batches = []
    total = len(image_array)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_images = image_array[start:end]
        batch_labels = label_array[start:end]
        padded_images, padded_labels, pad = _pad_to_batch(
            batch_images,
            batch_labels,
            batch_size,
        )
        adv_padded = attack.attack(padded_images, padded_labels)
        if pad:
            adv_batch = adv_padded[:-pad]
        else:
            adv_batch = adv_padded
        adv_batches.append(adv_batch)
        if progress_callback is not None:
            progress_callback(start, end, total)

    adv_examples = np.concatenate(adv_batches, axis=0)

    return np.clip(adv_examples, clip_min, clip_max).astype(np.float32)


def _model_session(model: Any) -> Any:
    session = getattr(model, "sess", None) or getattr(model, "session", None)
    if session is None:
        raise NotImplementedError("CW L2 requires a model exposing sess or session.")
    return session


@contextmanager
def _session_graph_context(session: Any):
    graph = getattr(session, "graph", None)
    if graph is None or not hasattr(graph, "as_default"):
        yield
        return
    with graph.as_default():
        yield


def generate_cw_l2_attack(
    model: Any,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    kappa: float = 0.0,
    confidence: Optional[float] = None,
    batch_size: int = 1,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    binary_search_steps: int = 9,
    clip_min: float = -0.5,
    clip_max: float = 0.5,
    initial_const: float = 1e-3,
    abort_early: bool = True,
    targeted: bool = False,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> np.ndarray:
    """Registry-compatible CW L2 attack for Table 10 model wrappers."""
    image_array = np.asarray(images, dtype=np.float32)
    session = _model_session(model)
    with _session_graph_context(session):
        return generate_cw_l2_examples(
            sess=session,
            model=model,
            x_placeholder=getattr(model, "input_tensor", None),
            images=image_array,
            labels=np.asarray(labels),
            confidence=float(kappa if confidence is None else confidence),
            batch_size=int(batch_size),
            max_iterations=int(max_iterations),
            learning_rate=float(learning_rate),
            binary_search_steps=int(binary_search_steps),
            clip_min=float(clip_min),
            clip_max=float(clip_max),
            initial_const=float(initial_const),
            abort_early=bool(abort_early),
            targeted=bool(targeted),
            nb_classes=int(getattr(model, "num_labels", 10)),
            image_shape=tuple(image_array.shape[1:]),
            progress_callback=progress_callback,
        )
