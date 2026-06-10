"""Adapter for the original defense-aware CW-L2 attack implementation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from deepdetector.attacks.cw_l2_nn_robust import (
    NnRobustM2Adapter,
    _one_hot,
    _patch_keras_backend_symbols,
    _patch_tensorflow_v1_symbols,
    _resolve_nn_robust_root,
    _set_keras_inference_phase,
)


PredictFn = Callable[[np.ndarray], np.ndarray]
TransformFn = Callable[[np.ndarray], np.ndarray]


def _load_carlini_l2_adaptive(root: Path) -> Any:
    """Load CarliniL2Adaptive from a configured nn_robust_attacks root."""
    _patch_tensorflow_v1_symbols()
    _patch_keras_backend_symbols()
    attack_path = root / "l2_adaptive_attack.py"
    if not attack_path.is_file():
        raise ImportError(
            "Missing nn_robust_attacks l2_adaptive_attack.py: {0}".format(attack_path)
        )

    spec = importlib.util.spec_from_file_location(
        "deepdetector_nn_robust_l2_adaptive_attack",
        str(attack_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not load nn_robust_attacks from {0}".format(attack_path)
        )

    added_path = False
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        added_path = True
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if added_path:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass
    return module.CarliniL2Adaptive


def _validate_unit_images(images: np.ndarray) -> np.ndarray:
    image_array = np.asarray(images, dtype=np.float32)
    if image_array.ndim != 4 or tuple(image_array.shape[1:]) != (28, 28, 1):
        raise ValueError("original_adaptive_cw_l2 expects images with shape (N, 28, 28, 1).")
    if not np.all(np.isfinite(image_array)):
        raise ValueError("original_adaptive_cw_l2 received NaN or Inf image values.")
    min_value = float(np.min(image_array))
    max_value = float(np.max(image_array))
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise ValueError("original_adaptive_cw_l2 expects image values in [0, 1].")
    return np.clip(image_array, 0.0, 1.0).astype(np.float32)


def generate_original_adaptive_cw_l2_attack(
    *,
    graph: dict[str, Any],
    images: np.ndarray,
    labels: np.ndarray,
    model: Any | None = None,
    nn_robust_attacks_root: Any = "nn_robust_attacks",
    batch_size: int = 1,
    max_iterations: int = 2000,
    binary_search_steps: int = 5,
    initial_const: float = 1.0,
    learning_rate: float = 0.1,
    confidence: float = 0.0,
    targeted: bool = False,
    abort_early: bool = True,
    boxmin: float = -0.5,
    boxmax: float = 0.5,
    model_input_shift: float = 0.5,
    transform_fn: TransformFn | None = None,
    predict_fn: PredictFn | None = None,
    diagnostics: Any = None,
    diagnostic_context: dict[str, Any] | None = None,
    reproduction_mode: str = "article_faithful",
    use_internal_filter: bool = True,
    use_project_transform_fn: bool = False,
    **_: Any,
) -> np.ndarray:
    """Generate original defense-aware CW-L2 examples and return [0, 1] images."""
    if "sess" not in graph:
        raise ValueError("original_adaptive_cw_l2 requires graph['sess'].")
    model_obj = model if model is not None else graph.get("model")
    if model_obj is None:
        raise ValueError("original_adaptive_cw_l2 requires graph['model'] or model.")

    clean_images = _validate_unit_images(images)
    root = _resolve_nn_robust_root(nn_robust_attacks_root)
    CarliniL2Adaptive = _load_carlini_l2_adaptive(root)

    mode = str(reproduction_mode)
    effective_use_internal_filter = bool(use_internal_filter)
    effective_use_project_transform_fn = bool(use_project_transform_fn)
    effective_transform_fn = transform_fn
    if mode == "article_faithful":
        effective_transform_fn = None
        effective_use_internal_filter = True
        effective_use_project_transform_fn = False
    elif not effective_use_project_transform_fn:
        effective_transform_fn = None

    centered_images = clean_images - float(model_input_shift)
    _patch_tensorflow_v1_symbols()
    _set_keras_inference_phase()
    attack = CarliniL2Adaptive(
        graph["sess"],
        NnRobustM2Adapter(model_obj),
        batch_size=int(batch_size),
        max_iterations=int(max_iterations),
        confidence=float(confidence),
        binary_search_steps=int(binary_search_steps),
        initial_const=float(initial_const),
        learning_rate=float(learning_rate),
        targeted=bool(targeted),
        abort_early=bool(abort_early),
        boxmin=float(boxmin),
        boxmax=float(boxmax),
        predict_fn=predict_fn,
        transform_fn=effective_transform_fn,
        model_input_shift=float(model_input_shift),
        diagnostics=diagnostics,
        diagnostic_context=diagnostic_context,
        reproduction_mode=mode,
        use_internal_filter=effective_use_internal_filter,
        use_project_transform_fn=effective_use_project_transform_fn,
    )
    centered_adv = np.asarray(attack.attack(centered_images, _one_hot(labels)), dtype=np.float32)
    if centered_adv.shape != centered_images.shape:
        raise ValueError("original_adaptive_cw_l2 returned an unexpected shape.")
    return np.clip(centered_adv + float(model_input_shift), 0.0, 1.0).astype(np.float32)
