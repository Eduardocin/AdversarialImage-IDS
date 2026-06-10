"Adapter for the article-faithful defense-aware CW-L2 attack implementation."""

from __future__ import annotations

import importlib.util
import inspect
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


_REQUIRED_ADAPTED_KWARGS = {
    "predict_fn",
    "transform_fn",
    "model_input_shift",
    "diagnostics",
    "diagnostic_context",
    "reproduction_mode",
    "use_internal_filter",
    "use_project_transform_fn",
}


def _load_carlini_l2_adaptive(root: Path) -> Any:
    """Load CarliniL2Adaptive from a configured nn_robust_attacks root.

    The original paper file does not know about the project's scale contract.
    Therefore, this adapter refuses to run if the local file is the unadapted
    original implementation. Running the unadapted file would silently call
    ``model.model.predict`` with centered [-0.5, 0.5] images and can produce
    misleading defense-aware rates.
    """
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
            "Could not load nn_robust_attacks l2_adaptive_attack.py from {0}".format(
                attack_path
            )
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

    if not hasattr(module, "CarliniL2Adaptive"):
        raise ImportError(
            "nn_robust_attacks l2_adaptive_attack.py does not expose CarliniL2Adaptive: "
            "{0}".format(attack_path)
        )

    CarliniL2Adaptive = module.CarliniL2Adaptive
    _assert_adapted_carlini_l2_adaptive(CarliniL2Adaptive, attack_path)
    return CarliniL2Adaptive


def _assert_adapted_carlini_l2_adaptive(CarliniL2Adaptive: Any, attack_path: Path) -> None:
    """Fail early if the loaded attack is the unadapted paper implementation."""
    try:
        signature = inspect.signature(CarliniL2Adaptive.__init__)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Could not inspect CarliniL2Adaptive.__init__ in {0}.".format(attack_path)
        ) from exc

    parameters = set(signature.parameters)
    missing = sorted(_REQUIRED_ADAPTED_KWARGS - parameters)
    if missing:
        raise RuntimeError(
            "The loaded l2_adaptive_attack.py appears to be the unadapted paper file. "
            "It is missing project-required constructor arguments: {0}. "
            "Replace {1} with the article-faithful adapted version before running "
            "original_adaptive_cw_l2.".format(", ".join(missing), attack_path)
        )


def _validate_unit_images(images: np.ndarray) -> np.ndarray:
    """Validate pipeline/model-space images in [0, 1]."""
    image_array = np.asarray(images, dtype=np.float32)
    if image_array.ndim != 4 or tuple(image_array.shape[1:]) != (28, 28, 1):
        raise ValueError(
            "original_adaptive_cw_l2 expects images with shape (N, 28, 28, 1)."
        )
    if not np.all(np.isfinite(image_array)):
        raise ValueError("original_adaptive_cw_l2 received NaN or Inf image values.")

    min_value = float(np.min(image_array))
    max_value = float(np.max(image_array))
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise ValueError(
            "original_adaptive_cw_l2 expects image values in [0, 1]; "
            "received min={0:.6f}, max={1:.6f}.".format(min_value, max_value)
        )

    return np.clip(image_array, 0.0, 1.0).astype(np.float32)


def _validate_scale_contract(
    *,
    boxmin: float,
    boxmax: float,
    model_input_shift: float,
) -> None:
    """Validate the expected article-faithful scale contract."""
    if abs(float(boxmin) + float(model_input_shift)) > 1e-6:
        raise ValueError(
            "Invalid scale contract: expected boxmin == -model_input_shift, "
            "got boxmin={0}, model_input_shift={1}.".format(boxmin, model_input_shift)
        )
    if abs(float(boxmax) - (1.0 - float(model_input_shift))) > 1e-6:
        raise ValueError(
            "Invalid scale contract: expected boxmax == 1 - model_input_shift, "
            "got boxmax={0}, model_input_shift={1}.".format(boxmax, model_input_shift)
        )


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
    """Generate defense-aware CW-L2 examples and return pipeline-space [0, 1] images.

    Input images are expected in [0, 1].  They are centered to [-0.5, 0.5]
    before entering the article-faithful CW implementation.  The returned
    adversarial images are converted back to [0, 1].
    """
    if "sess" not in graph:
        raise ValueError("original_adaptive_cw_l2 requires graph['sess'].")

    model_obj = model if model is not None else graph.get("model")
    if model_obj is None:
        raise ValueError("original_adaptive_cw_l2 requires graph['model'] or model.")

    _validate_scale_contract(
        boxmin=float(boxmin),
        boxmax=float(boxmax),
        model_input_shift=float(model_input_shift),
    )

    clean_images = _validate_unit_images(images)
    root = _resolve_nn_robust_root(nn_robust_attacks_root)
    CarliniL2Adaptive = _load_carlini_l2_adaptive(root)

    mode = str(reproduction_mode)
    effective_use_internal_filter = bool(use_internal_filter)
    effective_use_project_transform_fn = bool(use_project_transform_fn)
    effective_transform_fn = transform_fn

    if mode == "article_faithful":
        # The article-faithful mode must use the filter embedded in
        # l2_adaptive_attack.py.  The project transform_fn is reserved for the
        # project_integrated comparison mode.
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
        diagnostic_context=dict(diagnostic_context or {}),
        reproduction_mode=mode,
        use_internal_filter=effective_use_internal_filter,
        use_project_transform_fn=effective_use_project_transform_fn,
    )

    centered_adv = np.asarray(
        attack.attack(centered_images, _one_hot(labels)),
        dtype=np.float32,
    )
    if centered_adv.shape != centered_images.shape:
        raise ValueError(
            "original_adaptive_cw_l2 returned an unexpected shape: expected {0}, got {1}."
            .format(centered_images.shape, centered_adv.shape)
        )
    if not np.all(np.isfinite(centered_adv)):
        raise ValueError("original_adaptive_cw_l2 returned NaN or Inf values.")

    return np.clip(centered_adv + float(model_input_shift), 0.0, 1.0).astype(np.float32)
