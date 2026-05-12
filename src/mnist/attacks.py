"""Attack wrappers for the legacy MNIST replication stack."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import AttackConfig
from .data import clip_m1_images, clip_m2_images
from .models import CarliniModelWrapper


def generate_fgsm(
    sess: Any,
    x: Any,
    model: Any,
    samples: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Generate FGSM examples using CleverHans FastGradientMethod."""

    from cleverhans.attacks import FastGradientMethod
    from cleverhans.utils_keras import KerasModelWrapper

    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    adv_x = fgsm.generate(x, eps=eps, clip_min=0.0, clip_max=1.0)
    return clip_m1_images(sess.run(adv_x, feed_dict={x: samples}))


def _prepare_carlini_imports(nn_robust_attacks_root: Path) -> None:
    root_text = str(nn_robust_attacks_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def generate_cw_l2(
    sess: Any,
    keras_model: Any,
    inputs: np.ndarray,
    targets: np.ndarray,
    nn_robust_attacks_root: Path,
    config: AttackConfig,
) -> np.ndarray:
    """Generate C&W L2 examples using Carlini's original attack implementation."""

    _prepare_carlini_imports(nn_robust_attacks_root)
    from l2_attack import CarliniL2

    attack = CarliniL2(
        sess,
        CarliniModelWrapper(keras_model),
        batch_size=1,
        max_iterations=config.cw_l2_max_iterations,
        confidence=config.cw_l2_confidence,
        binary_search_steps=config.cw_l2_binary_search_steps,
        initial_const=config.cw_l2_initial_const,
        learning_rate=config.cw_l2_learning_rate,
        targeted=config.cw_l2_targeted,
    )
    return clip_m2_images(attack.attack(inputs, targets).reshape(inputs.shape))


def generate_cw_linf(
    sess: Any,
    keras_model: Any,
    inputs: np.ndarray,
    targets: np.ndarray,
    nn_robust_attacks_root: Path,
    config: AttackConfig,
) -> np.ndarray:
    """Generate C&W Linf examples using Carlini's original attack implementation."""

    _prepare_carlini_imports(nn_robust_attacks_root)
    from li_attack import CarliniLi

    attack = CarliniLi(
        sess,
        CarliniModelWrapper(keras_model),
        max_iterations=config.cw_linf_max_iterations,
        targeted=config.cw_linf_targeted,
    )
    return clip_m2_images(attack.attack(inputs, targets).reshape(inputs.shape))


def generate_deepfool(*args: Any, **kwargs: Any) -> np.ndarray:
    """DeepFool is intentionally deferred until a local MNIST reference exists."""

    raise NotImplementedError(
        "DeepFool MNIST is not implemented yet because the local DeepDetector "
        "reference tree does not include a MNIST DeepFool script."
    )

