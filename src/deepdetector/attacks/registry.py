"""Common attack registry and dispatcher."""

from __future__ import annotations

from typing import Any, Callable

from deepdetector.attacks.adaptive_cw_l2 import generate_native_adaptive_cw_l2_attack
from deepdetector.attacks.deepfool import generate_deepfool
from deepdetector.attacks.fgsm import generate_fgsm_examples
from deepdetector.attacks.nn_robust import (
    generate_nn_robust_cw_l2_attack,
    generate_nn_robust_cw_linf_attack,
)


AttackGenerator = Callable[..., Any]


ATTACK_REGISTRY: dict[str, AttackGenerator] = {
    "native_adaptive_cw_l2": generate_native_adaptive_cw_l2_attack,
    "cw_l2_nn_robust": generate_nn_robust_cw_l2_attack,
    "cw_linf_nn_robust": generate_nn_robust_cw_linf_attack,
    "fgsm": generate_fgsm_examples,
    "deepfool": generate_deepfool,
}


def generate_attack(name: str, **kwargs: Any) -> Any:
    """Dispatch an attack by registry name."""
    attack_name = str(name).strip().lower()
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError("Unknown attack: {0}".format(name))
    return ATTACK_REGISTRY[attack_name](**kwargs)
