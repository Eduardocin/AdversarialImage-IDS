"""Common attack registry and dispatcher."""

from __future__ import annotations

from typing import Any, Callable

from deepdetector.attacks.cw_l2 import generate_cw_l2_attack
from deepdetector.attacks.cw_linf import generate_cw_linf_attack
from deepdetector.attacks.deepfool import generate_deepfool
from deepdetector.attacks.fgsm import generate_fgsm_examples


AttackGenerator = Callable[..., Any]


ATTACK_REGISTRY: dict[str, AttackGenerator] = {
    "cw_l2": generate_cw_l2_attack,
    "cw_linf": generate_cw_linf_attack,
    "fgsm": generate_fgsm_examples,
    "deepfool": generate_deepfool,
}


def generate_attack(name: str, **kwargs: Any) -> Any:
    """Dispatch an attack by registry name."""
    attack_name = str(name).strip().lower()
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError("Unknown attack: {0}".format(name))
    return ATTACK_REGISTRY[attack_name](**kwargs)
