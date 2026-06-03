"""Adversarial attack integrations."""

from deepdetector.attacks.adaptive_cw_l2 import generate_adaptive_cw_l2_attack
from deepdetector.attacks.deepfool import generate_deepfool
from deepdetector.attacks.nn_robust import (
    generate_nn_robust_cw_l2_attack,
    generate_nn_robust_cw_linf_attack,
)
from deepdetector.attacks.registry import ATTACK_REGISTRY, generate_attack


__all__ = [
    "ATTACK_REGISTRY",
    "generate_attack",
    "generate_adaptive_cw_l2_attack",
    "generate_nn_robust_cw_l2_attack",
    "generate_nn_robust_cw_linf_attack",
    "generate_deepfool",
]
