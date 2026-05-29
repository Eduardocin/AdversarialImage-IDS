"""Adversarial attack integrations."""

from deepdetector.attacks.cw_l2 import generate_cw_l2_attack
from deepdetector.attacks.cw_linf import generate_cw_linf_attack
from deepdetector.attacks.deepfool import generate_deepfool
from deepdetector.attacks.registry import ATTACK_REGISTRY, generate_attack


__all__ = [
    "ATTACK_REGISTRY",
    "generate_attack",
    "generate_cw_l2_attack",
    "generate_cw_linf_attack",
    "generate_deepfool",
]
