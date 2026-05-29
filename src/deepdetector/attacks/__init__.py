"""Adversarial attack integrations."""

from deepdetector.attacks.deepfool import generate_deepfool
from deepdetector.attacks.registry import ATTACK_REGISTRY, generate_attack


__all__ = ["ATTACK_REGISTRY", "generate_attack", "generate_deepfool"]
