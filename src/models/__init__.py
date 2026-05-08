"""Model definitions for the modern DeepDetector reproduction."""

from src.models.target_models import (
    ImageNetBackboneName,
    load_torchscript_target,
    load_torchvision_imagenet_target,
)

__all__ = [
    "ImageNetBackboneName",
    "load_torchscript_target",
    "load_torchvision_imagenet_target",
]
