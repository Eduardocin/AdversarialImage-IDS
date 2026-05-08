"""Load fixed target models used by attacks and detectors."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

ImageNetBackboneName = Literal["alexnet", "googlenet", "inception_v3"]


def load_torchscript_target(model_path: str | Path, *, device=None):
    """Load a trained target model from a TorchScript file.

    This is the preferred MNIST path for the faithful reproduction: the detector
    receives an already trained classifier and never trains it internally.
    """

    import torch

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def load_torchvision_imagenet_target(
    name: ImageNetBackboneName,
    *,
    device=None,
):
    """Load a supported torchvision ImageNet model with default pretrained weights."""

    try:
        from torchvision.models import get_model, get_model_weights
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for supported pretrained ImageNet targets. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    weights_enum = get_model_weights(name)
    weights = weights_enum.DEFAULT
    model = get_model(name, weights=weights)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, weights.transforms()
