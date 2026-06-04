"""Load, generate, and persist adversarial images for article runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from deepdetector.attacks.fgsm_imagenet import generate_fgsm_imagenet
from deepdetector.evaluation.imagenet_utils import epsilon_255
from deepdetector.io.paths import resolve_project_path
from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper


logger = logging.getLogger(__name__)


def load_adversarial_images(
    path: Path,
    expected_shape: Tuple[int, ...],
    selected_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Load adversarial images and validate their shape."""
    adv_images = np.load(str(path)).astype(np.float32)
    if (
        selected_indices is not None
        and adv_images.shape != expected_shape
        and adv_images.ndim == len(expected_shape)
        and adv_images.shape[1:] == expected_shape[1:]
        and len(selected_indices) > 0
        and int(np.max(selected_indices)) < len(adv_images)
    ):
        adv_images = adv_images[selected_indices]

    if adv_images.shape != expected_shape:
        raise ValueError(
            "Expected adversarial array shape {0}, got {1}.".format(
                expected_shape,
                adv_images.shape,
            )
        )
    return adv_images


def generate_adversarial_images(
    config: Dict[str, Any],
    model: GoogLeNetCaffeWrapper,
    images: np.ndarray,
) -> Optional[np.ndarray]:
    """Generate FGSM examples with the shared Caffe article implementation."""
    result = generate_fgsm_imagenet(
        model=model,
        images=images,
        labels=None,
        epsilon_255=epsilon_255(config),
        skip_wrong_baseline=False,
        clip_min=0.0,
        clip_max=255.0,
    )
    return result.adversarial_images


def adversarial_images_for_run(
    config: Dict[str, Any],
    model: GoogLeNetCaffeWrapper,
    images: np.ndarray,
    override_path: Optional[str] = None,
    selected_indices: Optional[np.ndarray] = None,
    project_root: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Load configured adversarial images or generate them for the run."""
    attack_config = config.get("attack", {})
    adv_path = resolve_project_path(
        override_path or attack_config.get("adversarial_path"),
        project_root=project_root,
    )
    if adv_path is not None and adv_path.is_file():
        logger.info("Loaded ImageNet adversarial cache: %s", adv_path)
        return load_adversarial_images(
            adv_path,
            images.shape,
            selected_indices=selected_indices,
        )

    adv_images = generate_adversarial_images(config, model, images)
    if adv_images is None:
        return None

    save_path = resolve_project_path(
        attack_config.get("save_adversarial_path"),
        project_root=project_root,
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), adv_images)
        logger.info("Wrote ImageNet adversarial cache: %s", save_path)
    return adv_images
