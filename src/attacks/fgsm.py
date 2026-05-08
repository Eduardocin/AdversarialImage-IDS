"""PyTorch FGSM attack helper."""

from __future__ import annotations

from typing import Any


def fgsm_attack(
    model: Any,
    images: Any,
    labels: Any,
    epsilon: float,
    *,
    loss_fn: Any | None = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Any:
    """Generate FGSM adversarial examples for a PyTorch model.

    The implementation intentionally keeps a narrow interface: callers provide
    a model, input batch, labels, epsilon, and optional loss function. The model
    is not trained or mutated here.
    """

    import torch
    from torch import nn

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    if clip_min >= clip_max:
        raise ValueError("clip_min must be smaller than clip_max.")

    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    attack_images = images.detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(attack_images)
    loss = criterion(logits, labels)
    loss.backward()

    perturbation = epsilon * attack_images.grad.sign()
    adversarial_images = attack_images + perturbation
    return torch.clamp(adversarial_images, min=clip_min, max=clip_max).detach()
