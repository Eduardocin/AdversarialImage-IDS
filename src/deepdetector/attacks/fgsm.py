"""FGSM attack generation with the legacy CleverHans API."""

from __future__ import print_function

from typing import Any

import numpy as np


def generate_fgsm_examples(
    sess: Any,
    model: Any,
    x_placeholder: Any,
    images: np.ndarray,
    eps: float = 0.2,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray:
    """Generate FGSM adversarial examples for a batch of MNIST images."""
    from cleverhans.attacks import FastGradientMethod
    from cleverhans.utils_keras import KerasModelWrapper

    wrapper = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrapper, sess=sess)
    adv_tensor = fgsm.generate(
        x_placeholder,
        eps=eps,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    adv_examples = sess.run(adv_tensor, feed_dict={x_placeholder: images})
    return np.clip(adv_examples, clip_min, clip_max).astype(np.float32)
