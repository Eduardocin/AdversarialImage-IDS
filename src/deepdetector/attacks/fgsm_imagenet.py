"""FGSM generation for ImageNet-scale TensorFlow graphs."""

from __future__ import print_function

import logging
from typing import Any, List

import numpy as np


logger = logging.getLogger(__name__)


def _manual_fgsm_tensor(
    x_placeholder: Any,
    logits: Any,
    eps: float,
    clip_min: float,
    clip_max: float,
) -> Any:
    """Build a TF1 FGSM tensor using predicted labels."""
    import tensorflow as tf

    predicted_labels = tf.stop_gradient(
        tf.cast(
            tf.equal(logits, tf.reduce_max(logits, axis=1, keepdims=True)),
            tf.float32,
        )
    )
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=predicted_labels,
        logits=logits,
    )
    gradient = tf.compat.v1.gradients(loss, x_placeholder)[0]
    adv_tensor = x_placeholder + eps * tf.sign(gradient)
    return tf.clip_by_value(adv_tensor, clip_min, clip_max)


def _cleverhans_fgsm_tensor(
    sess: Any,
    x_placeholder: Any,
    logits: Any,
    eps: float,
    clip_min: float,
    clip_max: float,
) -> Any:
    """Build a CleverHans FastGradientMethod tensor."""
    from cleverhans.attacks import FastGradientMethod
    from cleverhans.model import CallableModelWrapper

    model = CallableModelWrapper(lambda _: logits, "logits")
    attack = FastGradientMethod(model, sess=sess)
    return attack.generate(
        x_placeholder,
        eps=eps,
        clip_min=clip_min,
        clip_max=clip_max,
    )


def _build_fgsm_tensor(
    sess: Any,
    x_placeholder: Any,
    logits: Any,
    eps: float,
    clip_min: float,
    clip_max: float,
) -> Any:
    """Build the FGSM tensor, using CleverHans when its import path is usable."""
    try:
        return _cleverhans_fgsm_tensor(
            sess=sess,
            x_placeholder=x_placeholder,
            logits=logits,
            eps=eps,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    except ImportError as exc:
        logger.warning(
            "CleverHans FastGradientMethod import failed; using equivalent TF1 graph: %s",
            exc,
        )
        return _manual_fgsm_tensor(
            x_placeholder=x_placeholder,
            logits=logits,
            eps=eps,
            clip_min=clip_min,
            clip_max=clip_max,
        )


def generate_fgsm_imagenet(
    sess: Any,
    x_placeholder: Any,
    logits: Any,
    images: np.ndarray,
    eps: float = 4.0 / 255.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    batch_size: int = 32,
) -> np.ndarray:
    """Generate ImageNet FGSM examples in batches.

    ImageNet inputs use normalized RGB pixels, so the default perturbation is
    ``4/255``. This is smaller than the MNIST default because a unit interval
    ImageNet pixel represents an 8-bit color channel.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if clip_min >= clip_max:
        raise ValueError("clip_min must be smaller than clip_max.")

    image_array = np.asarray(images, dtype=np.float32)
    if image_array.ndim != 4:
        raise ValueError("images must have shape (N, H, W, C).")

    adv_tensor = _build_fgsm_tensor(
        sess=sess,
        x_placeholder=x_placeholder,
        logits=logits,
        eps=eps,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    batches: List[np.ndarray] = []
    for start in range(0, len(image_array), batch_size):
        batch = image_array[start : start + batch_size]
        adv_batch = sess.run(adv_tensor, feed_dict={x_placeholder: batch})
        batches.append(np.asarray(adv_batch, dtype=np.float32))

    if not batches:
        return np.empty_like(image_array, dtype=np.float32)

    adv_examples = np.concatenate(batches, axis=0)
    return np.clip(adv_examples, clip_min, clip_max).astype(np.float32)
