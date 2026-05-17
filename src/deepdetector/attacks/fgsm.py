"""FGSM attack generation with CleverHans."""

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
    from cleverhans.utils_keras import KerasModelWrapper
    import tensorflow as tf

    wrapper = KerasModelWrapper(model)
    logits = wrapper.get_logits(x_placeholder)

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
    adv_tensor = tf.clip_by_value(adv_tensor, clip_min, clip_max)

    adv_examples = sess.run(adv_tensor, feed_dict={x_placeholder: images})
    return np.clip(adv_examples, clip_min, clip_max).astype(np.float32)
