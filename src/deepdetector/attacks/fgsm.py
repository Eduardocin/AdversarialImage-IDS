"""FGSM attack generation with CleverHans."""

from __future__ import print_function

from typing import Any, Dict

import numpy as np


def _keras_learning_phase_feed(value: int = 0) -> Dict[Any, Any]:
    """Return a feed dict for Keras learning phase when needed."""
    try:
        from keras import backend as K
    except Exception:
        return {}
    if not hasattr(K, "learning_phase"):
        return {}
    phase = K.learning_phase()
    if hasattr(phase, "op"):
        return {phase: value}
    return {}


def _fgsm_logits_from_model(
    model: Any,
    x_placeholder: Any,
    model_output: Any = None,
    output_is_probabilities: bool = False,
) -> Any:
    """Return logits for FGSM, with a fallback for legacy Keras Sequential models."""
    import tensorflow as tf

    if model_output is not None:
        if output_is_probabilities:
            return tf.math.log(tf.clip_by_value(model_output, 1e-12, 1.0))
        return model_output

    from cleverhans.utils_keras import KerasModelWrapper

    try:
        wrapper = KerasModelWrapper(model)
        return wrapper.get_logits(x_placeholder)
    except AttributeError as exc:
        if "'tuple' object has no attribute 'layer'" not in str(exc):
            raise

    probabilities = model(x_placeholder)
    return tf.math.log(tf.clip_by_value(probabilities, 1e-12, 1.0))


def generate_fgsm_examples(
    sess: Any,
    model: Any,
    x_placeholder: Any,
    images: np.ndarray,
    eps: float = 0.2,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    model_output: Any = None,
    output_is_probabilities: bool = False,
) -> np.ndarray:
    """Generate FGSM adversarial examples for a batch of MNIST images."""
    import tensorflow as tf

    logits = _fgsm_logits_from_model(
        model,
        x_placeholder,
        model_output=model_output,
        output_is_probabilities=output_is_probabilities,
    )

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

    feed_dict = {x_placeholder: images}
    feed_dict.update(_keras_learning_phase_feed(0))
    adv_examples = sess.run(adv_tensor, feed_dict=feed_dict)
    return np.clip(adv_examples, clip_min, clip_max).astype(np.float32)
