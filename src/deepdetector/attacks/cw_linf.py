"""Manual TF1 CW Linf-style attack generation for MNIST models."""

from __future__ import print_function

from typing import Any, Callable, Optional, Tuple

import numpy as np


def _one_hot_to_int(labels: np.ndarray) -> np.ndarray:
    """Convert one-hot labels to integer labels."""
    label_array = np.asarray(labels)
    if label_array.ndim == 1:
        return label_array.astype(np.int64)
    return np.argmax(label_array, axis=1).astype(np.int64)


def _one_hot(labels: np.ndarray, nb_classes: int = 10) -> np.ndarray:
    """Return one-hot labels as float32."""
    label_int = _one_hot_to_int(labels)
    encoded = np.zeros((len(label_int), nb_classes), dtype=np.float32)
    encoded[np.arange(len(label_int)), label_int] = 1.0
    return encoded


def _pad_to_batch(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pad arrays to a full batch size and return the pad count."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    pad = (-len(images)) % batch_size
    if pad == 0:
        return images, labels, 0
    pad_images = np.repeat(images[-1:], pad, axis=0)
    pad_labels = np.repeat(labels[-1:], pad, axis=0)
    return (
        np.concatenate([images, pad_images], axis=0),
        np.concatenate([labels, pad_labels], axis=0),
        pad,
    )


def generate_cw_linf_examples(
    sess: Any,
    model: Any,
    x_placeholder: Any,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    max_iterations: int,
    learning_rate: float,
    confidence: float = 0.0,
    initial_tau: float = 1.0,
    const: float = 1.0,
    tau_decay: float = 0.9,
    tau_check_interval: int = 50,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> np.ndarray:
    """Generate CW Linf-style adversarial examples with TensorFlow 1.x.

    The optimizer minimizes an untargeted margin loss while penalizing
    perturbation components that exceed a shrinking Linf threshold ``tau``.
    """
    import tensorflow as tf

    image_array = np.asarray(images, dtype=np.float32)
    if image_array.ndim != 4 or image_array.shape[1:] != (28, 28, 1):
        raise ValueError("Expected images with shape (N, 28, 28, 1).")

    label_array = _one_hot(labels)
    true_labels = _one_hot_to_int(labels)
    batch_size = int(batch_size)
    max_iterations = int(max_iterations)
    tau_check_interval = max(1, int(tau_check_interval))

    attack_x = tf.compat.v1.placeholder(
        tf.float32,
        shape=(batch_size, 28, 28, 1),
        name="cw_linf_x",
    )
    attack_y = tf.compat.v1.placeholder(
        tf.float32,
        shape=(batch_size, 10),
        name="cw_linf_y",
    )
    tau = tf.compat.v1.placeholder(tf.float32, shape=(), name="cw_linf_tau")

    delta = tf.Variable(
        tf.zeros((batch_size, 28, 28, 1), dtype=tf.float32),
        name="cw_linf_delta",
    )
    adv_tensor = tf.clip_by_value(attack_x + delta, clip_min, clip_max)
    clipped_delta = adv_tensor - attack_x
    scores = model(adv_tensor)

    real = tf.reduce_sum(attack_y * scores, axis=1)
    other = tf.reduce_max((1.0 - attack_y) * scores - attack_y * 10000.0, axis=1)
    margin_loss = tf.maximum(real - other + float(confidence), 0.0)
    tau_penalty = tf.reduce_sum(
        tf.maximum(tf.abs(clipped_delta) - tau, 0.0),
        axis=[1, 2, 3],
    )
    loss = tf.reduce_sum(float(const) * margin_loss + tau_penalty)

    optimizer = tf.compat.v1.train.AdamOptimizer(float(learning_rate))
    before_vars = set(tf.compat.v1.global_variables())
    train_op = optimizer.minimize(loss, var_list=[delta])
    optimizer_vars = [
        var for var in tf.compat.v1.global_variables() if var not in before_vars
    ]
    init_attack_vars = tf.compat.v1.variables_initializer([delta] + optimizer_vars)
    pred_tensor = tf.argmax(scores, axis=1)

    adv_batches = []
    total = len(image_array)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_images = image_array[start:end]
        batch_labels = label_array[start:end]
        batch_true = true_labels[start:end]
        padded_images, padded_labels, pad = _pad_to_batch(
            batch_images,
            batch_labels,
            batch_size,
        )

        sess.run(init_attack_vars)
        current_tau = float(initial_tau)
        best_adv = np.asarray(padded_images, dtype=np.float32).copy()
        best_linf = np.full((batch_size,), np.inf, dtype=np.float32)

        feed = {attack_x: padded_images, attack_y: padded_labels, tau: current_tau}
        for iteration in range(max_iterations):
            sess.run(train_op, feed_dict=feed)
            should_check = (
                iteration == max_iterations - 1
                or (iteration + 1) % tau_check_interval == 0
            )
            if not should_check:
                continue

            adv_value, pred_value = sess.run([adv_tensor, pred_tensor], feed_dict=feed)
            linf = np.max(
                np.abs(adv_value - padded_images).reshape((batch_size, -1)),
                axis=1,
            )
            success = pred_value.astype(np.int64) != _one_hot_to_int(padded_labels)
            improved = success & (linf < best_linf)
            best_adv[improved] = adv_value[improved]
            best_linf[improved] = linf[improved]

            if np.any(success):
                successful_linf = linf[success]
                current_tau = min(
                    current_tau * float(tau_decay),
                    float(np.max(successful_linf)) * float(tau_decay),
                )
                current_tau = max(current_tau, 0.0)
                feed[tau] = current_tau

        missing = np.isinf(best_linf)
        if np.any(missing):
            final_adv = sess.run(adv_tensor, feed_dict=feed)
            best_adv[missing] = final_adv[missing]

        if pad:
            adv_batch = best_adv[:-pad]
        else:
            adv_batch = best_adv
        adv_batches.append(adv_batch)

        del batch_true
        if progress_callback is not None:
            progress_callback(start, end, total)

    adv_examples = np.concatenate(adv_batches, axis=0)
    return np.clip(adv_examples, clip_min, clip_max).astype(np.float32)
