"""MNIST M2 training loop for the DeepDetector CW reproduction path."""

from __future__ import print_function

from typing import Any, Dict, Iterable

import numpy as np

from deepdetector.models.mnist_m2 import (
    latest_checkpoint,
    load_mnist_m2_model,
    save_mnist_m2_model,
)
from deepdetector.training.train_mnist import smooth_one_hot_labels


def train_or_load_mnist_m2_model(
    sess: Any,
    x: Any,
    y: Any,
    predictions: Any,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Train or restore the MNIST M2 CNN, then evaluate clean accuracy."""
    import math

    import tensorflow as tf
    from cleverhans.utils_tf import model_eval, model_loss

    def learning_phase_feed(value: int) -> Dict[Any, Any]:
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

    def epoch_progress(total_epochs: int) -> Iterable[int]:
        """Return a progress iterator over epochs."""
        try:
            from tqdm import trange

            return trange(total_epochs, desc="M2 epochs", unit="epoch")
        except Exception:
            return range(total_epochs)

    train_dir = str(
        config.get(
            "train_dir",
            "results/mnist/m2_cw/clean_baseline/checkpoints",
        )
    )
    filename = str(config.get("filename", "mnist_m2.ckpt"))
    batch_size = int(config.get("batch_size", 128))
    nb_epochs = int(config.get("nb_epochs", config.get("epochs", 10)))
    learning_rate = float(config.get("learning_rate", 0.001))
    load_model = bool(config.get("load_model", False))
    label_smoothing = float(config.get("label_smoothing", 0.1))
    rng = config.get("rng")
    if rng is None:
        rng = np.random.RandomState([2017, 8, 30])

    eval_params = {"batch_size": batch_size}
    eval_feed = learning_phase_feed(0)

    def evaluate() -> float:
        accuracy = model_eval(
            sess,
            x,
            y,
            predictions,
            X_test,
            Y_test,
            args=eval_params,
            feed=eval_feed,
        )
        print("m2_clean_test_accuracy={0:.4f}".format(accuracy), flush=True)
        return float(accuracy)

    restored_checkpoint = None
    trained_from_scratch = True

    if load_model:
        restored_checkpoint = load_mnist_m2_model(sess, train_dir)
        if restored_checkpoint is not None:
            trained_from_scratch = False

    if trained_from_scratch:
        Y_train_smooth = smooth_one_hot_labels(Y_train, label_smoothing)
        train_feed = learning_phase_feed(1)
        loss = model_loss(y, predictions)
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        sess.run(tf.compat.v1.global_variables_initializer())

        num_samples = len(X_train)
        num_batches = int(math.ceil(float(num_samples) / batch_size))
        progress = epoch_progress(nb_epochs)
        for _ in progress:
            indices = np.arange(num_samples)
            rng.shuffle(indices)
            for batch in range(num_batches):
                start = batch * batch_size
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]
                feed_dict = {
                    x: X_train[batch_idx],
                    y: Y_train_smooth[batch_idx],
                }
                if train_feed:
                    feed_dict.update(train_feed)
                sess.run(train_step, feed_dict=feed_dict)
            accuracy = evaluate()
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(clean_accuracy="{0:.4f}".format(accuracy))
        checkpoint = save_mnist_m2_model(sess, train_dir, filename)
    else:
        checkpoint = restored_checkpoint

    clean_accuracy = evaluate()

    return {
        "experiment": "mnist_m2_clean_baseline",
        "dataset": "mnist",
        "model": "M2",
        "clean_accuracy": clean_accuracy,
        "checkpoint_path": checkpoint or latest_checkpoint(train_dir),
        "trained_from_scratch": trained_from_scratch,
        "train_dir": train_dir,
        "filename": filename,
        "epochs": nb_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "label_smoothing": label_smoothing,
    }
