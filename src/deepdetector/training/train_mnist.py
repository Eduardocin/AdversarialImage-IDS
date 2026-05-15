"""MNIST training loop using CleverHans TensorFlow helpers."""

from __future__ import print_function

from typing import Any, Dict

import numpy as np

from deepdetector.models.mnist_cnn import (
    latest_checkpoint,
    load_mnist_model,
    save_mnist_model,
)


def smooth_one_hot_labels(labels: np.ndarray, smoothing: float = 0.1) -> np.ndarray:
    """Apply the label smoothing convention used by CleverHans examples."""
    if labels.ndim != 2:
        raise ValueError("Expected one-hot labels with shape (n_samples, n_classes).")
    if labels.shape[1] != 10:
        raise ValueError("Expected MNIST one-hot labels with 10 classes.")

    off_value = smoothing / float(labels.shape[1] - 1)
    on_value = 1.0 - smoothing
    return labels.clip(off_value, on_value)


def train_or_load_mnist_model(
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
    """Train or restore the MNIST CNN, then evaluate clean accuracy."""
    from cleverhans.utils_tf import model_eval, model_train

    train_dir = str(config.get("train_dir", "results/mnist/clean_baseline/checkpoints"))
    filename = str(config.get("filename", "mnist.ckpt"))
    batch_size = int(config.get("batch_size", 128))
    nb_epochs = int(config.get("nb_epochs", config.get("epochs", 6)))
    learning_rate = float(config.get("learning_rate", 0.001))
    load_model = bool(config.get("load_model", False))
    label_smoothing = float(config.get("label_smoothing", 0.1))

    eval_params = {"batch_size": batch_size}

    def evaluate() -> float:
        accuracy = model_eval(
            sess,
            x,
            y,
            predictions,
            X_test,
            Y_test,
            args=eval_params,
        )
        print("clean_test_accuracy={0:.4f}".format(accuracy), flush=True)
        return float(accuracy)

    restored_checkpoint = None
    trained_from_scratch = True

    if load_model:
        restored_checkpoint = load_mnist_model(sess, train_dir)
        if restored_checkpoint is not None:
            trained_from_scratch = False

    if trained_from_scratch:
        Y_train_smooth = smooth_one_hot_labels(Y_train, label_smoothing)
        train_params = {
            "nb_epochs": nb_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        rng = np.random.RandomState([2017, 8, 30])
        model_train(
            sess,
            x,
            y,
            predictions,
            X_train,
            Y_train_smooth,
            evaluate=evaluate,
            args=train_params,
            save=False,
            rng=rng,
        )
        checkpoint = save_mnist_model(sess, train_dir, filename)
    else:
        checkpoint = restored_checkpoint

    clean_accuracy = evaluate()

    return {
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
