"""Run the MNIST M1 classifier with the original CleverHans stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import numpy as np

from .config import M1Config, MnistExperimentConfig


@dataclass(frozen=True)
class M1RunResult:
    """Summary of an M1 train/load and clean evaluation run."""

    checkpoint_dir: Path
    checkpoint_path: Optional[str]
    clean_accuracy: float
    trained_from_scratch: bool
    epochs: int
    batch_size: int
    learning_rate: float


def run_m1(
    config: Optional[MnistExperimentConfig] = None,
    m1_config: Optional[M1Config] = None,
) -> M1RunResult:
    """Train or load M1 and evaluate clean MNIST accuracy."""

    started_at = time.time()
    cfg = config or MnistExperimentConfig()
    model_cfg = m1_config or cfg.m1
    checkpoint_dir = cfg.paths.m1_checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("[M1] starting project-owned M1 run", flush=True)
    print("[M1] checkpoint_dir={}".format(checkpoint_dir), flush=True)
    print(
        "[M1] config epochs={} batch_size={} learning_rate={} load_model={} filename={}".format(
            model_cfg.epochs,
            model_cfg.batch_size,
            model_cfg.learning_rate,
            model_cfg.load_model,
            model_cfg.filename,
        ),
        flush=True,
    )
    print(
        "[M1] split train={}..{} test={}..{}".format(
            cfg.splits.train_start,
            cfg.splits.train_end,
            cfg.splits.test_start,
            cfg.splits.test_end,
        ),
        flush=True,
    )

    import keras
    from keras import backend as keras_backend
    import tensorflow as tf
    from cleverhans.utils import AccuracyReport
    from cleverhans.utils_keras import cnn_model
    from cleverhans.utils_mnist import data_mnist
    from cleverhans.utils_tf import model_eval, model_train

    print("[M1] tensorflow_version={}".format(tf.__version__), flush=True)
    print("[M1] keras_version={}".format(keras.__version__), flush=True)
    print("[M1] configuring TensorFlow session", flush=True)
    keras_backend.set_image_dim_ordering("tf")
    tf.set_random_seed(1234)
    sess = tf.Session()
    keras_backend.set_session(sess)

    print("[M1] loading MNIST through CleverHans", flush=True)
    report = AccuracyReport()
    train_images, train_labels, test_images, test_labels = data_mnist(
        train_start=cfg.splits.train_start,
        train_end=cfg.splits.train_end,
        test_start=cfg.splits.test_start,
        test_end=cfg.splits.test_end,
    )
    print(
        "[M1] data train_images={} train_labels={} test_images={} test_labels={}".format(
            train_images.shape,
            train_labels.shape,
            test_images.shape,
            test_labels.shape,
        ),
        flush=True,
    )

    print("[M1] building CleverHans cnn_model graph", flush=True)
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    model = cnn_model()
    predictions = model(x)

    eval_params = {"batch_size": model_cfg.batch_size}

    def evaluate() -> float:
        accuracy = model_eval(
            sess,
            x,
            y,
            predictions,
            test_images,
            test_labels,
            args=eval_params,
        )
        report.clean_train_clean_eval = accuracy
        print("[M1] clean_test_accuracy=%0.4f" % accuracy, flush=True)
        return float(accuracy)

    train_params = {
        "nb_epochs": model_cfg.epochs,
        "batch_size": model_cfg.batch_size,
        "learning_rate": model_cfg.learning_rate,
        "train_dir": str(checkpoint_dir),
        "filename": model_cfg.filename,
    }

    checkpoint = tf.train.get_checkpoint_state(str(checkpoint_dir))
    checkpoint_path = False if checkpoint is None else checkpoint.model_checkpoint_path
    print("[M1] existing_checkpoint={}".format(checkpoint_path or None), flush=True)
    trained_from_scratch = True

    rng = np.random.RandomState([2017, 8, 30])
    if model_cfg.load_model and checkpoint_path:
        print("[M1] loading existing checkpoint", flush=True)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        trained_from_scratch = False
        print("[M1] model_loaded_from={}".format(checkpoint_path), flush=True)
        clean_accuracy = evaluate()
    else:
        print("[M1] training_from_scratch=True", flush=True)
        train_started_at = time.time()
        model_train(
            sess,
            x,
            y,
            predictions,
            train_images,
            train_labels,
            evaluate=evaluate,
            args=train_params,
            save=True,
            rng=rng,
        )
        print("[M1] training_finished_in_seconds={:.1f}".format(time.time() - train_started_at), flush=True)
        clean_accuracy = evaluate()
        checkpoint = tf.train.get_checkpoint_state(str(checkpoint_dir))
        checkpoint_path = None if checkpoint is None else checkpoint.model_checkpoint_path
        print("[M1] saved_checkpoint={}".format(checkpoint_path), flush=True)

    print("[M1] finished_in_seconds={:.1f}".format(time.time() - started_at), flush=True)

    return M1RunResult(
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=None if not checkpoint_path else str(checkpoint_path),
        clean_accuracy=clean_accuracy,
        trained_from_scratch=trained_from_scratch,
        epochs=model_cfg.epochs,
        batch_size=model_cfg.batch_size,
        learning_rate=model_cfg.learning_rate,
    )
