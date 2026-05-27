"""TensorFlow 1.x/Keras MNIST CNN helpers."""

from __future__ import print_function

import glob
import os
from typing import Any, Optional, Tuple


def create_tf_session(allow_growth: bool = True) -> Any:
    """Create a TensorFlow session and attach it to the Keras backend."""
    import tensorflow as tf
    from keras import backend as keras_backend

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = allow_growth

    sess = tf.compat.v1.Session(config=config)
    keras_backend.set_session(sess)
    keras_backend.set_image_dim_ordering("tf")
    return sess


def build_mnist_model(x_placeholder: Any) -> Tuple[Any, Any]:
    """Build the CleverHans/Keras MNIST CNN and return predictions."""
    from cleverhans.utils_keras import cnn_model

    model = cnn_model()
    predictions = model(x_placeholder)
    return model, predictions


def checkpoint_path(train_dir: str, filename: str) -> str:
    """Return the base checkpoint path used by TensorFlow Saver."""
    return os.path.join(train_dir, filename)


def latest_checkpoint(train_dir: str) -> Optional[str]:
    """Return the latest checkpoint path in a training directory."""
    import tensorflow as tf

    checkpoint = tf.train.get_checkpoint_state(train_dir)
    if checkpoint is not None and _checkpoint_files_exist(checkpoint.model_checkpoint_path):
        return checkpoint.model_checkpoint_path

    local_checkpoint = checkpoint_path(train_dir, "mnist.ckpt")
    if _checkpoint_files_exist(local_checkpoint):
        return local_checkpoint

    if checkpoint is None:
        return None
    return checkpoint.model_checkpoint_path


def _checkpoint_files_exist(base_path: Optional[str]) -> bool:
    """Return whether a TensorFlow Saver checkpoint base path is usable."""
    if not base_path:
        return False
    return os.path.isfile(base_path + ".index") and bool(glob.glob(base_path + ".data-*"))


def save_mnist_model(sess: Any, train_dir: str, filename: str) -> str:
    """Save graph variables with a TensorFlow 1.x saver."""
    import tensorflow as tf

    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)

    saver = tf.compat.v1.train.Saver()
    save_path = saver.save(sess, checkpoint_path(train_dir, filename))
    return save_path


def load_mnist_model(sess: Any, train_dir: str) -> Optional[str]:
    """Load the latest checkpoint when one exists."""
    import tensorflow as tf

    checkpoint = latest_checkpoint(train_dir)
    if checkpoint is None:
        return None

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, checkpoint)
    return checkpoint
