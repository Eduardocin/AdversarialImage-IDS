"""TensorFlow 1.x/Keras M3 model for Fashion-MNIST experiments."""

from __future__ import print_function

import glob
import os
from typing import Any, Optional, Tuple

from deepdetector.models.mnist_cnn import make_convolution2d


CONV_DROPOUT_RATE = 0.25
DENSE_DROPOUT_RATE = 0.5


def build_mnist_m3_model(x_placeholder: Any) -> Tuple[Any, Any]:
    """Build the Fashion-MNIST M3 CNN and return ``(model, predictions)``.

    M3 is a compact CNN intended for Fashion-MNIST while preserving the
    legacy TensorFlow 1.x/Keras model contract used by the MNIST flows.
    """
    from keras.layers import Activation, Dense, Dropout, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.models import Sequential

    model = Sequential()
    model.add(
        make_convolution2d(
            Convolution2D,
            32,
            (3, 3),
            padding="same",
            input_shape=(28, 28, 1),
        )
    )
    model.add(Activation("relu"))
    model.add(make_convolution2d(Convolution2D, 32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(CONV_DROPOUT_RATE))

    model.add(make_convolution2d(Convolution2D, 64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(make_convolution2d(Convolution2D, 64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(CONV_DROPOUT_RATE))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(DENSE_DROPOUT_RATE))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    predictions = model(x_placeholder)
    return model, predictions


def checkpoint_path(train_dir: str, filename: str) -> str:
    """Return the base checkpoint path used by TensorFlow Saver."""
    return os.path.join(train_dir, filename)


def latest_checkpoint(train_dir: str) -> Optional[str]:
    """Return the latest M3 checkpoint path in a training directory."""
    import tensorflow as tf

    checkpoint = tf.train.get_checkpoint_state(train_dir)
    if checkpoint is not None and _checkpoint_files_exist(checkpoint.model_checkpoint_path):
        return checkpoint.model_checkpoint_path

    local_checkpoint = checkpoint_path(train_dir, "mnist_m3.ckpt")
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


def save_mnist_m3_model(sess: Any, train_dir: str, filename: str) -> str:
    """Save M3 graph variables with a TensorFlow 1.x saver."""
    import tensorflow as tf

    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)

    saver = tf.compat.v1.train.Saver()
    return saver.save(sess, checkpoint_path(train_dir, filename))


def load_mnist_m3_model(sess: Any, train_dir: str) -> Optional[str]:
    """Load the latest M3 checkpoint when one exists."""
    import tensorflow as tf

    checkpoint = latest_checkpoint(train_dir)
    if checkpoint is None:
        return None

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, checkpoint)
    return checkpoint
