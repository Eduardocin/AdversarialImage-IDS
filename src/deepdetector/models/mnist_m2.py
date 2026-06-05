"""TensorFlow 1.x/Keras M2 model used for MNIST CW experiments."""

from __future__ import print_function

import os
from typing import Any, Optional, Tuple

from deepdetector.models.mnist_cnn import make_convolution2d


def build_mnist_m2_model(x_placeholder: Any) -> Tuple[Any, Any]:
    """Build the MNIST M2 CNN and return ``(model, predictions)``.

    The model stacks convolutional blocks, dense layers, dropout, and a logits
    output layer:

    Conv(32) -> Conv(32) -> MaxPool -> Conv(64) -> Conv(64) -> MaxPool
    -> Flatten -> Dense(200) -> Dropout -> Dense(200) -> Dense(10).

    The final layer returns logits without a softmax activation.
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
            padding="valid",
            input_shape=(28, 28, 1),
        )
    )
    model.add(Activation("relu"))
    model.add(make_convolution2d(Convolution2D, 32, (3, 3), padding="valid"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(make_convolution2d(Convolution2D, 64, (3, 3), padding="valid"))
    model.add(Activation("relu"))
    model.add(make_convolution2d(Convolution2D, 64, (3, 3), padding="valid"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(Dense(10))

    predictions = model(x_placeholder)
    return model, predictions


def checkpoint_path(train_dir: str, filename: str) -> str:
    """Return the base checkpoint path used by TensorFlow Saver."""
    return os.path.join(train_dir, filename)


def latest_checkpoint(train_dir: str) -> Optional[str]:
    """Return the latest M2 checkpoint path in a training directory."""
    import tensorflow as tf

    checkpoint = tf.train.get_checkpoint_state(train_dir)
    if checkpoint is not None and _checkpoint_files_exist(checkpoint.model_checkpoint_path):
        return checkpoint.model_checkpoint_path

    local_checkpoint = checkpoint_path(train_dir, "mnist_m2.ckpt")
    if _checkpoint_files_exist(local_checkpoint):
        return local_checkpoint

    if checkpoint is None:
        return None
    return checkpoint.model_checkpoint_path


def _checkpoint_files_exist(base_path: Optional[str]) -> bool:
    """Return whether a TensorFlow Saver checkpoint base path is usable."""
    if not base_path:
        return False
    return os.path.isfile(base_path + ".index") and os.path.isfile(
        base_path + ".data-00000-of-00001"
    )


def save_mnist_m2_model(sess: Any, train_dir: str, filename: str) -> str:
    """Save M2 graph variables with a TensorFlow 1.x saver."""
    import tensorflow as tf

    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)

    saver = tf.compat.v1.train.Saver()
    return saver.save(sess, checkpoint_path(train_dir, filename))


def load_mnist_m2_model(sess: Any, train_dir: str) -> Optional[str]:
    """Load the latest M2 checkpoint when one exists."""
    import tensorflow as tf

    checkpoint = latest_checkpoint(train_dir)
    if checkpoint is None:
        return None

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, checkpoint)
    return checkpoint
