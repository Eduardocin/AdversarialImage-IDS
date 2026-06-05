"""TensorFlow 1.x/Keras MNIST CNN helpers."""

from __future__ import print_function

import glob
import os
from typing import Any, Optional, Tuple


def _set_keras_channels_last(keras_backend: Any) -> None:
    """Configure Keras for TensorFlow-style NHWC image tensors."""
    if hasattr(keras_backend, "set_image_dim_ordering"):
        keras_backend.set_image_dim_ordering("tf")
        return
    if hasattr(keras_backend, "set_image_data_format"):
        keras_backend.set_image_data_format("channels_last")
        return
    raise AttributeError(
        "Keras backend does not expose image data format configuration helpers."
    )


def _disable_tensorflow_eager(tf_module: Any) -> None:
    """Disable eager execution for TF1-style MNIST graph code."""
    try:
        tf_module.compat.v1.disable_eager_execution()
    except Exception:
        pass


def patch_tensorflow_v1_symbols(tf_module: Any) -> None:
    """Expose TF1 symbols used by legacy CleverHans/Keras code on TF2."""
    _disable_tensorflow_eager(tf_module)
    names = [
        "GraphKeys",
        "get_collection",
        "add_to_collection",
        "variable_scope",
        "get_variable",
        "global_variables",
        "local_variables",
        "trainable_variables",
        "global_variables_initializer",
        "variables_initializer",
        "placeholder",
        "Session",
        "ConfigProto",
        "reset_default_graph",
        "get_default_graph",
        "name_scope",
        "control_dependencies",
        "assign",
        "assign_add",
        "gradients",
        "random_uniform",
        "set_random_seed",
    ]
    for name in names:
        if not hasattr(tf_module, name) and hasattr(tf_module.compat.v1, name):
            setattr(tf_module, name, getattr(tf_module.compat.v1, name))

    try:
        if not hasattr(tf_module.train, "AdamOptimizer"):
            tf_module.train.AdamOptimizer = tf_module.compat.v1.train.AdamOptimizer
    except Exception:
        pass


def make_convolution2d(
    convolution2d: Any,
    filters: int,
    kernel_size: Tuple[int, int],
    *,
    padding: str,
    input_shape: Optional[Tuple[int, int, int]] = None,
) -> Any:
    """Create a Conv2D layer across Keras 1 and Keras 2 argument names."""
    kwargs = {"padding": padding}
    if input_shape is not None:
        kwargs["input_shape"] = input_shape
    try:
        return convolution2d(filters, kernel_size, **kwargs)
    except TypeError as exc:
        if "Keyword argument not understood" not in str(exc) and "missing" not in str(exc):
            raise

    legacy_kwargs = {"border_mode": padding}
    if input_shape is not None:
        legacy_kwargs["input_shape"] = input_shape
    return convolution2d(filters, kernel_size[0], kernel_size[1], **legacy_kwargs)


def create_tf_session(allow_growth: bool = True) -> Any:
    """Create a TensorFlow session and attach it to the Keras backend."""
    import tensorflow as tf
    from keras import backend as keras_backend

    patch_tensorflow_v1_symbols(tf)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = allow_growth

    sess = tf.compat.v1.Session(config=config)
    keras_backend.set_session(sess)
    _set_keras_channels_last(keras_backend)
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
