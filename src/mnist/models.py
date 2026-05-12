"""TensorFlow/Keras MNIST classifiers used by the replication."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple


def build_m1_model() -> Any:
    """Build the CleverHans MNIST CNN used by the FGSM reference."""

    from cleverhans.utils_keras import cnn_model

    return cnn_model()


def build_m2_model(input_shape: Tuple[int, int, int] = (28, 28, 1)) -> Any:
    """Build Carlini's MNIST M2 architecture."""

    from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from keras.models import Sequential

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(Dense(10))
    return model


def build_m2_inference_model(weights_path: Path, input_shape: Tuple[int, int, int] = (28, 28, 1)) -> Any:
    """Build M2 without dropout and load Carlini-compatible weights."""

    from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
    from keras.models import Sequential

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.load_weights(str(weights_path))
    return model


class CarliniModelWrapper:
    """Adapter exposing the interface expected by nn_robust_attacks."""

    num_channels = 1
    image_size = 28
    num_labels = 10

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, data: Any) -> Any:
        """Return logits for Carlini's attack code."""

        return self.model(data)


def predict_label(model: Any, image: Any) -> int:
    """Predict a single MNIST label from a Keras model."""

    import numpy as np

    image_batch = np.reshape(image, (1, 28, 28, 1))
    logits = np.squeeze(model.predict(image_batch))
    return int(logits.argmax())


def predict_confidence(model: Any, image: Any) -> float:
    """Return the highest softmax probability for a single MNIST image."""

    import numpy as np
    import tensorflow as tf
    from keras import backend as keras_backend

    image_batch = np.reshape(image, (1, 28, 28, 1))
    logits = np.squeeze(model.predict(image_batch))
    return float(np.max(keras_backend.get_session().run(tf.nn.softmax(logits))))
