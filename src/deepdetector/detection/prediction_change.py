"""Prediction-change detector for normalized image classifiers."""

from __future__ import print_function

from typing import Any, Callable, Dict, Optional

import keras.backend as K
import numpy as np
import tensorflow as tf
from cleverhans.utils_tf import model_argmax


FilterFn = Callable[[np.ndarray], np.ndarray]
PredictLabelFn = Callable[[np.ndarray], np.ndarray]


class PredictionChangeDetector(object):
    """Detect adversarial images by comparing predictions after filtering."""

    def __init__(
        self,
        sess: tf.compat.v1.Session,
        x_placeholder: tf.Tensor,
        predictions: tf.Tensor,
        filter_fn: FilterFn,
        predict_label_fn: Optional[PredictLabelFn] = None,
    ) -> None:
        """Store prediction handles used by the detector."""
        self.sess = sess
        self.x_placeholder = x_placeholder
        self.predictions = predictions
        self.filter_fn = filter_fn
        self.predict_label_fn = predict_label_fn
        if sess is not None:
            K.set_session(sess)

    @staticmethod
    def _as_single_image_batch(image: np.ndarray) -> np.ndarray:
        """Return one image as a single-example float32 batch."""
        image_array = np.asarray(image, dtype=np.float32)

        if image_array.ndim == 3:
            return image_array.reshape((1,) + image_array.shape)
        if image_array.ndim == 4 and image_array.shape[0] == 1:
            return image_array

        raise ValueError("Expected image shape (H, W, C) or (1, H, W, C).")

    @staticmethod
    def _label_to_int(label: Any) -> int:
        """Convert an integer or one-hot encoded label to a Python int."""
        label_array = np.asarray(label)
        if label_array.ndim == 0:
            return int(label_array)
        return int(np.argmax(label_array))

    def predict_label(self, image: np.ndarray) -> int:
        """Predict a single image label."""
        image_batch = self._as_single_image_batch(image)
        if self.predict_label_fn is not None:
            labels = self.predict_label_fn(image_batch)
            return int(np.asarray(labels).reshape(-1)[0])

        if self.sess is None or self.x_placeholder is None or self.predictions is None:
            raise ValueError("TF graph handles or predict_label_fn must be provided.")

        label = model_argmax(
            self.sess,
            self.x_placeholder,
            self.predictions,
            image_batch,
        )
        return int(np.asarray(label).reshape(-1)[0])

    def detect(self, image: np.ndarray) -> bool:
        """Return True when the configured filter changes the prediction."""
        input_pred = self.predict_label(image)
        filtered_image = self.filter_fn(np.asarray(image, dtype=np.float32))
        filtered_pred = self.predict_label(filtered_image)
        return bool(filtered_pred != input_pred)

    def detect_pair(
        self,
        clean_image: np.ndarray,
        adv_image: np.ndarray,
        true_label: Any,
    ) -> Dict[str, Any]:
        """Evaluate clean/adversarial predictions before and after filtering."""
        true_label_int = self._label_to_int(true_label)
        clean_pred = self.predict_label(clean_image)
        adv_pred = self.predict_label(adv_image)

        filtered_clean = self.filter_fn(np.asarray(clean_image, dtype=np.float32))
        filtered_adv = self.filter_fn(np.asarray(adv_image, dtype=np.float32))
        filtered_clean_pred = self.predict_label(filtered_clean)
        filtered_adv_pred = self.predict_label(filtered_adv)

        return {
            "clean_pred": int(clean_pred),
            "adv_pred": int(adv_pred),
            "filtered_clean_pred": int(filtered_clean_pred),
            "filtered_adv_pred": int(filtered_adv_pred),
            "true_label": int(true_label_int),
            "detected": bool(filtered_adv_pred != adv_pred),
            "corrected": bool(filtered_adv_pred == true_label_int),
            "false_positive": bool(filtered_clean_pred != clean_pred),
        }
