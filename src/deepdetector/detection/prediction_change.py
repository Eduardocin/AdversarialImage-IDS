"""Prediction-change detector used by the DeepDetector-style pipeline."""

from __future__ import print_function

from typing import Any, Callable, Dict

import numpy as np


FilterFn = Callable[[np.ndarray], np.ndarray]


class PredictionChangeDetector(object):
    """Detect adversarial examples by comparing predictions before/after filtering."""

    def __init__(
        self,
        sess: Any,
        x_placeholder: Any,
        predictions: Any,
        filter_fn: FilterFn,
    ) -> None:
        self.sess = sess
        self.x_placeholder = x_placeholder
        self.predictions = predictions
        self.filter_fn = filter_fn

    @staticmethod
    def _as_batch(image: np.ndarray) -> np.ndarray:
        """Return one image as a batch accepted by CleverHans model_argmax."""
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim == 2:
            image_array = image_array.reshape((28, 28, 1))
        if image_array.ndim == 3:
            return image_array.reshape((1,) + image_array.shape)
        if image_array.ndim == 4 and image_array.shape[0] == 1:
            return image_array
        raise ValueError("Expected a single image with shape HWC or 1xHWC.")

    @staticmethod
    def _label_to_int(label: Any) -> int:
        """Convert an integer or one-hot label to a Python int."""
        label_array = np.asarray(label)
        if label_array.ndim == 0:
            return int(label_array)
        return int(np.argmax(label_array))

    def predict_label(self, image: np.ndarray) -> int:
        """Predict the class label for one image using CleverHans model_argmax."""
        from cleverhans.utils_tf import model_argmax

        batch = self._as_batch(image)
        labels = model_argmax(self.sess, self.x_placeholder, self.predictions, batch)
        return int(np.asarray(labels).reshape(-1)[0])

    def detect(self, image: np.ndarray) -> bool:
        """Return True when filtering changes the model prediction."""
        original_pred = self.predict_label(image)
        filtered_image = self.filter_fn(np.asarray(image, dtype=np.float32))
        filtered_pred = self.predict_label(filtered_image)
        return bool(filtered_pred != original_pred)

    def detect_pair(
        self,
        clean_image: np.ndarray,
        adv_image: np.ndarray,
        true_label: Any,
    ) -> Dict[str, Any]:
        """Evaluate detector decisions for a clean/adversarial image pair."""
        true_label_int = self._label_to_int(true_label)
        clean_pred = self.predict_label(clean_image)
        adv_pred = self.predict_label(adv_image)

        filtered_clean = self.filter_fn(np.asarray(clean_image, dtype=np.float32))
        filtered_adv = self.filter_fn(np.asarray(adv_image, dtype=np.float32))
        filtered_clean_pred = self.predict_label(filtered_clean)
        filtered_adv_pred = self.predict_label(filtered_adv)

        detected = filtered_adv_pred != adv_pred
        false_positive = filtered_clean_pred != clean_pred
        corrected = filtered_adv_pred == true_label_int

        return {
            "true_label": true_label_int,
            "clean_pred": int(clean_pred),
            "adv_pred": int(adv_pred),
            "filtered_clean_pred": int(filtered_clean_pred),
            "filtered_adv_pred": int(filtered_adv_pred),
            "detected": bool(detected),
            "false_positive": bool(false_positive),
            "corrected": bool(corrected),
        }
