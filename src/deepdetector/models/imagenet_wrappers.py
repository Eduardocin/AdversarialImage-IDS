"""Backend-agnostic ImageNet model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class ImageNetModelWrapper(ABC):
    """Expose preprocessing and batch prediction for ImageNet classifiers."""

    @abstractmethod
    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """Convert normalized input images to the model input tensor."""

    @abstractmethod
    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """Return logits or probabilities with shape ``(N, num_classes)``."""

    def predict_label(self, images: np.ndarray) -> np.ndarray:
        """Return the top-1 class index for every image in a batch."""
        scores = self.predict_batch(images)
        return np.asarray(np.argmax(scores, axis=1), dtype=np.int32)


class GoogLeNetCaffeWrapper(ImageNetModelWrapper):
    """Run BVLC GoogLeNet through Caffe.

    This wrapper requires a working Caffe Python package and local GoogLeNet
    ``.prototxt`` and ``.caffemodel`` files. The standard BVLC GoogLeNet files
    are listed in the Caffe Model Zoo.
    """

    def __init__(
        self,
        model_dir: str,
        deploy_prototxt: str,
        caffemodel: str,
        mean_file: Optional[str] = None,
        use_gpu: bool = False,
        batch_size: int = 32,
    ) -> None:
        """Load a Caffe network and optional per-channel mean data."""
        try:
            import caffe
        except ImportError as exc:
            raise ImportError(
                "Caffe is required for GoogLeNetCaffeWrapper. Install pycaffe "
                "and provide deploy.prototxt plus bvlc_googlenet.caffemodel."
            ) from exc

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.caffe = caffe
        self.model_dir = Path(model_dir)
        self.deploy_prototxt = Path(deploy_prototxt)
        self.caffemodel = Path(caffemodel)
        self.mean_file = Path(mean_file) if mean_file else None
        self.use_gpu = bool(use_gpu)
        self.batch_size = int(batch_size)
        self.input_blob = "data"
        self.output_candidates = ("prob", "loss3/classifier")
        self.image_size = 224

        self._check_required_files()
        if self.use_gpu:
            self.caffe.set_mode_gpu()
        else:
            self.caffe.set_mode_cpu()

        self.net = self.caffe.Net(
            str(self.deploy_prototxt),
            str(self.caffemodel),
            self.caffe.TEST,
        )
        self.mean = self._load_mean(self.mean_file)

    def _check_required_files(self) -> None:
        """Validate model file paths before Caffe tries to open them."""
        missing = [
            path
            for path in (self.deploy_prototxt, self.caffemodel)
            if not path.is_file()
        ]
        if missing:
            raise IOError(
                "Missing GoogLeNet Caffe file(s): {0}".format(
                    ", ".join(str(path) for path in missing)
                )
            )
        if self.mean_file is not None and not self.mean_file.is_file():
            raise IOError("Missing GoogLeNet mean file: {0}".format(self.mean_file))

    def _load_mean(self, mean_file: Optional[Path]) -> Optional[np.ndarray]:
        """Load an optional Caffe mean array."""
        if mean_file is None:
            return None

        mean = np.load(str(mean_file)).astype(np.float32)
        mean = np.squeeze(mean)

        if mean.ndim == 3 and mean.shape[-1] == 3:
            mean = np.transpose(mean, (2, 0, 1))
        if mean.ndim == 3:
            return self._center_crop_mean(mean)
        if mean.ndim == 1 and mean.shape[0] == 3:
            return mean.reshape((3, 1, 1)).astype(np.float32)

        raise ValueError("Expected mean shape (3,), (3,H,W), or (H,W,3).")

    def _center_crop_mean(self, mean: np.ndarray) -> np.ndarray:
        """Crop a spatial mean array to the wrapper input size."""
        if mean.shape[0] != 3:
            raise ValueError("Expected mean channel dimension to be 3.")

        height, width = mean.shape[1], mean.shape[2]
        if height < self.image_size or width < self.image_size:
            raise ValueError("Mean spatial size must be at least 224x224.")

        top = (height - self.image_size) // 2
        left = (width - self.image_size) // 2
        return mean[
            :,
            top : top + self.image_size,
            left : left + self.image_size,
        ].astype(np.float32)

    def _resize_one(self, image: np.ndarray) -> np.ndarray:
        """Resize one normalized RGB image to the Caffe input size."""
        from PIL import Image

        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("Expected image shape (H, W, 3).")

        clipped = np.clip(image_array, 0.0, 1.0)
        pil_image = Image.fromarray((clipped * 255.0).astype(np.uint8), mode="RGB")
        resized = pil_image.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0

    def _as_batch(self, images: np.ndarray) -> np.ndarray:
        """Return input images as an NHWC float32 batch."""
        image_array = np.asarray(images, dtype=np.float32)
        if image_array.ndim == 3:
            image_array = image_array.reshape((1,) + image_array.shape)
        if image_array.ndim != 4 or image_array.shape[-1] != 3:
            raise ValueError("Expected image batch shape (N, H, W, 3).")
        return image_array

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """Convert normalized RGB NHWC images to Caffe BGR NCHW input."""
        image_batch = self._as_batch(images)
        resized = np.asarray(
            [self._resize_one(image) for image in image_batch],
            dtype=np.float32,
        )
        scaled = resized * 255.0
        bgr = scaled[:, :, :, ::-1]
        nchw = np.transpose(bgr, (0, 3, 1, 2)).astype(np.float32)

        if self.mean is not None:
            nchw = nchw - self.mean
        return nchw.astype(np.float32)

    def _output_from_forward(self, output: dict) -> np.ndarray:
        """Select the Caffe output blob containing class scores."""
        for name in self.output_candidates:
            if name in output:
                return np.asarray(output[name], dtype=np.float32)
            if name in self.net.blobs:
                return np.asarray(self.net.blobs[name].data, dtype=np.float32)

        available = sorted(set(output.keys()) | set(self.net.blobs.keys()))
        raise KeyError(
            "Could not find GoogLeNet output blob. Available blobs: {0}".format(
                ", ".join(available)
            )
        )

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """Run Caffe forward passes and return class scores."""
        preprocessed = self.preprocess(images)
        outputs = []

        for start in range(0, len(preprocessed), self.batch_size):
            batch = preprocessed[start : start + self.batch_size]
            self.net.blobs[self.input_blob].reshape(*batch.shape)
            self.net.reshape()
            self.net.blobs[self.input_blob].data[...] = batch
            output = self.net.forward()
            scores = self._output_from_forward(output)
            outputs.append(np.asarray(scores[: len(batch)], dtype=np.float32))

        if not outputs:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(outputs, axis=0).astype(np.float32)
