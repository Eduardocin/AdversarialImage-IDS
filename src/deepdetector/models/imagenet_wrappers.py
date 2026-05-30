"""Backend-agnostic ImageNet model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
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

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return scores for one image."""
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim == 3:
            image_array = image_array.reshape((1,) + image_array.shape)
        return self.predict_batch(image_array)[0]

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

    model_label = "GoogLeNet"
    default_image_size = 224
    default_output_candidates = ("prob", "loss3/classifier")

    def __init__(
        self,
        model_dir: str,
        deploy_prototxt: str,
        caffemodel: str,
        attack_deploy_prototxt: Optional[str] = None,
        mean_file: Optional[str] = None,
        use_gpu: bool = False,
        batch_size: int = 32,
    ) -> None:
        """Load prediction and optional attack Caffe networks."""
        try:
            os.environ.setdefault("GLOG_minloglevel", "2")
            import caffe
        except ImportError as exc:
            raise ImportError(
                "Caffe is required for {0}. Install pycaffe and provide "
                "deploy.prototxt plus a compatible .caffemodel.".format(
                    self.__class__.__name__
                )
            ) from exc

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.caffe = caffe
        self.model_dir = Path(model_dir)
        self.deploy_prototxt = Path(deploy_prototxt)
        self.attack_deploy_prototxt = (
            Path(attack_deploy_prototxt) if attack_deploy_prototxt else None
        )
        self.caffemodel = Path(caffemodel)
        self.mean_file = Path(mean_file) if mean_file else None
        self.use_gpu = bool(use_gpu)
        self.batch_size = int(batch_size)
        self.input_blob = "data"
        self.output_candidates = tuple(self.default_output_candidates)
        self.image_size = int(self.default_image_size)

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
        self.attack_net = (
            self.caffe.Net(
                str(self.attack_deploy_prototxt),
                str(self.caffemodel),
                self.caffe.TEST,
            )
            if self.attack_deploy_prototxt is not None
            else self.net
        )
        self.mean = self._load_mean(self.mean_file)

    def _check_required_files(self) -> None:
        """Validate model file paths before Caffe tries to open them."""
        model_paths = [self.deploy_prototxt, self.caffemodel]
        if self.attack_deploy_prototxt is not None:
            model_paths.append(self.attack_deploy_prototxt)
        missing = [
            path
            for path in model_paths
            if not path.is_file()
        ]
        if missing:
            raise IOError(
                "Missing {0} Caffe file(s): {1}".format(
                    self.model_label,
                    ", ".join(str(path) for path in missing)
                )
            )
        if self.mean_file is not None and not self.mean_file.is_file():
            raise IOError("Missing {0} mean file: {1}".format(self.model_label, self.mean_file))

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
            raise ValueError(
                "Mean spatial size must be at least {0}x{0}.".format(self.image_size)
            )

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

    def _output_from_forward(self, output: dict, net: Optional[object] = None) -> np.ndarray:
        """Select the Caffe output blob containing class scores."""
        selected_net = net or self.net
        output_blob = self._output_blob_name(output, net=selected_net)
        if output_blob in output:
            return np.asarray(output[output_blob], dtype=np.float32)
        return np.asarray(selected_net.blobs[output_blob].data, dtype=np.float32)

    def _output_blob_name(
        self,
        output: Optional[dict] = None,
        net: Optional[object] = None,
    ) -> str:
        """Return the Caffe output blob name containing class scores."""
        selected_net = net or self.net
        output = output or {}
        for name in self.output_candidates:
            if name in output:
                return name
            if name in selected_net.blobs:
                return name

        available = sorted(set(output.keys()) | set(selected_net.blobs.keys()))
        raise KeyError(
            "Could not find {0} output blob. Available blobs: {1}".format(
                self.model_label,
                ", ".join(available)
            )
        )

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """Run Caffe forward passes and return class scores."""
        preprocessed = self.preprocess(images)
        return self.predict_preprocessed_batch(preprocessed)

    def predict_preprocessed_batch(self, preprocessed_images: np.ndarray) -> np.ndarray:
        """Run Caffe forward passes for NCHW BGR images in Caffe input space."""
        preprocessed = np.asarray(preprocessed_images, dtype=np.float32)
        if preprocessed.ndim == 3:
            preprocessed = preprocessed.reshape((1,) + preprocessed.shape)
        if preprocessed.ndim != 4 or preprocessed.shape[1] != 3:
            raise ValueError("Expected preprocessed batch shape (N, 3, H, W).")

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

    def predict_preprocessed_label(self, preprocessed_images: np.ndarray) -> np.ndarray:
        """Return top-1 labels for NCHW BGR images in Caffe input space."""
        scores = self.predict_preprocessed_batch(preprocessed_images)
        return np.asarray(np.argmax(scores, axis=1), dtype=np.int32)

    def gradient(self, image: np.ndarray, class_id: int) -> np.ndarray:
        """Return the Caffe input gradient for one class id.

        NHWC RGB images are preprocessed with this wrapper. A direct Caffe NCHW
        tensor can also be passed when the caller is already operating in
        preprocessed Caffe space.
        """
        image_array = np.asarray(image, dtype=np.float32)

        input_was_nhwc = not (
            image_array.ndim == 3 and image_array.shape[0] == 3
        )

        if image_array.ndim == 3 and image_array.shape[0] == 3:
            preprocessed = image_array.reshape((1,) + image_array.shape)
        else:
            preprocessed = self.preprocess(image_array)

        if len(preprocessed) != 1:
            raise ValueError("gradient expects exactly one image.")

        gradient_net = self.attack_net
        gradient_net.blobs[self.input_blob].reshape(*preprocessed.shape)
        gradient_net.reshape()
        gradient_net.blobs[self.input_blob].data[...] = preprocessed

        output = gradient_net.forward()

        if "prob" in gradient_net.blobs:
            output_blob = "prob"
        else:
            output_blob = self._output_blob_name(output, net=gradient_net)

        output_diff = np.zeros_like(
            gradient_net.blobs[output_blob].data,
            dtype=np.float32,
        )
        output_diff[0, int(class_id)] = -1.0

        backward_result = gradient_net.backward(**{output_blob: output_diff})

        if isinstance(backward_result, dict) and self.input_blob in backward_result:
            gradient = np.asarray(
                backward_result[self.input_blob][0],
                dtype=np.float32,
            )
        else:
            gradient = np.asarray(
                gradient_net.blobs[self.input_blob].diff[0],
                dtype=np.float32,
            )

        if not input_was_nhwc:
            return gradient.astype(np.float32)

        gradient_hwc = np.transpose(gradient, (1, 2, 0))[:, :, ::-1] * 255.0

        if image_array.shape[:2] == gradient_hwc.shape[:2]:
            return gradient_hwc.astype(np.float32)

        from PIL import Image

        resized_channels = []
        target_size = (int(image_array.shape[1]), int(image_array.shape[0]))

        for channel in range(gradient_hwc.shape[2]):
            channel_image = Image.fromarray(
                gradient_hwc[:, :, channel].astype(np.float32)
            )
            resized = channel_image.resize(target_size, Image.BILINEAR)
            resized_channels.append(np.asarray(resized, dtype=np.float32))

        return np.stack(resized_channels, axis=2).astype(np.float32)


class CaffeNetCaffeWrapper(GoogLeNetCaffeWrapper):
    """Run CaffeNet with separate prediction and DeepFool attack deploy files.

    The original DeepDetector CaffeNet flow uses ``deploy_original.prototxt``
    for classification/detection and ``deploy_removeSoftmax.prototxt`` for
    DeepFool gradients. This subclass keeps that split while reusing the common
    Caffe preprocessing, prediction, and gradient contract.
    """

    model_label = "CaffeNet"
    default_image_size = 227
    default_output_candidates = ("prob", "fc8")


class InceptionV3TensorFlowWrapper(ImageNetModelWrapper):
    """Run frozen TensorFlow Inception v3 graphs for Table 10 ImageNet CW rows."""

    def __init__(
        self,
        graph_path: str,
        input_map_name: str = "ResizeBilinear:0",
        output_tensor_name: str = "softmax/logits:0",
        batch_size: int = 32,
        sess: Optional[object] = None,
    ) -> None:
        """Load a frozen Inception v3 graph and expose logits for CW attacks."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.graph_path = Path(graph_path)
        if not self.graph_path.is_file():
            raise IOError("Missing Inception v3 graph file: {0}".format(self.graph_path))

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow is required for InceptionV3TensorFlowWrapper.") from exc

        self.tf = tf
        self.image_size = 299
        self.num_labels = 1008
        self.batch_size = int(batch_size)
        self.input_map_name = str(input_map_name)
        self.output_tensor_name = str(output_tensor_name)
        self._attack_import_index = 0

        graph_def = self._load_graph_def()
        self.graph_def = graph_def
        self.graph = self.tf.Graph()
        with self.graph.as_default():
            self.input_tensor = self.tf.compat.v1.placeholder(
                self.tf.float32,
                shape=(None, self.image_size, self.image_size, 3),
                name="inception_input",
            )
            self.logits_tensor = self._import_logits(self.input_tensor, name="")

        self.sess = sess or self.tf.compat.v1.Session(graph=self.graph)
        self.session = self.sess

    def _load_graph_def(self) -> object:
        graph_def = self.tf.compat.v1.GraphDef()
        with self.graph_path.open("rb") as handle:
            graph_def.ParseFromString(handle.read())
        return graph_def

    def _scaled_input(self, tensor: object) -> object:
        """Map Table 10 Inception input [-0.5, 0.5] to the graph's 0-255 input."""
        if self.input_map_name == "ResizeBilinear:0":
            return (tensor + 0.5) * 255.0
        if self.input_map_name == "Sub:0":
            return (tensor + 0.5) * 255.0 - 128.0
        if self.input_map_name == "Mul:0":
            return tensor * 2.0
        return tensor

    def _import_logits(self, tensor: object, name: str) -> object:
        elements = self.tf.import_graph_def(
            self.graph_def,
            name=name,
            input_map={self.input_map_name: self._scaled_input(tensor)},
            return_elements=[self.output_tensor_name],
        )
        return elements[0]

    def _resize_one(self, image: np.ndarray) -> np.ndarray:
        """Resize one RGB image to Inception's fixed 299x299 input size."""
        from PIL import Image

        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("Expected image shape (H, W, 3).")

        clipped = np.clip(image_array, 0.0, 1.0)
        pil_image = Image.fromarray((clipped * 255.0).astype(np.uint8), mode="RGB")
        resized = pil_image.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0

    def _as_batch(self, images: np.ndarray) -> np.ndarray:
        image_array = np.asarray(images, dtype=np.float32)
        if image_array.ndim == 3:
            image_array = image_array.reshape((1,) + image_array.shape)
        if image_array.ndim != 4 or image_array.shape[-1] != 3:
            raise ValueError("Expected image batch shape (N, H, W, 3).")
        return image_array

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """Convert normalized RGB images from [0, 1] to Inception [-0.5, 0.5]."""
        image_batch = self._as_batch(images)
        resized = np.asarray(
            [self._resize_one(image) for image in image_batch],
            dtype=np.float32,
        )
        return (resized - 0.5).astype(np.float32)

    def _looks_preprocessed(self, images: np.ndarray) -> bool:
        if images.size == 0:
            return False
        return bool(float(np.nanmin(images)) < 0.0)

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        image_batch = self._as_batch(images)
        if self._looks_preprocessed(image_batch):
            return self.predict_preprocessed_batch(image_batch)
        return self.predict_preprocessed_batch(self.preprocess(image_batch))

    def predict_preprocessed_batch(self, preprocessed_images: np.ndarray) -> np.ndarray:
        preprocessed = self._as_batch(preprocessed_images)
        outputs = []
        for start in range(0, len(preprocessed), self.batch_size):
            batch = preprocessed[start : start + self.batch_size]
            values = self.sess.run(self.logits_tensor, feed_dict={self.input_tensor: batch})
            outputs.append(np.asarray(values[: len(batch)], dtype=np.float32))
        if not outputs:
            return np.empty((0, self.num_labels), dtype=np.float32)
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def predict_preprocessed_label(self, preprocessed_images: np.ndarray) -> np.ndarray:
        scores = self.predict_preprocessed_batch(preprocessed_images)
        return np.asarray(np.argmax(scores, axis=1), dtype=np.int32)

    def predict_label(self, images: np.ndarray) -> np.ndarray:
        scores = self.predict_batch(images)
        return np.asarray(np.argmax(scores, axis=1), dtype=np.int32)

    def get_logits(self, tensor: object) -> object:
        """Return logits for an attack tensor already in Inception [-0.5, 0.5]."""
        self._attack_import_index += 1
        return self._import_logits(
            tensor,
            name="inception_attack_{0}".format(self._attack_import_index),
        )

    def __call__(self, tensor: object) -> object:
        return self.get_logits(tensor)
