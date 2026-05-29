from pathlib import Path
import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.models.imagenet_wrappers import (
    GoogLeNetCaffeWrapper,
    InceptionV3TensorFlowWrapper,
)


def test_googlenet_caffe_wrapper_reports_missing_caffe(monkeypatch) -> None:
    """Check that the Caffe dependency failure is explicit."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "caffe":
            raise ImportError("no caffe")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="Caffe is required"):
        GoogLeNetCaffeWrapper(
            model_dir="models/googlenet",
            deploy_prototxt="models/googlenet/deploy.prototxt",
            caffemodel="models/googlenet/bvlc_googlenet.caffemodel",
        )


class FakeBlob:
    def __init__(self, shape=(1, 2)):
        self.reshape(*shape)

    def reshape(self, *shape):
        self.data = np.zeros(shape, dtype=np.float32)
        self.diff = np.zeros(shape, dtype=np.float32)


class FakeNet:
    def __init__(self, deploy, caffemodel, mode):
        self.deploy = deploy
        self.caffemodel = caffemodel
        self.mode = mode
        self.forward_calls = 0
        self.backward_calls = []
        output_blob = "loss3/classifier" if "removeSoftmax" in deploy else "prob"
        self.output_blob = output_blob
        self.blobs = {"data": FakeBlob((1, 3, 2, 2)), output_blob: FakeBlob((1, 3))}

    def reshape(self):
        return None

    def forward(self):
        self.forward_calls += 1
        values = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        self.blobs[self.output_blob].data[...] = values
        return {self.output_blob: values}

    def backward(self, **kwargs):
        self.backward_calls.append(kwargs)
        marker = 7.0 if "removeSoftmax" in self.deploy else 3.0
        return {"data": np.full_like(self.blobs["data"].data, marker, dtype=np.float32)}


def _install_fake_caffe(monkeypatch):
    fake_caffe = SimpleNamespace(TEST="test", Net=FakeNet)
    fake_caffe.set_mode_cpu = lambda: None
    fake_caffe.set_mode_gpu = lambda: None
    monkeypatch.setitem(sys.modules, "caffe", fake_caffe)


def _touch_model_files(tmp_path):
    model_dir = tmp_path / "googlenet"
    model_dir.mkdir()
    deploy = model_dir / "deploy_original.prototxt"
    attack_deploy = model_dir / "deploy_removeSoftmax.prototxt"
    caffemodel = model_dir / "bvlc_googlenet.caffemodel"
    for path in (deploy, attack_deploy, caffemodel):
        path.write_text("placeholder", encoding="utf-8")
    return model_dir, deploy, attack_deploy, caffemodel


def test_googlenet_caffe_wrapper_uses_attack_deploy_for_gradient(monkeypatch, tmp_path) -> None:
    """Prediction should use the original net, while gradients use removeSoftmax."""
    _install_fake_caffe(monkeypatch)
    model_dir, deploy, attack_deploy, caffemodel = _touch_model_files(tmp_path)

    wrapper = GoogLeNetCaffeWrapper(
        model_dir=str(model_dir),
        deploy_prototxt=str(deploy),
        attack_deploy_prototxt=str(attack_deploy),
        caffemodel=str(caffemodel),
    )

    assert wrapper.net.deploy == str(deploy)
    assert wrapper.attack_net.deploy == str(attack_deploy)
    assert wrapper.predict_preprocessed_batch(np.zeros((1, 3, 2, 2), dtype=np.float32)).shape == (
        1,
        3,
    )

    gradient = wrapper.gradient(np.zeros((3, 2, 2), dtype=np.float32), class_id=2)

    np.testing.assert_array_equal(gradient, np.full((3, 2, 2), 7.0, dtype=np.float32))
    assert wrapper.net.backward_calls == []
    assert len(wrapper.attack_net.backward_calls) == 1
    assert "loss3/classifier" in wrapper.attack_net.backward_calls[0]


def test_googlenet_caffe_wrapper_falls_back_to_prediction_deploy_for_gradient(
    monkeypatch,
    tmp_path,
) -> None:
    """Without attack_deploy_prototxt, legacy single-net gradient behavior remains."""
    _install_fake_caffe(monkeypatch)
    model_dir, deploy, _attack_deploy, caffemodel = _touch_model_files(tmp_path)

    wrapper = GoogLeNetCaffeWrapper(
        model_dir=str(model_dir),
        deploy_prototxt=str(deploy),
        caffemodel=str(caffemodel),
    )

    assert wrapper.attack_net is wrapper.net

    gradient = wrapper.gradient(np.zeros((3, 2, 2), dtype=np.float32), class_id=1)

    np.testing.assert_array_equal(gradient, np.full((3, 2, 2), 3.0, dtype=np.float32))
    assert len(wrapper.net.backward_calls) == 1
    assert "prob" in wrapper.net.backward_calls[0]


def test_inception_v3_wrapper_reports_missing_graph(tmp_path) -> None:
    """Missing Inception graph assets should fail before TensorFlow graph loading."""
    with pytest.raises(IOError, match="Missing Inception v3 graph file"):
        InceptionV3TensorFlowWrapper(graph_path=str(tmp_path / "missing.pb"))


def test_inception_v3_preprocess_uses_299_centered_range() -> None:
    """Inception v3 Table 10 inputs should be resized to 299 and centered."""
    wrapper = object.__new__(InceptionV3TensorFlowWrapper)
    wrapper.image_size = 299

    images = np.asarray(
        [
            np.zeros((4, 4, 3), dtype=np.float32),
            np.ones((4, 4, 3), dtype=np.float32),
        ]
    )

    preprocessed = InceptionV3TensorFlowWrapper.preprocess(wrapper, images)

    assert preprocessed.shape == (2, 299, 299, 3)
    assert float(preprocessed[0].min()) == -0.5
    assert float(preprocessed[1].max()) == 0.5
