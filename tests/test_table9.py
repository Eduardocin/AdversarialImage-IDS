from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.filters.article_final import article_final_detection_filter  # noqa: E402
from deepdetector.filters.registry import FILTER_REGISTRY  # noqa: E402
from deepdetector.data import mnist as mnist_data  # noqa: E402
from deepdetector.models import mnist_cnn  # noqa: E402


def test_table9_config_documents_spec_contract() -> None:
    """The refactored Table 9 config should use the shared split-runner contract."""
    config_path = PROJECT_ROOT / "configs" / "experiments.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    table9 = config["experiments"]["table_9"]

    assert table9["kind"] == "split_eval"
    assert table9["slices"] == [
        {"name": "Training", "start": 0, "end": 4500},
        {"name": "Validation", "start": 4500, "end": 5500},
    ]
    assert table9["filter"] == "proposed_filter"
    assert table9["output_dir"] == "results/experiments/table_9"
    assert config["defaults"]["epsilon"] == 0.2


def test_article_final_filter_is_registered_and_preserves_scales() -> None:
    """The final filter should be registry-visible and preserve input scale."""
    assert FILTER_REGISTRY["article_final"] is article_final_detection_filter

    normalized = np.linspace(0.0, 1.0, 28 * 28, dtype=np.float32).reshape(28, 28, 1)
    normalized_output = article_final_detection_filter(normalized)
    assert normalized_output.shape == normalized.shape
    assert normalized_output.dtype == np.float32
    assert float(normalized_output.min()) >= 0.0
    assert float(normalized_output.max()) <= 1.0

    caffe = np.linspace(0.0, 255.0, 3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    caffe_output = article_final_detection_filter(caffe)
    assert caffe_output.shape == caffe.shape
    assert caffe_output.dtype == np.float32
    assert float(caffe_output.min()) >= 0.0
    assert float(caffe_output.max()) <= 255.0


def test_table9_legacy_script_was_removed() -> None:
    """The old Table 9 entry point should no longer be an active runtime path."""
    assert not (PROJECT_ROOT / "scripts/article_reproduction/table_9.py").exists()


def test_mnist_latest_checkpoint_falls_back_to_local_base(monkeypatch, tmp_path) -> None:
    """WSL runs should ignore stale Windows paths in TensorFlow checkpoint metadata."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "mnist.ckpt.index").write_text("index", encoding="utf-8")
    (checkpoint_dir / "mnist.ckpt.data-00000-of-00001").write_text("data", encoding="utf-8")
    fake_checkpoint = SimpleNamespace(model_checkpoint_path="C:\\stale\\mnist.ckpt")
    fake_tf = SimpleNamespace(
        train=SimpleNamespace(get_checkpoint_state=lambda train_dir: fake_checkpoint)
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    checkpoint = mnist_cnn.latest_checkpoint(str(checkpoint_dir))

    assert checkpoint == str(checkpoint_dir / "mnist.ckpt")


def test_mnist_keras_fallback_format_matches_cleverhans_shape() -> None:
    """The Keras fallback should preserve the legacy MNIST array contract."""
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    labels = np.asarray([3, 7], dtype=np.int64)

    formatted_images = mnist_data._images_to_nhwc(images)
    formatted_labels = mnist_data._one_hot(labels)

    assert formatted_images.shape == (2, 28, 28, 1)
    assert formatted_images.dtype == np.float32
    assert formatted_labels.shape == (2, 10)
    assert formatted_labels.dtype == np.float32
    assert formatted_labels[0, 3] == 1.0
    assert formatted_labels[1, 7] == 1.0
