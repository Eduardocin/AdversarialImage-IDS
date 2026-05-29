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
from deepdetector.experiments import table9_runner  # noqa: E402
from deepdetector.models import mnist_cnn  # noqa: E402


def test_table9_config_documents_spec_contract() -> None:
    """The refactored Table 9 config should combine MNIST and ImageNet."""
    config_path = PROJECT_ROOT / "configs" / "experiments.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    table9 = config["experiments"]["table_9"]

    assert table9["kind"] == "table_9"
    assert table9["datasets"] == ["mnist", "imagenet"]
    assert table9["mnist"]["dataset"]["slices"] == [
        {"name": "Training", "start": 0, "end": 4500},
        {"name": "Validation", "start": 4500, "end": 5500},
    ]
    assert table9["imagenet"]["dataset"]["splits"]["train"] == [
        {"name": "goldfish", "label": 1, "path": "data/imagenet/train/goldfish"},
        {"name": "pineapple", "label": 953, "path": "data/imagenet/train/pineapple"},
        {
            "name": "digital_clock",
            "label": 530,
            "path": "data/imagenet/train/digital_clock",
        },
    ]
    assert table9["mnist"]["filter"] == {
        "name": "proposed_detection_filter",
        "type": "proposed_detection_filter",
    }
    assert table9["imagenet"]["filter"] == table9["mnist"]["filter"]
    assert table9["output_dir"] == "results/experiments/table_9"
    assert table9["mnist"]["attack"]["epsilon"] == 0.2
    assert table9["imagenet"]["attack"]["epsilon_255"] == 1.0


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


def test_run_table9_experiment_writes_combined_official_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    """The official Table 9 runner should aggregate MNIST and ImageNet outputs."""
    monkeypatch.setattr(
        table9_runner,
        "evaluate_mnist_table9",
        lambda config: [
            {"split": "Training", "TP": 1, "FN": 1, "FP": 0},
            {"split": "Validation", "TP": 2, "FN": 0, "FP": 1},
        ],
    )
    monkeypatch.setattr(
        table9_runner,
        "evaluate_imagenet_table9",
        lambda config: [
            {"split": "train", "TP": 3, "FN": 1, "FP": 2},
            {"split": "validation", "TP": 1, "FN": 3, "FP": 0},
        ],
    )

    rows = table9_runner.run_table9_experiment(
        {
            "experiment_id": "table_9",
            "kind": "table_9",
            "mnist": {"dataset": {"name": "mnist"}},
            "imagenet": {"dataset": {"name": "imagenet"}},
            "split_order": ["train", "validation"],
            "output": {"dir": str(tmp_path), "csv": "metrics.csv", "json": "metrics.json"},
        }
    )

    assert [row["split"] for row in rows] == ["train", "validation"]
    assert rows[0]["TP"] == 4
    assert rows[0]["FN"] == 2
    assert rows[0]["FP"] == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "metrics.csv",
        "metrics.json",
    ]


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
