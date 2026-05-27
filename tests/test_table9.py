import json
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
from scripts.article_reproduction import table_9 as table9_script  # noqa: E402


def test_table9_config_documents_spec_contract() -> None:
    """The Table 9 config should contain the required flows and outputs."""
    config_path = PROJECT_ROOT / "configs" / "article_reproduction" / "table_9.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["experiment"]["name"] == "table_9"
    assert config["orchestration"]["flows"] == [
        "mnist_m1_fgsm",
        "imagenet_googlenet_fgsm",
    ]
    assert config["orchestration"]["aggregate_counts_before_metrics"] is True
    assert config["splits"] == ["Training", "Validation"]
    assert config["detection"]["filter_name"] == "article_final"
    assert config["datasets"]["mnist"]["attack"]["epsilon"] == 0.2
    assert config["datasets"]["imagenet"]["attack"]["epsilon_255"] == 1.0
    assert config["output"] == {
        "results_dir": "results/article_reproduction/table_9",
        "csv": "table_9.csv",
        "markdown": "table_9.md",
        "status_json": "status.json",
    }


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


def test_table9_aggregation_sums_counts_before_metrics() -> None:
    """Table 9 metrics must be computed after summing counters by split."""
    per_flow = {
        "mnist_m1_fgsm": {
            "Training": {"TP": 1, "FN": 9, "FP": 0},
            "Validation": {"TP": 0, "FN": 0, "FP": 0},
        },
        "imagenet_googlenet_fgsm": {
            "Training": {"TP": 9, "FN": 0, "FP": 9},
            "Validation": {"TP": 2, "FN": 1, "FP": 1},
        },
    }

    aggregate = table9_script.aggregate_counters(per_flow)
    rows = table9_script.rows_from_counters(aggregate)
    training = rows[0]

    assert training["TP"] == 10
    assert training["FN"] == 9
    assert training["FP"] == 9
    assert training["recall_percent"] == 52.63
    assert training["precision_percent"] == 52.63
    assert training["f1_percent"] == 52.63


def test_table9_csv_uses_exact_schema_and_splits(tmp_path) -> None:
    """The final CSV should have exactly the Table 9 fields and rows."""
    rows = table9_script.rows_from_counters(
        {
            "Training": {"TP": 1, "FN": 2, "FP": 3},
            "Validation": {"TP": 4, "FN": 5, "FP": 6},
        }
    )

    path = table9_script.write_table9_csv(tmp_path / "table_9.csv", rows)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "split,TP,FN,FP,recall_percent,precision_percent,f1_percent"
    assert [line.split(",")[0] for line in lines[1:]] == ["Training", "Validation"]
    assert len(lines) == 3


def test_table9_dry_run_writes_status_json(monkeypatch, tmp_path) -> None:
    """Dry-run should write status.json without requiring Caffe assets."""
    config = table9_script.load_config(table9_script.DEFAULT_CONFIG)
    monkeypatch.setattr(
        table9_script,
        "_check_mnist_dry_run",
        lambda config: ("available", "mnist available"),
    )
    monkeypatch.setattr(
        table9_script,
        "_check_imagenet_dry_run",
        lambda config: ("blocked", "blocked_imagenet_caffe"),
    )

    payload = table9_script.run_dry_run(
        config,
        config_path=table9_script.DEFAULT_CONFIG,
        output_dir=tmp_path,
    )

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "partial"
    assert status["status"] == "partial"
    assert status["completed_flows"] == ["mnist_m1_fgsm"]
    assert status["skipped_flows"] == ["imagenet_googlenet_fgsm"]
    assert status["warnings"] == ["blocked_imagenet_caffe"]
    assert not (tmp_path / "table_9.csv").exists()


def test_table9_run_writes_outputs_without_diagnostics(monkeypatch, tmp_path) -> None:
    """A real run should emit only CSV, Markdown, and status by default."""
    config = table9_script.load_config(table9_script.DEFAULT_CONFIG)
    monkeypatch.setattr(
        table9_script,
        "run_mnist_flow",
        lambda config, filter_fn, sample_size: {
            "Training": {"TP": 1, "FN": 1, "FP": 0},
            "Validation": {"TP": 2, "FN": 0, "FP": 1},
        },
    )
    monkeypatch.setattr(
        table9_script,
        "run_imagenet_flow",
        lambda config, filter_fn, sample_size: {
            "Training": {"TP": 3, "FN": 0, "FP": 1},
            "Validation": {"TP": 4, "FN": 1, "FP": 0},
        },
    )

    payload = table9_script.run_table9(
        config,
        config_path=table9_script.DEFAULT_CONFIG,
        output_dir=tmp_path,
        sample_size=8,
    )

    assert payload["status"] == "completed"
    assert payload["sample_size"] == 8
    assert (tmp_path / "table_9.csv").is_file()
    assert (tmp_path / "table_9.md").is_file()
    assert (tmp_path / "status.json").is_file()
    assert not (tmp_path / "table_9_diagnostics.csv").exists()


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
