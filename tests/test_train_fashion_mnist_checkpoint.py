from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from scripts import train_fashion_mnist_checkpoint as train_script  # noqa: E402


class FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _config() -> dict:
    return {
        "experiments": {
            "fashion_mnist_fgsm_m1": {
                "model_group": "m1",
                "dataset": {
                    "name": "fashion_mnist",
                    "csv_path": "data/fashion_mnist/fashion-mnist_test.csv",
                    "split_strategy": {
                        "name": "balanced_by_class",
                        "train_samples": 20,
                        "evaluation_samples": 10,
                        "train_class_quota": 2,
                        "evaluation_class_quota": 1,
                    },
                },
                "model": {
                    "checkpoint_dir": "artifacts/models/fashion_mnist/m1/checkpoints"
                },
                "checkpoint_training": {
                    "source_csv": "data/fashion_mnist/fashion-mnist_test.csv",
                    "split_name": "train",
                    "selection": {
                        "method": "first_n_per_class",
                        "per_class_start": 0,
                        "per_class_end": 2,
                        "class_quota": 2,
                        "total_samples": 20,
                    },
                    "validation_sample": {
                        "split_name": "evaluation",
                        "method": "next_n_per_class",
                        "per_class_start": 2,
                        "per_class_end": 3,
                        "class_quota": 1,
                        "total_samples": 10,
                        "excludes_training": True,
                    },
                },
            },
            "fashion_mnist_cw_l2_m2": {
                "model_group": "m2",
                "dataset": {
                    "name": "fashion_mnist",
                    "csv_path": "data/fashion_mnist/fashion-mnist_test.csv",
                    "split_strategy": {
                        "name": "balanced_by_class",
                        "train_samples": 20,
                        "evaluation_samples": 10,
                        "train_class_quota": 2,
                        "evaluation_class_quota": 1,
                    },
                },
                "model": {
                    "checkpoint_dir": "artifacts/models/fashion_mnist/m2/checkpoints"
                },
                "checkpoint_training": {
                    "source_csv": "data/fashion_mnist/fashion-mnist_test.csv",
                    "split_name": "train",
                    "selection": {
                        "method": "first_n_per_class",
                        "per_class_start": 0,
                        "per_class_end": 2,
                        "class_quota": 2,
                        "total_samples": 20,
                    },
                    "validation_sample": {
                        "split_name": "evaluation",
                        "method": "next_n_per_class",
                        "per_class_start": 2,
                        "per_class_end": 3,
                        "class_quota": 1,
                        "total_samples": 10,
                        "excludes_training": True,
                    },
                },
            },
        }
    }


def _fake_split(dataset_config: dict) -> dict:
    assert dataset_config["name"] == "fashion_mnist"
    return {
        "train_images": np.zeros((20, 28, 28, 1), dtype=np.float32),
        "train_labels_one_hot": np.eye(10, dtype=np.float32)[[0, 1] * 10],
        "evaluation_images": np.zeros((10, 28, 28, 1), dtype=np.float32),
        "evaluation_labels_one_hot": np.eye(10, dtype=np.float32),
        "metadata": {
            "csv_path": dataset_config["csv_path"],
            "split_strategy": {
                "train_class_quota": 2,
                "evaluation_class_quota": 1,
            },
        },
    }


def _fake_graph(session: FakeSession) -> dict:
    return {
        "sess": session,
        "x": object(),
        "y": object(),
        "predictions": object(),
    }


def test_train_fashion_mnist_checkpoint_uses_m1_experiment_defaults(monkeypatch) -> None:
    session = FakeSession()
    calls = {}

    def fake_trainer(sess, x, y, predictions, x_train, y_train, x_test, y_test, config):
        calls["trainer"] = "m1"
        calls["session"] = sess
        calls["x_train_shape"] = x_train.shape
        calls["y_train_shape"] = y_train.shape
        calls["x_test_shape"] = x_test.shape
        calls["y_test_shape"] = y_test.shape
        calls["config"] = dict(config)
        return {"checkpoint_path": str(Path(config["train_dir"]) / config["filename"])}

    monkeypatch.setattr(train_script, "split_fashion_mnist_balanced", _fake_split)
    monkeypatch.setattr(train_script, "build_training_graph", lambda model: _fake_graph(session))
    monkeypatch.setattr(train_script, "train_or_load_mnist_model", fake_trainer)

    result = train_script.run_training("m1", _config(), epochs=1)

    assert calls["trainer"] == "m1"
    assert calls["session"] is session
    assert calls["x_train_shape"] == (20, 28, 28, 1)
    assert calls["y_train_shape"] == (20, 10)
    assert calls["x_test_shape"] == (10, 28, 28, 1)
    assert calls["y_test_shape"] == (10, 10)
    assert calls["config"]["train_dir"].endswith(
        "artifacts/models/fashion_mnist/m1/checkpoints"
    )
    assert calls["config"]["filename"] == "mnist.ckpt"
    assert calls["config"]["epochs"] == 1
    assert result["experiment_id"] == "fashion_mnist_fgsm_m1"
    assert result["model"] == "m1"
    assert result["train_samples"] == 20
    assert result["evaluation_samples"] == 10
    assert result["train_class_quota"] == 2
    assert result["evaluation_class_quota"] == 1
    assert result["checkpoint_training"]["selection"]["per_class_end"] == 2
    assert result["checkpoint_training"]["validation_sample"]["per_class_start"] == 2
    assert session.closed is True


def test_train_fashion_mnist_checkpoint_uses_m2_trainer_and_checkpoint(monkeypatch) -> None:
    session = FakeSession()
    calls = {}

    def fake_m1(*args, **kwargs):
        raise AssertionError("M1 trainer should not be called for --model m2")

    def fake_m2(sess, x, y, predictions, x_train, y_train, x_test, y_test, config):
        calls["config"] = dict(config)
        return {"checkpoint_path": str(Path(config["train_dir"]) / config["filename"])}

    monkeypatch.setattr(train_script, "split_fashion_mnist_balanced", _fake_split)
    monkeypatch.setattr(train_script, "build_training_graph", lambda model: _fake_graph(session))
    monkeypatch.setattr(train_script, "train_or_load_mnist_model", fake_m1)
    monkeypatch.setattr(train_script, "train_or_load_mnist_m2_model", fake_m2)

    result = train_script.run_training("m2", _config(), batch_size=64, load_model=True)

    assert calls["config"]["train_dir"].endswith(
        "artifacts/models/fashion_mnist/m2/checkpoints"
    )
    assert calls["config"]["filename"] == "mnist_m2.ckpt"
    assert calls["config"]["epochs"] == 10
    assert calls["config"]["batch_size"] == 64
    assert calls["config"]["load_model"] is True
    assert result["experiment_id"] == "fashion_mnist_cw_l2_m2"
    assert result["model"] == "m2"
    assert session.closed is True


def test_train_fashion_mnist_checkpoint_rejects_model_experiment_mismatch() -> None:
    with pytest.raises(ValueError, match="configured for model m2, not m1"):
        train_script.run_training(
            "m1",
            _config(),
            experiment_id="fashion_mnist_cw_l2_m2",
        )


def test_train_fashion_mnist_checkpoint_rejects_training_sample_mismatch(
    monkeypatch,
) -> None:
    config = _config()
    config["experiments"]["fashion_mnist_fgsm_m1"]["checkpoint_training"]["selection"][
        "per_class_end"
    ] = 3

    monkeypatch.setattr(train_script, "split_fashion_mnist_balanced", _fake_split)

    with pytest.raises(
        ValueError,
        match="checkpoint_training.selection.per_class_end must be 2",
    ):
        train_script.run_training("m1", config)


def test_train_fashion_mnist_checkpoint_cli_prints_json(monkeypatch, capsys) -> None:
    calls = {}

    def fake_run_training(**kwargs):
        calls.update(kwargs)
        return {
            "checkpoint_path": "/tmp/fashion_mnist/m1/mnist.ckpt",
            "model": kwargs["model_name"],
            "train_samples": 9000,
        }

    monkeypatch.setattr(train_script, "load_yaml_config", lambda path: {"experiments": {}})
    monkeypatch.setattr(train_script, "run_training", fake_run_training)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_fashion_mnist_checkpoint.py",
            "--model",
            "m1",
            "--epochs",
            "2",
            "--batch-size",
            "32",
            "--learning-rate",
            "0.002",
            "--filename",
            "custom.ckpt",
            "--load-model",
        ],
    )

    assert train_script.main() == 0

    output = capsys.readouterr().out
    assert '"checkpoint_path": "/tmp/fashion_mnist/m1/mnist.ckpt"' in output
    assert calls["model_name"] == "m1"
    assert calls["filename"] == "custom.ckpt"
    assert calls["epochs"] == 2
    assert calls["batch_size"] == 32
    assert calls["learning_rate"] == 0.002
    assert calls["load_model"] is True
