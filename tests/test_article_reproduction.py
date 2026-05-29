from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.evaluation.article_reproduction import (  # noqa: E402
    evaluate_filter_predictions,
    interval_size,
)


def test_interval_size_uses_article_mapping() -> None:
    """Check interval-count to interval-size mapping."""
    assert interval_size(2) == 128
    assert interval_size(6) == 43
    assert interval_size(10) == 26


def test_interval_size_rejects_unsupported_count() -> None:
    """Unsupported interval counts should fail explicitly."""
    with pytest.raises(ValueError):
        interval_size(11)


def test_evaluate_filter_predictions_matches_article_count_semantics() -> None:
    """Check #F, TP, FN, FP and RTP on a small deterministic example."""
    y_true = np.array([1, 2, 3, 4])
    clean_pred = np.array([1, 2, 3, 4])
    adv_pred = np.array([1, 0, 0, 0])
    filtered_clean_pred = np.array([1, 9, 3, 4])
    filtered_adv_pred = np.array([1, 2, 0, 4])

    metrics = evaluate_filter_predictions(
        y_true=y_true,
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        filtered_clean_pred=filtered_clean_pred,
        filtered_adv_pred=filtered_adv_pred,
    )

    assert metrics["F"] == 1
    assert metrics["TP"] == 2
    assert metrics["FN"] == 1
    assert metrics["FP"] == 1
    assert metrics["RTP"] == 2
    assert metrics["recall_percent"] == pytest.approx(66.6666667)
    assert metrics["precision_percent"] == pytest.approx(66.6666667)


def test_evaluate_filter_predictions_can_skip_invalid_table_3_pairs() -> None:
    """Table 3 skips clean errors and failed attacks before metric counting."""
    y_true = np.array([1, 2, 3, 4])
    clean_pred = np.array([1, 9, 3, 4])
    adv_pred = np.array([0, 0, 3, 0])
    filtered_clean_pred = np.array([1, 8, 0, 9])
    filtered_adv_pred = np.array([1, 2, 0, 0])

    metrics = evaluate_filter_predictions(
        y_true=y_true,
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        filtered_clean_pred=filtered_clean_pred,
        filtered_adv_pred=filtered_adv_pred,
        exclude_invalid_pairs=True,
    )

    assert metrics["clean_errors"] == 1
    assert metrics["F"] == 1
    assert metrics["TP"] == 1
    assert metrics["FN"] == 1
    assert metrics["FP"] == 1
    assert metrics["recall_percent"] == pytest.approx(50.0)
    assert metrics["precision_percent"] == pytest.approx(50.0)


def test_table_3_config_documents_experiment_parameters() -> None:
    """Table 3 should record the parameters needed to reproduce the run."""
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    table3 = config["experiments"]["table_3"]

    assert table3["kind"] == "filter_grid"
    assert table3["dataset"] == {
        "name": "mnist",
        "split": "test",
        "start": 0,
        "end": 100,
    }
    assert table3["attack"]["name"] == "fgsm"
    assert table3["attack"]["epsilon"] == 0.2
    assert table3["model"]["checkpoint_dir"] == (
        "artifacts/models/mnist/m1/clean_baseline/checkpoints"
    )
    assert table3["evaluation"]["exclude_invalid_pairs"] is True
    assert table3["evaluation"]["include_filter_time"] is True
    assert table3["output"]["include_filter_name"] is False

    filters = table3["filters"]
    assert filters[0]["type"] == "scalar_quantization"
    assert filters[0]["intervals"] == 2
    assert filters[1]["type"] == "nonuniform_quantization"
    assert len(filters) == 2


def test_table_4_config_documents_experiment_parameters() -> None:
    """Table 4 should record the scalar interval sweep parameters."""
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    table4 = config["experiments"]["table_4"]
    table4_mnist = config["experiments"]["table_4_mnist"]

    assert table4["kind"] == "composite"
    assert table4["components"] == ["table_4_mnist", "table_4_imagenet"]
    assert table4["output_dir"] == "results/experiments/table_4"
    assert table4_mnist["kind"] == "filter_grid"
    assert table4_mnist["output_dir"] == "results/experiments/table_4/mnist"
    assert table4_mnist["dataset"] == {
        "name": "mnist",
        "split": "test",
        "start": 0,
        "end": 4500,
    }
    assert table4_mnist["attack"]["name"] == "fgsm"
    assert table4_mnist["attack"]["epsilon"] == 0.2
    assert table4_mnist["model"]["checkpoint_dir"] == (
        "artifacts/models/mnist/m1/clean_baseline/checkpoints"
    )
    assert [row["intervals"] for row in table4_mnist["filters"]] == [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    assert {row["type"] for row in table4_mnist["filters"]} == {"scalar_quantization"}


def test_imagenet_table_4_config_documents_spec_parameters() -> None:
    """ImageNet Table 4 should record the GoogLeNet/FGSM scalar sweep."""
    experiments = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    config = experiments["experiments"]["table_4_imagenet"]

    assert config["kind"] == "imagenet_table_4"
    assert config["output_dir"] == "results/experiments/table_4/imagenet"
    assert config["dataset"]["name"] == "imagenet"
    assert config["dataset"]["split"] == "train"
    assert config["dataset"]["images_dir"] == "data/imagenet/train"
    assert config["dataset"]["class_indices"] == {
        "goldfish": 1,
        "pineapple": 953,
        "digital_clock": 530,
    }
    assert config["model"]["name"] == "googlenet_caffe"
    assert config["model"]["reference"] == "BVLC GoogLeNet"
    assert config["model"]["mean_file"] is None
    assert config["attack"]["name"] == "fgsm"
    assert config["attack"]["epsilon_255"] == 1.0
    assert config["quantization"]["intervals"] == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert config["quantization"]["interval_sizes"][6] == 43
    assert config["output"]["csv"] == "table_4_imagenet.csv"
    assert "diagnostics_csv" not in config["output"]
    assert config["output"]["status_json"] == "table_4_status.json"


def test_table_6_config_documents_experiment_parameters() -> None:
    """Table 6 should record adaptive quantization validation parameters."""
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    table6 = config["experiments"]["table_6"]

    assert table6["kind"] == "split_eval"
    assert table6["dataset"] == {
        "name": "mnist",
        "split": "test",
        "slices": [
            {"name": "Training", "start": 0, "end": 4500},
            {"name": "Validation", "start": 4500, "end": 5500},
        ],
    }
    assert table6["dataset"]["slices"] == [
        {"name": "Training", "start": 0, "end": 4500},
        {"name": "Validation", "start": 4500, "end": 5500},
    ]
    assert table6["attack"]["name"] == "fgsm"
    assert table6["attack"]["epsilon"] == 0.2
    assert table6["model"]["checkpoint_dir"] == (
        "artifacts/models/mnist/m1/clean_baseline/checkpoints"
    )
    assert table6["filter"] == {
        "name": "adaptive_quantization",
        "type": "adaptive_quantization",
    }
    assert table6["output_dir"] == "results/experiments/table_6"


def test_table_10_m2_config_documents_experiment_parameters() -> None:
    """Table 10 M2 should record saved CW adversarial evaluation parameters."""
    config_path = PROJECT_ROOT / "configs" / "article_reproduction" / "mnist_table_10_m2.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["dataset"] == {
        "name": "mnist",
        "split": "test",
        "start": 9000,
        "samples": 1000,
        "image_shape": [28, 28, 1],
        "value_range": [0.0, 1.0],
    }
    assert config["model"]["name"] == "M2"
    assert config["model"]["checkpoint_dir"] == (
        "artifacts/models/mnist/m2/clean_baseline/checkpoints"
    )
    assert config["detection"]["filter"] == "final"
    assert config["evaluation"]["use_saved_adversarial_examples"] is True
    assert config["evaluation"]["train_model"] is False
    assert config["evaluation"]["generate_attacks"] is False

    attacks = config["attacks"]
    assert attacks[0]["name"] == "CW L2 / M2"
    assert attacks[0]["norm"] == "L2"
    assert attacks[0]["kappas"] == [0.0, 0.5, 1.0, 2.0, 4.0]
    assert attacks[0]["adversarial_template"] == (
        "artifacts/adversarial_examples/mnist/m2/cw_l2/"
        "kappa_{kappa}/adversarial_examples.npy"
    )
    assert attacks[1]["name"] == "CW Linf / M2"
    assert attacks[1]["norm"] == "Linf"
    assert attacks[1]["adversarial_path"] == (
        "artifacts/adversarial_examples/mnist/m2/cw_linf/adversarial_examples.npy"
    )

    assert config["metrics"]["columns"] == [
        "Attack/Model",
        "Dataset",
        "#F",
        "TP",
        "FN",
        "FP",
        "RTP",
        "RTP%",
        "Recall",
        "Precision",
        "F1",
    ]
