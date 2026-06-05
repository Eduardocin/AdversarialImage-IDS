import csv
import json
from pathlib import Path
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.evaluation.tables.table_10 import (  # noqa: E402
    TABLE_10_SCHEMA,
    build_pending_table_10_row,
    evaluate_table_10_fashion_mnist_row,
    evaluate_table_10_imagenet_row,
    evaluate_table_10_googlenet_row,
    run_table_10_group,
)
from deepdetector.evaluation.tables import table_10 as table_10_module  # noqa: E402
from deepdetector.experiments import runner as experiment_runner  # noqa: E402
from deepdetector.filters.article_final import article_final_detection_filter  # noqa: E402
from scripts import run_experiment as run_experiment_script  # noqa: E402


TABLE10_EXPERIMENTS = {
    "table_10_m1": ("m1", "MNIST", [1, 2, 3, 4]),
    "table_10_googlenet": ("googlenet", "ImageNet", [5, 6, 7]),
    "table_10_caffenet": ("caffenet", "ImageNet", [8]),
    "table_10_m2": ("m2", "MNIST", [9, 10, 11, 12, 13, 19]),
    "table_10_inception_v3": (
        "inception_v3",
        "ImageNet",
        [14, 15, 16, 17, 18, 20],
    ),
}

IMAGENET_NEW_CLASS_EXPERIMENTS = {
    "imagenet_new_classes_fgsm_googlenet": (
        "fgsm_googlenet",
        "googlenet",
        [5],
        ["fgsm"],
    ),
    "imagenet_new_classes_deepfool_caffenet": (
        "deepfool_caffenet",
        "caffenet",
        [8],
        ["deepfool"],
    ),
    "imagenet_new_classes_cw_l2_inception_v3": (
        "cw_l2_inception_v3",
        "inception_v3",
        [14],
        ["cw_l2"],
    ),
}

FASHION_MNIST_EXPERIMENTS = {
    "fashion_mnist_fgsm_m1": ("fgsm_m1", "m1", [1], ["fgsm"]),
    "fashion_mnist_cw_l2_m2": ("cw_l2_m2", "m2", [9], ["cw_l2_nn_robust"]),
}


def _consolidated_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )


def test_table10_config_declares_one_experiment_per_model_group() -> None:
    """Table 10 should be executable independently by model group."""
    config = _consolidated_config()
    experiments = config["experiments"]

    for experiment_name, (model_group, dataset_label, row_numbers) in TABLE10_EXPERIMENTS.items():
        experiment = experiments[experiment_name]
        expected_output_dir = "results/experiments/table_10/{0}".format(model_group)
        if experiment_name in {"table_10_m1", "table_10_m2"}:
            expected_output_dir = "results/mnist/article_reproduction/{0}".format(
                experiment_name
            )
        if experiment_name == "table_10_googlenet":
            expected_output_dir = "results/experiments/table_10/imagenet/googlenet"

        assert experiment["kind"] == "table_10_group"
        assert experiment["model_group"] == model_group
        assert experiment["dataset_label"] == dataset_label
        assert experiment["output_dir"] == expected_output_dir
        assert [row["no"] for row in experiment["rows"]] == row_numbers


def test_table10_googlenet_config_enables_deepfool_metrics() -> None:
    """GoogLeNet row 7 should be configured for real DeepFool evaluation."""
    config = _consolidated_config()
    experiment = config["experiments"]["table_10_googlenet"]
    row = next(item for item in experiment["rows"] if item["no"] == 7)

    assert experiment["dataset"]["split"] == "test"
    assert experiment["dataset"]["images_dir"] == "data/imagenet/test"
    assert experiment["dataset"]["n_samples"] == "all"
    assert experiment["dataset"]["class_indices"] == {
        "cab": 468,
        "panda": 388,
        "zebra": 340,
    }
    assert experiment["evaluation"]["n_samples"] == "all"
    assert experiment["model"]["name"] == "googlenet_caffe"
    assert (
        experiment["model"]["deploy_proto"]
        == "artifacts/models/imagenet/googlenet/deploy_original.prototxt"
    )
    assert (
        experiment["model"]["attack_deploy_proto"]
        == "artifacts/models/imagenet/googlenet/deploy_removeSoftmax.prototxt"
    )
    assert experiment["model"]["use_gpu"] is True
    assert experiment["filter"]["type"] == "proposed_detection_filter"
    assert [item["status"] for item in experiment["rows"]] == [
        "implemented",
        "implemented",
        "implemented",
    ]
    assert row["status"] == "implemented"
    assert row["attack"]["name"] == "deepfool"
    assert row["attack"]["num_classes"] == 10


def test_table10_googlenet_builder_passes_attack_deploy_to_wrapper(
    monkeypatch,
    tmp_path,
) -> None:
    """Table 10 should configure the wrapper with separate prediction/attack prototxts."""
    captured = {}

    def fake_wrapper(**kwargs):
        captured.update(kwargs)
        return object()

    model_dir = tmp_path / "googlenet"
    model_dir.mkdir()
    deploy = model_dir / "deploy_original.prototxt"
    attack_deploy = model_dir / "deploy_removeSoftmax.prototxt"
    caffemodel = model_dir / "bvlc_googlenet.caffemodel"
    for path in (deploy, attack_deploy, caffemodel):
        path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(table_10_module, "GoogLeNetCaffeWrapper", fake_wrapper)

    table_10_module.build_table_10_googlenet_model(
        {
            "model": {
                "model_dir": str(model_dir),
                "deploy_proto": str(deploy),
                "attack_deploy_proto": str(attack_deploy),
                "caffemodel": str(caffemodel),
                "batch_size": 8,
                "use_gpu": False,
            }
        }
    )

    assert captured["deploy_prototxt"] == str(deploy)
    assert captured["attack_deploy_prototxt"] == str(attack_deploy)
    assert captured["caffemodel"] == str(caffemodel)


def test_table10_caffenet_config_uses_separate_deploy_files() -> None:
    """CaffeNet row 8 should be wired structurally for DeepFool execution."""
    config = _consolidated_config()
    experiment = config["experiments"]["table_10_caffenet"]
    row = experiment["rows"][0]

    assert experiment["kind"] == "table_10_group"
    assert experiment["dataset"]["image_size"] == 227
    assert experiment["dataset"]["image_shape"] == [227, 227, 3]
    assert experiment["model"]["name"] == "caffenet"
    assert (
        experiment["model"]["deploy_proto"]
        == "artifacts/models/imagenet/caffenet/deploy_original.prototxt"
    )
    assert (
        experiment["model"]["attack_deploy_proto"]
        == "artifacts/models/imagenet/caffenet/deploy_removeSoftmax.prototxt"
    )
    assert experiment["model"]["use_gpu"] is True
    assert row["no"] == 8
    assert row["attack_model"] == "DeepFool/CaffeNet"
    assert row["status"] == "implemented"
    assert "blocked_reason" not in row
    assert row["attack"]["name"] == "deepfool"
    assert row["attack"]["max_iter"] == 50
    assert row["attack"]["overshoot"] == 0.02
    assert row["attack"]["clip_min"] == 0.0
    assert row["attack"]["clip_max"] == 1.0
    assert row["attack"]["num_classes"] == 10


def test_table10_caffenet_builder_passes_attack_deploy_to_wrapper(
    monkeypatch,
    tmp_path,
) -> None:
    """Table 10 should configure CaffeNet with separate prediction/attack prototxts."""
    captured = {}

    def fake_wrapper(**kwargs):
        captured.update(kwargs)
        return object()

    model_dir = tmp_path / "caffenet"
    model_dir.mkdir()
    deploy = model_dir / "deploy_original.prototxt"
    attack_deploy = model_dir / "deploy_removeSoftmax.prototxt"
    caffemodel = model_dir / "bvlc_reference_caffenet.caffemodel"
    for path in (deploy, attack_deploy, caffemodel):
        path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(table_10_module, "CaffeNetCaffeWrapper", fake_wrapper)

    table_10_module.build_table_10_caffenet_model(
        {
            "model": {
                "model_dir": str(model_dir),
                "deploy_proto": str(deploy),
                "attack_deploy_proto": str(attack_deploy),
                "caffemodel": str(caffemodel),
                "batch_size": 8,
                "use_gpu": False,
            }
        }
    )

    assert captured["deploy_prototxt"] == str(deploy)
    assert captured["attack_deploy_prototxt"] == str(attack_deploy)
    assert captured["caffemodel"] == str(caffemodel)


def test_table10_imagenet_groups_use_test_dataset() -> None:
    """Every ImageNet Table 10 group should point at the test split."""
    config = _consolidated_config()
    for experiment_name in (
        "table_10_googlenet",
        "table_10_caffenet",
    ):
        dataset = config["experiments"][experiment_name]["dataset"]
        assert dataset["name"] == "imagenet"
        assert dataset["split"] == "test"
        assert dataset["images_dir"] == "data/imagenet/test"
        assert dataset["class_indices"] == {
            "cab": 468,
            "panda": 388,
            "zebra": 340,
        }

    inception_dataset = config["experiments"]["table_10_inception_v3"]["dataset"]
    assert inception_dataset["name"] == "imagenet"
    assert inception_dataset["split"] == "test"
    assert inception_dataset["images_dir"] == "data/inceptionV3"
    assert inception_dataset["class_indices"] == {
        "zebra": 80,
        "panda": 169,
        "cab": 267,
    }
    assert inception_dataset["shuffle"] is False
    assert inception_dataset["class_order"] == ["zebra", "panda", "cab"]
    assert inception_dataset["class_quotas"] == {
        "zebra": 40,
        "panda": 40,
        "cab": 20,
    }


def test_table10_inception_v3_config_enables_cw_rows() -> None:
    """Inception v3 should be configured as a real Table 10 ImageNet group."""
    config = _consolidated_config()
    experiment = config["experiments"]["table_10_inception_v3"]

    assert experiment["kind"] == "table_10_group"
    assert experiment["dataset"]["image_size"] == 299
    assert experiment["dataset"]["image_shape"] == [299, 299, 3]
    assert experiment["dataset"]["value_range"] == [-0.5, 0.5]
    assert "n_samples" not in experiment["dataset"]
    assert experiment["dataset"]["shuffle"] is False
    assert experiment["dataset"]["require_clean_correct"] is True
    assert experiment["dataset"]["class_order"] == ["zebra", "panda", "cab"]
    assert experiment["dataset"]["class_quotas"] == {
        "zebra": 40,
        "panda": 40,
        "cab": 20,
    }
    assert experiment["model"]["name"] == "inception_v3"
    assert (
        experiment["model"]["graph_path"]
        == "artifacts/models/imagenet/inceptionv3/classify_image_graph_def.pb"
    )
    assert experiment["model"]["input_map_name"] == "ResizeBilinear:0"
    assert [row["no"] for row in experiment["rows"]] == [14, 15, 16, 17, 18, 20]
    assert [row["status"] for row in experiment["rows"]] == ["implemented"] * 6
    assert [row["attack"]["name"] for row in experiment["rows"]] == [
        "cw_l2",
        "cw_l2",
        "cw_l2",
        "cw_l2",
        "cw_l2",
        "cw_linf",
    ]
    assert [row["attack"].get("kappa") for row in experiment["rows"][:5]] == [
        0.0,
        0.5,
        1.0,
        2.0,
        4.0,
    ]
    for row in experiment["rows"][:5]:
        assert row["attack"] == {
            "name": "cw_l2",
            "kappa": row["attack"]["kappa"],
            "batch_size": 1,
            "max_iterations": 1000,
            "learning_rate": 0.01,
            "binary_search_steps": 9,
            "initial_const": 0.001,
            "abort_early": True,
            "targeted": False,
            "clip_min": -0.5,
            "clip_max": 0.5,
        }


def test_imagenet_new_classes_configs_are_separate_table10_experiments() -> None:
    """The new ImageNet dataset should expose one public command per selected row."""
    config = _consolidated_config()
    experiments = config["experiments"]

    assert "imagenet_new_classes_image_models" not in experiments
    assert "imagenet_new_classes" not in experiments

    for experiment_name, (
        output_leaf,
        model_group,
        row_numbers,
        attack_names,
    ) in IMAGENET_NEW_CLASS_EXPERIMENTS.items():
        experiment = experiments[experiment_name]
        dataset = experiment["dataset"]

        assert experiment["kind"] == "table_10_group"
        assert experiment["output_dir"] == (
            "results/experiments/imagenet_new_classes/{0}".format(output_leaf)
        )
        assert experiment["model_group"] == model_group
        assert experiment["dataset_group"] == "imagenet_new_classes"
        assert experiment["dataset_label"] == "ImageNet-NewClasses"
        assert experiment["output"]["manifest"] is True
        assert dataset["name"] == "imagenet"
        assert dataset["split"] == "new_test"
        assert dataset["images_dir"] == "data/imagenet/new_test"
        assert dataset["require_clean_correct"] is True
        assert dataset["class_order"] == ["ambulance", "scholar_bus", "soccer_ball"]
        assert dataset["class_quotas"] == {
            "ambulance": 40,
            "scholar_bus": 40,
            "soccer_ball": 20,
        }
        assert [row["no"] for row in experiment["rows"]] == row_numbers
        assert [row["attack"]["name"] for row in experiment["rows"]] == attack_names


def test_imagenet_new_classes_configs_use_model_specific_labels() -> None:
    """Caffe models and Inception use their own ImageNet label conventions."""
    config = _consolidated_config()
    experiments = config["experiments"]

    caffe_labels = {
        "ambulance": 407,
        "scholar_bus": 779,
        "soccer_ball": 805,
    }
    assert (
        experiments["imagenet_new_classes_fgsm_googlenet"]["dataset"]["class_indices"]
        == caffe_labels
    )
    assert (
        experiments["imagenet_new_classes_deepfool_caffenet"]["dataset"]["class_indices"]
        == caffe_labels
    )
    assert experiments["imagenet_new_classes_cw_l2_inception_v3"]["dataset"][
        "class_indices"
    ] == {
        "ambulance": 265,
        "scholar_bus": 962,
        "soccer_ball": 222,
    }


def test_fashion_mnist_configs_are_separate_table10_experiments() -> None:
    """Fashion-MNIST should expose one public command per selected Table 10 row."""
    config = _consolidated_config()
    experiments = config["experiments"]

    for experiment_name, (
        output_leaf,
        model_group,
        row_numbers,
        attack_names,
    ) in FASHION_MNIST_EXPERIMENTS.items():
        experiment = experiments[experiment_name]
        dataset = experiment["dataset"]

        assert experiment["kind"] == "table_10_group"
        assert experiment["output_dir"] == (
            "results/experiments/fashion_mnist/{0}".format(output_leaf)
        )
        assert experiment["model_group"] == model_group
        assert experiment["dataset_group"] == "fashion_mnist"
        assert experiment["dataset_label"] == "Fashion-MNIST"
        assert experiment["output"]["manifest"] is True
        assert dataset["name"] == "fashion_mnist"
        assert dataset["domain"] == "mnist_compatible"
        assert dataset["csv_path"] == "data/fashion_mnist/fashion-mnist_test.csv"
        assert dataset["split_strategy"] == {
            "name": "balanced_by_class",
            "train_samples": 9000,
            "evaluation_samples": 1000,
            "train_class_quota": 900,
            "evaluation_class_quota": 100,
        }
        assert experiment["checkpoint_training"] == {
            "source_csv": "data/fashion_mnist/fashion-mnist_test.csv",
            "split_name": "train",
            "selection": {
                "method": "first_n_per_class",
                "per_class_start": 0,
                "per_class_end": 900,
                "class_quota": 900,
                "total_samples": 9000,
            },
            "validation_sample": {
                "split_name": "evaluation",
                "method": "next_n_per_class",
                "per_class_start": 900,
                "per_class_end": 1000,
                "class_quota": 100,
                "total_samples": 1000,
                "excludes_training": True,
            },
        }
        assert dataset["image_shape"] == [28, 28, 1]
        assert set(dataset["class_quotas"].values()) == {100}
        assert [row["no"] for row in experiment["rows"]] == row_numbers
        assert [row["attack"]["name"] for row in experiment["rows"]] == attack_names


def test_table10_fashion_mnist_row_computes_metrics(monkeypatch) -> None:
    """Fashion-MNIST rows should discard clean errors and compute Table 10 metrics."""
    graph = {
        "sess": type("Session", (), {"close": lambda self: None})(),
        "model": object(),
        "x": object(),
    }
    images = table_10_module.np.asarray(
        [[[[0.0]]], [[[0.1]]], [[[1.0]]], [[[1.1]]]],
        dtype=table_10_module.np.float32,
    )
    labels = table_10_module.np.asarray([0, 0, 1, 1], dtype=table_10_module.np.int64)
    metadata = {
        "name": "fashion_mnist",
        "domain": "mnist_compatible",
        "split": "test",
        "csv_path": "data/fashion_mnist/fashion-mnist_test.csv",
        "split_strategy": {"name": "balanced_by_class"},
        "image_shape": [28, 28, 1],
        "value_range": {"min": 0.0, "max": 1.0},
        "class_order": ["first", "second"],
        "class_quotas": {"first": 1, "second": 1},
        "candidate_counts": {"first": 2, "second": 2},
    }
    predictions = iter(
        [
            table_10_module.np.asarray([0, 9, 1, 9], dtype=table_10_module.np.int64),
            table_10_module.np.asarray([1, 1], dtype=table_10_module.np.int64),
            table_10_module.np.asarray([0, 1], dtype=table_10_module.np.int64),
            table_10_module.np.asarray([0, 1], dtype=table_10_module.np.int64),
        ]
    )

    monkeypatch.setattr(
        table_10_module,
        "_build_table_10_fashion_mnist_graph",
        lambda config: graph,
    )
    monkeypatch.setattr(
        table_10_module,
        "load_fashion_mnist_evaluation_split",
        lambda dataset_config: (images, labels, dict(metadata)),
    )
    monkeypatch.setattr(
        table_10_module,
        "_mnist_graph_predict",
        lambda graph, images, batch_size: next(predictions),
    )
    monkeypatch.setattr(table_10_module, "_table_10_filter", lambda config: (lambda image: image))

    def fake_generate_attack(name, **kwargs):
        assert name == "fgsm"
        assert kwargs["eps"] == 0.2
        return kwargs["images"].copy()

    monkeypatch.setattr(table_10_module, "generate_attack", fake_generate_attack)

    group_config = {
        "dataset": {
            "name": "fashion_mnist",
            "domain": "mnist_compatible",
            "class_order": ["first", "second"],
            "class_indices": {"first": 0, "second": 1},
            "class_quotas": {"first": 2, "second": 2},
            "require_clean_correct": True,
        },
        "model_group": "m1",
        "model": {"family": "mnist", "dataset_name": "fashion_mnist"},
        "evaluation": {"batch_size": 2},
    }

    result = evaluate_table_10_fashion_mnist_row(
        group_config,
        {
            "no": 1,
            "attack_model": "FGSM (ε=0.2)/M1",
            "status": "implemented",
            "attack": {"name": "fgsm", "epsilon": 0.2},
        },
    )

    assert result["metrics"] == {
        "num_failures": 1,
        "tp": 1,
        "fn": 0,
        "fp": 0,
        "rtp": 1,
        "rtp_percent": 100.0,
        "recall": 100.0,
        "precision": 100.0,
        "f1": 100.0,
    }
    assert group_config["_table_10_dataset_summary"]["clean_errors"] == {
        "first": 1,
        "second": 1,
    }
    assert group_config["_table_10_dataset_summary"]["clean_correct"] == {
        "first": 1,
        "second": 1,
    }
    assert group_config["_table_10_dataset_summary"]["selected_clean_correct"] == {
        "first": 1,
        "second": 1,
    }
    assert (
        group_config["_table_10_dataset_summary"]["selection_policy"]
        == "discard_clean_errors"
    )


def test_table10_fashion_mnist_manifest_records_dataset_and_attack(
    monkeypatch,
    tmp_path,
) -> None:
    """Fashion-MNIST manifests should preserve CSV split and attack metadata."""
    dataset_summary = {
        "name": "fashion_mnist",
        "domain": "mnist_compatible",
        "split": "test",
        "csv_path": "data/fashion_mnist/fashion-mnist_test.csv",
        "split_strategy": {
            "name": "balanced_by_class",
            "train_samples": 9000,
            "evaluation_samples": 1000,
            "train_class_quota": 900,
            "evaluation_class_quota": 100,
        },
        "training_sample": {
            "split_name": "train",
            "method": "first_n_per_class",
            "per_class_start": 0,
            "per_class_end": 900,
            "class_quota": 900,
            "total_samples": 9000,
        },
        "evaluation_sample": {
            "split_name": "evaluation",
            "method": "next_n_per_class",
            "per_class_start": 900,
            "per_class_end": 1000,
            "class_quota": 100,
            "total_samples": 1000,
            "excludes_training": True,
        },
        "image_shape": [28, 28, 1],
        "value_range": {"min": 0.0, "max": 1.0},
        "class_order": ["first"],
        "class_quotas": {"first": 100},
        "candidate_counts": {"first": 1000},
        "candidates_read": {"first": 100},
        "clean_errors": {"first": 0},
        "clean_correct": {"first": 100},
        "selected_clean_correct": {"first": 100},
        "selection_policy": "discard_clean_errors",
    }

    def fake_evaluate(group_config, row_config):
        group_config["_table_10_dataset_summary"] = dataset_summary
        return {
            "metrics": {
                "num_failures": 0,
                "tp": 1,
                "fn": 0,
                "fp": 0,
                "rtp": 1,
                "rtp_percent": 100.0,
                "recall": 100.0,
                "precision": 100.0,
                "f1": 100.0,
            }
        }

    monkeypatch.setattr(
        table_10_module,
        "evaluate_table_10_fashion_mnist_row",
        fake_evaluate,
    )

    run_table_10_group(
        {
            "experiment_id": "fashion_mnist_fgsm_m1",
            "kind": "table_10_group",
            "dataset": {"name": "fashion_mnist"},
            "checkpoint_training": {
                "source_csv": "data/fashion_mnist/fashion-mnist_test.csv",
                "split_name": "train",
                "selection": {
                    "method": "first_n_per_class",
                    "per_class_start": 0,
                    "per_class_end": 900,
                    "class_quota": 900,
                    "total_samples": 9000,
                },
                "validation_sample": {
                    "split_name": "evaluation",
                    "method": "next_n_per_class",
                    "per_class_start": 900,
                    "per_class_end": 1000,
                    "class_quota": 100,
                    "total_samples": 1000,
                    "excludes_training": True,
                },
            },
            "model_group": "m1",
            "dataset_group": "fashion_mnist",
            "dataset_label": "Fashion-MNIST",
            "output": {"dir": str(tmp_path), "manifest": True},
            "rows": [
                {
                    "no": 1,
                    "attack_model": "FGSM (ε=0.2)/M1",
                    "status": "implemented",
                    "attack": {"name": "fgsm", "epsilon": 0.2},
                }
            ],
        }
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "fashion_mnist_fgsm_m1"
    assert manifest["dataset_group"] == "fashion_mnist"
    assert manifest["dataset"]["csv_path"] == "data/fashion_mnist/fashion-mnist_test.csv"
    assert manifest["dataset"]["split_strategy"]["train_samples"] == 9000
    assert manifest["dataset"]["training_sample"]["per_class_end"] == 900
    assert manifest["dataset"]["evaluation_sample"]["per_class_start"] == 900
    assert manifest["dataset"]["checkpoint_training"]["selection"]["total_samples"] == 9000
    assert manifest["dataset"]["selected_clean_correct"] == {"first": 100}
    assert manifest["dataset"]["selection_policy"] == "discard_clean_errors"
    assert manifest["rows"] == [
        {
            "no": 1,
            "attack_model": "FGSM (ε=0.2)/M1",
            "status": "completed",
            "attack": {"name": "fgsm", "epsilon": 0.2},
        }
    ]


def test_table_10_schema_matches_paper_fields() -> None:
    assert TABLE_10_SCHEMA == [
        "no",
        "attack_model",
        "dataset",
        "num_failures",
        "tp",
        "fn",
        "fp",
        "rtp",
        "rtp_percent",
        "recall",
        "precision",
        "f1",
    ]


def test_build_pending_table_10_row() -> None:
    row = build_pending_table_10_row(
        no=5,
        attack_model="FGSM (\u03b5=1/255)/GoogLeNet",
        dataset="ImageNet",
    )

    assert row == {
        "no": 5,
        "attack_model": "FGSM (\u03b5=1/255)/GoogLeNet",
        "dataset": "ImageNet",
        "num_failures": None,
        "tp": None,
        "fn": None,
        "fp": None,
        "rtp": None,
        "rtp_percent": None,
        "recall": None,
        "precision": None,
        "f1": None,
    }


def test_table10_runner_writes_official_schema_without_manifest(tmp_path) -> None:
    """A Table 10 group should write only metrics CSV and JSON."""
    rows = run_table_10_group(
        {
            "experiment_id": "table_10_test",
            "kind": "table_10_group",
            "dataset": {"name": "imagenet"},
            "model_group": "googlenet",
            "dataset_label": "ImageNet",
            "rows": [
                {
                    "no": 5,
                    "attack_model": "FGSM (\u03b5=1/255)/GoogLeNet",
                    "status": "planned",
                },
                {
                    "no": 7,
                    "attack_model": "DeepFool/GoogLeNet",
                    "status": "planned",
                },
            ],
            "output": {"dir": str(tmp_path), "csv": "metrics.csv", "json": "metrics.json"},
        }
    )

    assert [row["no"] for row in rows] == [5, 7]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["metrics.csv", "metrics.json"]
    with (tmp_path / "metrics.csv").open("r", encoding="utf-8") as handle:
        csv_rows = list(csv.reader(handle))
    assert csv_rows[0] == TABLE_10_SCHEMA
    assert csv_rows[1] == [
        "5",
        "FGSM (\u03b5=1/255)/GoogLeNet",
        "ImageNet",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["table"] == 10
    assert metrics_payload["dataset_group"] == "imagenet"
    assert metrics_payload["model_group"] == "googlenet"
    assert metrics_payload["rows"][0]["num_failures"] is None
    assert metrics_payload["rows"][0]["no"] == 5


def test_table10_caffenet_blocked_group_writes_manifest(tmp_path) -> None:
    """CaffeNet should keep blocked reasons in manifest, not official metrics."""
    run_table_10_group(
        {
            "experiment_id": "table_10_caffenet",
            "kind": "table_10_group",
            "dataset": {"name": "imagenet"},
            "model_group": "caffenet",
            "dataset_label": "ImageNet",
            "rows": [
                {
                    "no": 8,
                    "attack_model": "DeepFool/CaffeNet",
                    "status": "blocked",
                    "blocked_reason": "CaffeNet is not implemented.",
                }
            ],
            "output": {"dir": str(tmp_path), "csv": "metrics.csv", "json": "metrics.json"},
        }
    )

    assert "CaffeNet is not implemented" not in (tmp_path / "metrics.csv").read_text(
        encoding="utf-8"
    )
    assert "CaffeNet is not implemented" not in (tmp_path / "metrics.json").read_text(
        encoding="utf-8"
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_group"] == "caffenet"
    assert manifest["rows"][0] == {
        "no": 8,
        "attack_model": "DeepFool/CaffeNet",
        "status": "blocked",
        "blocked_reason": "CaffeNet is not implemented.",
    }


def test_table10_output_manifest_flag_writes_experiment_manifest(monkeypatch, tmp_path) -> None:
    """New dataset experiments should be able to force a manifest for any model."""
    dataset_summary = {
        "classes": ["ambulance", "scholar_bus", "soccer_ball"],
        "quotas": {"ambulance": 40, "scholar_bus": 40, "soccer_ball": 20},
        "candidate_counts": {"ambulance": 44, "scholar_bus": 45, "soccer_ball": 22},
        "candidates_read": {"ambulance": 42, "scholar_bus": 43, "soccer_ball": 20},
        "clean_errors": {"ambulance": 2, "scholar_bus": 3, "soccer_ball": 0},
        "clean_correct": {"ambulance": 40, "scholar_bus": 40, "soccer_ball": 20},
    }

    def fake_evaluate(group_config, row_config):
        group_config["_table_10_dataset_summary"] = dataset_summary
        return {
            "metrics": {
                "num_failures": 0,
                "tp": 1,
                "fn": 0,
                "fp": 0,
                "rtp": 1,
                "rtp_percent": 100.0,
                "recall": 100.0,
                "precision": 100.0,
                "f1": 100.0,
            }
        }

    monkeypatch.setattr(table_10_module, "evaluate_table_10_googlenet_row", fake_evaluate)

    run_table_10_group(
        {
            "experiment_id": "imagenet_new_classes_fgsm_googlenet",
            "kind": "table_10_group",
            "dataset": {"name": "imagenet"},
            "model_group": "googlenet",
            "dataset_group": "imagenet_new_classes",
            "dataset_label": "ImageNet-NewClasses",
            "output": {"dir": str(tmp_path), "manifest": True},
            "rows": [
                {
                    "no": 5,
                    "attack_model": "FGSM (ε=1/255)/GoogLeNet",
                    "status": "implemented",
                    "attack": {"name": "fgsm"},
                }
            ],
        }
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "imagenet_new_classes_fgsm_googlenet"
    assert manifest["dataset_group"] == "imagenet_new_classes"
    assert manifest["model_group"] == "googlenet"
    assert manifest["dataset"] == dataset_summary
    assert manifest["rows"] == [
        {
            "no": 5,
            "attack_model": "FGSM (ε=1/255)/GoogLeNet",
            "status": "completed",
        }
    ]


def test_table10_inception_v3_group_writes_manifest_for_blocked_rows(tmp_path) -> None:
    """Inception v3 should keep row execution status in a side manifest."""
    rows = run_table_10_group(
        {
            "experiment_id": "table_10_inception_v3",
            "kind": "table_10_group",
            "dataset": {"name": "imagenet"},
            "model_group": "inception_v3",
            "dataset_label": "ImageNet",
            "output_dir": str(tmp_path),
            "rows": [
                {
                    "no": 14,
                    "attack_model": "CW L2 (κ=0.0)/Inception v3",
                    "status": "blocked",
                    "blocked_reason": "missing graph",
                }
            ],
        }
    )

    assert rows[0]["no"] == 14
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "manifest.json",
        "metrics.csv",
        "metrics.json",
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_group"] == "inception_v3"
    assert manifest["rows"][0]["blocked_reason"] == "missing graph"


def test_table_10_googlenet_group_generates_three_rows(tmp_path) -> None:
    config = {
        "kind": "table_10_group",
        "output_dir": str(tmp_path),
        "dataset": {"name": "imagenet"},
        "model": {"name": "googlenet"},
        "model_group": "googlenet",
        "dataset_label": "ImageNet",
        "rows": [
            {
                "no": 5,
                "attack_model": "FGSM (\u03b5=1/255)/GoogLeNet",
                "status": "planned",
                "attack": {"name": "fgsm", "epsilon": 1 / 255},
            },
            {
                "no": 6,
                "attack_model": "FGSM (\u03b5=2/255)/GoogLeNet",
                "status": "planned",
                "attack": {"name": "fgsm", "epsilon": 2 / 255},
            },
            {
                "no": 7,
                "attack_model": "DeepFool/GoogLeNet",
                "status": "planned",
                "attack": {"name": "deepfool"},
            },
        ],
    }

    rows = run_table_10_group(config)

    assert [row["no"] for row in rows] == [5, 6, 7]
    assert all(row["dataset"] == "ImageNet" for row in rows)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "manifest.json").exists()


class Table10DummyModel:
    """Small model exposing the ImageNet/GoogLeNet row-evaluator contract."""

    def predict_preprocessed_batch(self, images):
        batch = []
        for image in images:
            label = int(round(float(image.reshape(-1)[0])))
            scores = [0.0, 0.0, 0.0, 0.0, 0.0]
            scores[label] = 1.0
            batch.append(scores)
        return batch

    def predict_batch(self, images):
        return self.predict_preprocessed_batch(images)

    def predict_label(self, images):
        return [
            int(max(range(len(scores)), key=lambda index: scores[index]))
            for scores in self.predict_preprocessed_batch(images)
        ]

    def gradient(self, image, class_id):
        return image


def test_table10_googlenet_deepfool_row_computes_metrics(monkeypatch) -> None:
    """Implemented row 7 should use attack, filter, and detector metric helpers."""
    images = [
        [[[1.0]]],  # successful attack, detected and recovered
        [[[2.0]]],  # attack failure
        [[[3.0]]],  # clean error
    ]
    labels = [1, 2, 99]

    monkeypatch.setattr(
        table_10_module,
        "build_table_10_googlenet_model",
        lambda config: Table10DummyModel(),
    )
    monkeypatch.setattr(
        table_10_module,
        "_load_table_10_googlenet_images",
        lambda config, model: (
            table_10_module.np.asarray(images, dtype=table_10_module.np.float32),
            table_10_module.np.asarray(labels, dtype=table_10_module.np.int32),
        ),
    )

    def fake_generate_attack(name, model, images, labels, **kwargs):
        assert name == "deepfool"
        assert kwargs["max_iter"] == 3
        adversarial = images.copy()
        adversarial[0, ...] = 2.0 if float(images[0].reshape(-1)[0]) == 1.0 else images[0]
        return adversarial

    def fake_build_filter(config):
        def filter_fn(image):
            return table_10_module.np.where(image == 2.0, 1.0, image)

        return "test_filter", filter_fn, {}

    monkeypatch.setattr(table_10_module, "generate_attack", fake_generate_attack)
    monkeypatch.setattr(table_10_module, "build_filter_from_config", fake_build_filter)

    result = evaluate_table_10_googlenet_row(
        {
            "dataset": {"name": "imagenet"},
            "model": {"name": "googlenet_caffe"},
            "filter": {"name": "test_filter", "type": "proposed_detection_filter"},
        },
        {
            "no": 7,
            "attack_model": "DeepFool/GoogLeNet",
            "status": "implemented",
            "attack": {"name": "deepfool", "max_iter": 3},
        },
    )

    assert result["metrics"] == {
        "num_failures": 1,
        "tp": 1,
        "fn": 0,
        "fp": 0,
        "rtp": 1,
        "rtp_percent": 100.0,
        "recall": 100.0,
        "precision": 100.0,
        "f1": 100.0,
    }


def test_table10_googlenet_fgsm_row_uses_caffe_attack(monkeypatch) -> None:
    """FGSM GoogLeNet rows should use the Caffe-scale ImageNet attack helper."""
    calls = []

    def fake_generate_fgsm_caffe_image(
        model,
        image,
        class_id,
        epsilon_255,
        clip_min,
        clip_max,
    ):
        calls.append((class_id, epsilon_255, clip_min, clip_max))
        return image + 1.0

    monkeypatch.setattr(
        table_10_module,
        "generate_fgsm_caffe_image",
        fake_generate_fgsm_caffe_image,
    )

    adversarial = table_10_module._generate_table_10_adversarial(
        attack_name="fgsm",
        row_config={"attack": {"name": "fgsm", "epsilon": 1 / 255, "clip_max": 255.0}},
        model=Table10DummyModel(),
        clean_image=table_10_module.np.zeros((1, 1, 1), dtype=table_10_module.np.float32),
        true_label=1,
        clean_pred=1,
    )

    assert calls == [(1, 1.0, 0.0, 255.0)]
    assert float(adversarial.reshape(-1)[0]) == 1.0


def test_table10_inception_v3_row_computes_cw_metrics(monkeypatch) -> None:
    """Inception v3 implemented rows should use the shared ImageNet evaluator."""
    images = [
        [[[1.0]]],
        [[[2.0]]],
        [[[3.0]]],
    ]
    labels = [1, 2, 99]

    monkeypatch.setattr(
        table_10_module,
        "build_table_10_inception_v3_model",
        lambda config: Table10DummyModel(),
    )
    monkeypatch.setattr(
        table_10_module,
        "_load_table_10_imagenet_images",
        lambda config, model: (
            table_10_module.np.asarray(images, dtype=table_10_module.np.float32),
            table_10_module.np.asarray(labels, dtype=table_10_module.np.int32),
        ),
    )

    def fake_generate_attack(name, model, images, labels, **kwargs):
        assert name == "cw_l2"
        assert kwargs["kappa"] == 0.5
        adversarial = images.copy()
        adversarial[0, ...] = 2.0 if float(images[0].reshape(-1)[0]) == 1.0 else images[0]
        return adversarial

    def fake_build_filter(config):
        def filter_fn(image):
            return table_10_module.np.where(image == 2.0, 1.0, image)

        return "test_filter", filter_fn, {}

    monkeypatch.setattr(table_10_module, "generate_attack", fake_generate_attack)
    monkeypatch.setattr(table_10_module, "build_filter_from_config", fake_build_filter)

    result = evaluate_table_10_imagenet_row(
        {
            "dataset": {"name": "imagenet"},
            "model_group": "inception_v3",
            "model": {"name": "inception_v3"},
            "filter": {"name": "test_filter", "type": "proposed_detection_filter"},
        },
        {
            "no": 15,
            "attack_model": "CW L2 (κ=0.5)/Inception v3",
            "status": "implemented",
            "attack": {"name": "cw_l2", "kappa": 0.5},
        },
    )

    assert result["metrics"]["num_failures"] == 1
    assert result["metrics"]["tp"] == 1
    assert result["metrics"]["fn"] == 0
    assert result["metrics"]["fp"] == 0


def test_table10_metrics_use_clean_predictions_for_false_positive() -> None:
    """Detected adversarials must not become FP unless the clean prediction changes."""
    metrics = table_10_module._table_10_metrics_from_records(
        [
            {
                "detected": True,
                "corrected": True,
                "false_positive": False,
            },
            {
                "detected": True,
                "corrected": False,
                "false_positive": False,
            },
        ]
    )

    assert metrics["tp"] == 2
    assert metrics["fn"] == 0
    assert metrics["fp"] == 0
    assert metrics["precision"] == 100.0


def test_table10_metrics_match_line_15_formulas() -> None:
    """Recall, precision, F1, and RTP% should match the Table 10 formulas."""
    records = [
        {"detected": True, "corrected": True, "false_positive": False}
        for _ in range(98)
    ]
    records.extend(
        {"detected": True, "corrected": False, "false_positive": True}
        for _ in range(2)
    )

    metrics = table_10_module._table_10_metrics_from_records(records)

    assert metrics["num_failures"] == 0
    assert metrics["tp"] == 100
    assert metrics["fn"] == 0
    assert metrics["fp"] == 2
    assert metrics["rtp"] == 98
    assert metrics["recall"] == pytest.approx(100.0)
    assert metrics["precision"] == pytest.approx(98.0392157)
    assert metrics["f1"] == pytest.approx(99.0099010)
    assert metrics["rtp_percent"] == pytest.approx(98.0)


def test_table10_inception_loader_fills_quotas_with_clean_correct_samples(
    monkeypatch,
    tmp_path,
) -> None:
    """Clean errors should be skipped without shrinking the final Inception sample."""
    Image = pytest.importorskip("PIL.Image")
    images_dir = tmp_path / "images"
    class_pixels = {
        "zebra": [9, 1, 1],
        "panda": [2],
        "cab": [3],
    }
    for class_name, pixels in class_pixels.items():
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True)
        for index, pixel in enumerate(pixels):
            image = table_10_module.np.full(
                (1, 1, 3),
                int(pixel),
                dtype=table_10_module.np.uint8,
            )
            Image.fromarray(image, mode="RGB").save(
                str(class_dir / "{0:03d}.png".format(index))
            )

    def fake_preprocess(model, image_size):
        def preprocess(image):
            pixel = int(round(float(image[0, 0, 0]) * 255.0))
            return table_10_module.np.asarray([[[pixel]]], dtype=table_10_module.np.float32)

        return preprocess

    monkeypatch.setattr(table_10_module, "_preprocess_table_10_image", fake_preprocess)
    monkeypatch.setattr(
        table_10_module,
        "_predict_one",
        lambda model, image: int(round(float(image.reshape(-1)[0]))),
    )

    config = {
        "model_group": "inception_v3",
        "dataset": {
            "name": "imagenet",
            "images_dir": str(images_dir),
            "image_size": 1,
            "shuffle": False,
            "require_clean_correct": True,
            "class_order": ["zebra", "panda", "cab"],
            "class_indices": {"zebra": 1, "panda": 2, "cab": 3},
            "class_quotas": {"zebra": 2, "panda": 1, "cab": 1},
        },
    }
    images, labels = table_10_module._load_table_10_imagenet_images(
        config,
        Table10DummyModel(),
    )

    assert images.shape == (4, 1, 1, 1)
    assert labels.tolist() == [1, 1, 2, 3]
    assert images.reshape(-1).tolist() == [1.0, 1.0, 2.0, 3.0]
    assert config["_table_10_dataset_summary"] == {
        "classes": ["zebra", "panda", "cab"],
        "quotas": {"zebra": 2, "panda": 1, "cab": 1},
        "candidate_counts": {"zebra": 3, "panda": 1, "cab": 1},
        "candidates_read": {"zebra": 3, "panda": 1, "cab": 1},
        "clean_errors": {"zebra": 1, "panda": 0, "cab": 0},
        "clean_correct": {"zebra": 2, "panda": 1, "cab": 1},
    }


def test_table10_inception_filter_preserves_centered_range() -> None:
    """The proposed detector should preserve Inception's [-0.5, 0.5] domain."""
    rng = table_10_module.np.random.RandomState(20170830)
    image = rng.uniform(-0.5, 0.5, size=(299, 299, 3)).astype(table_10_module.np.float32)

    filtered = article_final_detection_filter(image)

    assert filtered.shape == image.shape
    assert filtered.dtype == table_10_module.np.float32
    assert float(filtered.min()) >= -0.5 - 1e-6
    assert float(filtered.max()) <= 0.5 + 1e-6


def test_table10_googlenet_evaluator_requires_dataset_config(monkeypatch) -> None:
    """Implemented GoogLeNet rows should fail clearly without dataset inputs."""
    monkeypatch.setattr(
        table_10_module,
        "build_table_10_googlenet_model",
        lambda config: Table10DummyModel(),
    )

    with pytest.raises(ValueError, match="images_dir"):
        evaluate_table_10_googlenet_row(
            {"dataset": {"name": "imagenet"}, "model": {"name": "googlenet_caffe"}},
            {"attack": {"name": "deepfool"}},
        )


def test_table10_googlenet_class_folder_loader_uses_all_samples(tmp_path) -> None:
    """The full Table 10 config should be able to load all class-folder samples."""
    Image = pytest.importorskip("PIL.Image")
    images_dir = tmp_path / "images"
    first_dir = images_dir / "first"
    second_dir = images_dir / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)

    image = table_10_module.np.zeros((2, 2, 3), dtype=table_10_module.np.uint8)
    Image.fromarray(image, mode="RGB").save(str(first_dir / "a.JPEG"))
    Image.fromarray(image + 10, mode="RGB").save(str(first_dir / "b.JPEG"))
    Image.fromarray(image + 20, mode="RGB").save(str(second_dir / "c.JPEG"))

    images, labels = table_10_module._load_table_10_googlenet_images(
        {
            "dataset": {
                "name": "imagenet",
                "images_dir": str(images_dir),
                "image_size": 2,
                "n_samples": "all",
                "class_indices": {"first": 1, "second": 2},
            },
            "evaluation": {"n_samples": "all"},
        },
        Table10DummyModel(),
    )

    assert images.shape == (3, 2, 2, 3)
    assert labels.tolist() == [1, 1, 2]


def test_table10_imagenet_class_folder_loader_respects_quotas_and_order(tmp_path) -> None:
    """The Inception v3 reproduction subset should match zebra/panda/cab order."""
    images_dir = tmp_path / "images"
    class_specs = {
        "zebra": (80, 40),
        "panda": (169, 40),
        "cab": (267, 20),
    }
    for class_name, (_, count) in class_specs.items():
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True)
        for index in range(count + 2):
            (class_dir / "{0:03d}.JPEG".format(index)).write_bytes(b"placeholder")

    rows = table_10_module._class_folder_rows(
        images_dir,
        {"cab": 267, "panda": 169, "zebra": 80},
        class_order=["zebra", "panda", "cab"],
        class_quotas={"zebra": 40, "panda": 40, "cab": 20},
    )
    labels = [label for _, label in rows]

    assert len(rows) == 100
    assert labels[:40] == [80] * 40
    assert labels[40:80] == [169] * 40
    assert labels[80:] == [267] * 20


def test_table10_group_writes_computed_deepfool_metrics(monkeypatch, tmp_path) -> None:
    """The group runner should dispatch implemented DeepFool rows."""
    monkeypatch.setattr(
        table_10_module,
        "evaluate_table_10_googlenet_row",
        lambda group_config, row_config: {
            "metrics": {
                "num_failures": 1,
                "tp": 2,
                "fn": 3,
                "fp": 4,
                "rtp": 1,
                "rtp_percent": 50.0,
                "recall": 40.0,
                "precision": 33.3333333333,
                "f1": 36.3636363636,
            }
        },
    )

    rows = run_table_10_group(
        {
            "kind": "table_10_group",
            "output_dir": str(tmp_path),
            "dataset": {"name": "imagenet"},
            "model_group": "googlenet",
            "dataset_label": "ImageNet",
            "rows": [
                {
                    "no": 5,
                    "attack_model": "FGSM (\u03b5=1/255)/GoogLeNet",
                    "status": "implemented",
                    "attack": {"name": "fgsm"},
                },
                {
                    "no": 7,
                    "attack_model": "DeepFool/GoogLeNet",
                    "status": "implemented",
                    "attack": {"name": "deepfool"},
                },
            ],
        }
    )

    assert rows[0]["num_failures"] == 1
    assert rows[1]["num_failures"] == 1
    assert rows[1]["tp"] == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == ["metrics.csv", "metrics.json"]


def test_run_experiment_dispatches_table10_group(monkeypatch) -> None:
    """The official runner should dispatch Table 10 groups through their runner."""
    config = _consolidated_config()
    calls = []

    def fake_table10_runner(component_config):
        calls.append(
            (
                component_config["experiment_id"],
                component_config["kind"],
                component_config["model_group"],
            )
        )
        return {"status": "completed"}

    monkeypatch.setattr(
        experiment_runner,
        "run_table10_group_experiment",
        fake_table10_runner,
    )

    result = experiment_runner.run_experiment("table_10_m2", config)

    assert result == {"status": "completed"}
    assert calls == [("table_10_m2", "table_10_group", "m2")]


@pytest.mark.parametrize(
    "experiment_name",
    sorted(TABLE10_EXPERIMENTS)
    + sorted(IMAGENET_NEW_CLASS_EXPERIMENTS)
    + sorted(FASHION_MNIST_EXPERIMENTS),
)
def test_run_experiment_cli_accepts_table10_groups(monkeypatch, experiment_name) -> None:
    """scripts/run_experiment.py should accept every Table 10 model-group command."""
    calls = []

    def fake_run_experiment(name, config):
        calls.append((name, config["experiments"][name]["kind"]))
        return {}

    monkeypatch.setattr(run_experiment_script, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiment.py", "--experiment", experiment_name],
    )

    assert run_experiment_script.main() == 0
    assert calls == [(experiment_name, "table_10_group")]
