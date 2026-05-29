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
    evaluate_table_10_googlenet_row,
    run_table_10_group,
)
from deepdetector.evaluation.tables import table_10 as table_10_module  # noqa: E402
from deepdetector.experiments import runner as experiment_runner  # noqa: E402
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


def test_table10_imagenet_groups_use_test_dataset() -> None:
    """Every ImageNet Table 10 group should point at the test split."""
    config = _consolidated_config()
    for experiment_name in (
        "table_10_googlenet",
        "table_10_caffenet",
        "table_10_inception_v3",
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


def test_table10_blocked_reasons_stay_out_of_outputs(tmp_path) -> None:
    """Blocked reasons should not leak into official metrics outputs."""
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
    assert not (tmp_path / "manifest.json").exists()


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


@pytest.mark.parametrize("experiment_name", sorted(TABLE10_EXPERIMENTS))
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
