import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.evaluation import table8 as table8_module
from deepdetector.evaluation.table8 import Table8FilterResult
from deepdetector.experiments import table8_imagenet_runner


class MarkerModel:
    """Predict from a marker value stored in the first Caffe tensor element."""

    def predict_preprocessed_label(self, images: np.ndarray) -> np.ndarray:
        marker_to_label = {10: 1, 11: 1, 20: 2, 21: 2, 30: 1}
        return np.asarray(
            [
                marker_to_label.get(
                    int(round(float(image[0, 0, 0]))),
                    int(round(float(image[0, 0, 0]))),
                )
                for image in images
            ],
            dtype=np.int32,
        )


def _marker_image(marker: int) -> np.ndarray:
    image = np.zeros((3, 3, 3), dtype=np.float32)
    image[0, 0, 0] = float(marker)
    return image


def test_table8_configured_filters_are_the_five_fixed_article_filters() -> None:
    """Table 8 must use the five superior filters selected from Table 7."""
    config = {
        "filter": {
            "filters": [
                {"mask_type": "cross", "size": 5},
                {"mask_type": "cross", "size": 7},
                {"mask_type": "diamond", "size": 5},
                {"mask_type": "diamond", "size": 7},
                {"mask_type": "box", "size": 5},
            ]
        }
    }

    assert list(table8_imagenet_runner.configured_filters(config)) == [
        ("cross", 5),
        ("cross", 7),
        ("diamond", 5),
        ("diamond", 7),
        ("box", 5),
    ]


def test_table8_rejects_non_fixed_filter_sets() -> None:
    """Local config must not re-select Table 8 filters dynamically."""
    config = {"filter": {"filters": [{"mask_type": "box", "size": 3}]}}

    with pytest.raises(ValueError, match="Table 8 must evaluate exactly"):
        list(table8_imagenet_runner.configured_filters(config))


def test_table8_write_pivot_csv_uses_fixed_columns_and_metric_rows(tmp_path) -> None:
    """The Table 8 CSV should be the article-style pivot required by the spec."""
    rows = [
        {
            "mask_type": mask_type,
            "size": size,
            "recall": 1.0,
            "precision": 0.5,
            "f1": 2.0 / 3.0,
        }
        for mask_type, size in table8_imagenet_runner.TABLE8_FILTERS
    ]

    path = table8_imagenet_runner.write_pivot_csv(
        tmp_path / "table_8_imagenet.csv",
        rows,
        columns=table8_imagenet_runner.TABLE8_COLUMNS,
    )

    assert path.name == "table_8_imagenet.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "metric,cross_5x5,cross_7x7,diamond_5x5,diamond_7x7,box_5x5"
    )
    assert [line.split(",")[0] for line in lines[1:]] == [
        "Recall",
        "Precision",
        "F1 Score",
    ]


def test_table8_evaluation_filters_clean_baseline_and_disturbed_failures(monkeypatch) -> None:
    """Wrong clean predictions and disturbed failures must not enter TP/FN/FP."""
    dataset = (
        np.asarray([_marker_image(10), _marker_image(11), _marker_image(30)], dtype=np.float32),
        np.asarray([1, 1, 5], dtype=np.int32),
        np.asarray([_marker_image(20), _marker_image(11), _marker_image(21)], dtype=np.float32),
    )

    def fake_filter(
        image: np.ndarray,
        mask_type: str,
        size: int,
    ) -> np.ndarray:
        del mask_type, size
        marker = int(round(float(image[0, 0, 0])))
        marker_after_filter = {10: 9, 20: 3}.get(marker, marker)
        filtered = np.array(image, copy=True)
        filtered[0, 0, 0] = float(marker_after_filter)
        return filtered

    monkeypatch.setattr(table8_module, "_apply_table8_filter_to_model_input", fake_filter)

    result = table8_module.evaluate_table8_filter(
        model=MarkerModel(),
        dataset=dataset,
        mask_type="cross",
        size=5,
    )

    assert result.attack_success == 1
    assert result.disturbed_failure == 1
    assert result.skipped_wrong_baseline == 1
    assert result.tp == 1
    assert result.fn == 0
    assert result.fp == 1
    assert result.precision == 0.5


def test_table8_default_config_uses_validation_split() -> None:
    """The checked-in Table 8 config must point at validation, not training."""
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    table8 = config["experiments"]["table_8"]

    assert table8["kind"] == "imagenet_table_8"
    assert table8["dataset"]["split"] == "validation"
    assert table8["dataset"]["classes"] == [
        {"name": "jellyfish", "label": 107, "path": "data/imagenet/validation/jellyfish"}
    ]
    assert "test" not in table8["dataset"]["classes"][0]["path"]
    assert (
        table8["attack"]["adversarial_path"]
        == "artifacts/adversarial_examples/imagenet/fgsm/validation/adversarial_examples.npy"
    )


def test_table8_rejects_validation_config_pointing_to_test_split() -> None:
    """Validation runs must not accidentally consume ImageNet test classes."""
    config = {
        "dataset": {
            "split": "validation",
            "classes": [
                {
                    "name": "zebra",
                    "label": 340,
                    "path": "data/imagenet/imagenet/test/zebra",
                }
            ],
        }
    }

    with pytest.raises(ValueError, match="validation split cannot use test path"):
        table8_imagenet_runner.load_imagenet_table7_subset(config)


def test_table8_runner_writes_partial_status_when_no_validation_images(
    monkeypatch,
    tmp_path,
) -> None:
    """Missing validation data should produce partial status instead of metrics."""
    monkeypatch.setattr(table8_imagenet_runner, "build_imagenet_table7_model", lambda config: object())
    monkeypatch.setattr(
        table8_imagenet_runner,
        "load_imagenet_table7_subset",
        lambda config: (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        ),
    )

    result = table8_imagenet_runner.run_table8_imagenet_experiment(
        {
            "output": {"dir": str(tmp_path), "status_json": "table_8_status.json"},
            "dataset": {"name": "imagenet", "split": "validation"},
            "model": {},
            "attack": {},
            "filter": {},
        }
    )

    assert result["status"] == "parcial"
    status = json.loads((tmp_path / "table_8_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "parcial"
    assert status["limitation"] == "nenhuma_imagem_carregada"
    assert status["attack_success"] == 0
    assert not (tmp_path / "table_8_imagenet.csv").exists()


def test_table8_runner_does_not_complete_when_no_attack_succeeds(
    monkeypatch,
    tmp_path,
) -> None:
    """Zero successful adversarial examples should not produce completed metrics."""
    clean_images = np.asarray([_marker_image(10), _marker_image(11)], dtype=np.float32)
    labels = np.asarray([1, 1], dtype=np.int32)
    adversarial_images = np.asarray([_marker_image(10), _marker_image(11)], dtype=np.float32)

    monkeypatch.setattr(table8_imagenet_runner, "build_imagenet_table7_model", lambda config: MarkerModel())
    monkeypatch.setattr(
        table8_imagenet_runner,
        "load_imagenet_table7_subset",
        lambda config: (clean_images, labels),
    )
    monkeypatch.setattr(table8_imagenet_runner, "_article_model_inputs", lambda model, images: images)
    monkeypatch.setattr(
        table8_imagenet_runner,
        "adversarial_images_for_run",
        lambda config, model, images, selected_indices=None: adversarial_images,
    )

    result = table8_imagenet_runner.run_table8_imagenet_experiment(
        {
            "output": {"dir": str(tmp_path), "status_json": "table_8_status.json"},
            "dataset": {"name": "imagenet", "split": "validation"},
            "model": {},
            "attack": {},
            "filter": {},
        }
    )

    assert result["status"] == "parcial"
    status = json.loads((tmp_path / "table_8_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "parcial"
    assert status["limitation"] == "nenhum_adversarial_bem_sucedido"
    assert status["attack_success"] == 0
    assert status["disturbed_failure"] == 2
    assert not (tmp_path / "table_8_imagenet.csv").exists()


def test_run_table8_experiment_writes_pivot_and_status(monkeypatch, tmp_path) -> None:
    """The official Table 8 experiment should reuse the ImageNet pivot flow."""
    calls = []
    clean_images = np.asarray([_marker_image(10)], dtype=np.float32)
    labels = np.asarray([1], dtype=np.int32)
    adversarial_images = np.asarray([_marker_image(20)], dtype=np.float32)
    (tmp_path / "metrics.csv").write_text("stale", encoding="utf-8")
    (tmp_path / "metrics.json").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(table8_imagenet_runner, "build_imagenet_table7_model", lambda config: MarkerModel())
    monkeypatch.setattr(
        table8_imagenet_runner,
        "load_imagenet_table7_subset",
        lambda config: (clean_images, labels),
    )
    monkeypatch.setattr(table8_imagenet_runner, "_article_model_inputs", lambda model, images: images)
    monkeypatch.setattr(
        table8_imagenet_runner,
        "adversarial_images_for_run",
        lambda config, model, images, selected_indices=None: adversarial_images,
    )

    def fake_evaluate(**kwargs):
        calls.append((kwargs["mask_type"], kwargs["size"]))
        return Table8FilterResult(
            mask_type=kwargs["mask_type"],
            size=kwargs["size"],
            tp=1,
            fn=1,
            fp=0,
            recall=0.5,
            precision=1.0,
            f1=2.0 / 3.0,
            attack_success=1,
            disturbed_failure=0,
            skipped_wrong_baseline=0,
        )

    monkeypatch.setattr(table8_imagenet_runner, "evaluate_table8_filter", fake_evaluate)

    result = table8_imagenet_runner.run_table8_imagenet_experiment(
        {
            "output": {
                "dir": str(tmp_path),
                "pivot_csv": "table_8_imagenet.csv",
                "status_json": "table_8_status.json",
            },
            "dataset": {"name": "imagenet", "split": "validation"},
            "model": {},
            "attack": {},
            "filter": {
                "filters": [
                    {"mask_type": "cross", "size": 5},
                    {"mask_type": "cross", "size": 7},
                    {"mask_type": "diamond", "size": 5},
                    {"mask_type": "diamond", "size": 7},
                    {"mask_type": "box", "size": 5},
                ]
            },
        }
    )

    assert result["status"] == "completo"
    assert calls == [
        ("cross", 5),
        ("cross", 7),
        ("diamond", 5),
        ("diamond", 7),
        ("box", 5),
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "table_8_imagenet.csv",
        "table_8_status.json",
    ]
    assert (tmp_path / "table_8_imagenet.csv").read_text(encoding="utf-8").splitlines() == [
        "metric,cross_5x5,cross_7x7,diamond_5x5,diamond_7x7,box_5x5",
        "Recall,0.500000,0.500000,0.500000,0.500000,0.500000",
        "Precision,1.000000,1.000000,1.000000,1.000000,1.000000",
        "F1 Score,0.666667,0.666667,0.666667,0.666667,0.666667",
    ]
