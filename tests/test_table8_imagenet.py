import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.evaluation import table8 as table8_module
from scripts.article_reproduction import table_8_imagenet as table8_script


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

    assert list(table8_script.configured_filters(config)) == [
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
        list(table8_script.configured_filters(config))


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
        for mask_type, size in table8_script.TABLE8_FILTERS
    ]

    path = table8_script.write_pivot_csv(tmp_path / "table_8_imagenet.csv", rows)

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
    config_text = table8_script.DEFAULT_CONFIG.read_text(encoding="utf-8")

    assert "split: validation" in config_text
    assert "name: zebra" not in config_text
    assert "data/imagenet/imagenet/test/zebra" not in config_text
    assert "data/imagenet/validation/jellyfish" in config_text
    assert (
        "artifacts/adversarial_examples/imagenet/fgsm/validation/adversarial_examples.npy"
        in config_text
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
        table8_script._validate_dataset_split(config)


def test_table8_main_writes_partial_status_when_no_validation_images(
    monkeypatch,
    tmp_path,
) -> None:
    """Missing validation data should produce partial status instead of metrics."""
    config_path = tmp_path / "table8.yaml"
    config_path.write_text(
        "\n".join(
            [
                "outputs:",
                "  results_dir: results/imagenet/article_reproduction",
                "  pivot_csv: table_8_imagenet.csv",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["table_8_imagenet", "--config", str(config_path), "--output-dir", str(tmp_path)])
    monkeypatch.setattr(table8_script, "load_config", lambda path: {})
    monkeypatch.setattr(table8_script, "build_model", lambda config: object())
    monkeypatch.setattr(
        table8_script,
        "load_subset_images",
        lambda config: (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        ),
    )

    assert table8_script.main() == 0

    status = json.loads((tmp_path / "table_8_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "parcial"
    assert status["limitation"] == "nenhuma_imagem_carregada"
    assert status["attack_success"] == 0
    assert not (tmp_path / "table_8_imagenet.csv").exists()


def test_table8_main_does_not_complete_when_no_attack_succeeds(
    monkeypatch,
    tmp_path,
) -> None:
    """Zero successful adversarial examples should not produce completed metrics."""
    config_path = tmp_path / "table8.yaml"
    config_path.write_text(
        "\n".join(
            [
                "outputs:",
                "  results_dir: results/imagenet/article_reproduction",
                "  pivot_csv: table_8_imagenet.csv",
            ]
        ),
        encoding="utf-8",
    )
    clean_images = np.asarray([_marker_image(10), _marker_image(11)], dtype=np.float32)
    labels = np.asarray([1, 1], dtype=np.int32)
    adversarial_images = np.asarray([_marker_image(10), _marker_image(11)], dtype=np.float32)

    monkeypatch.setattr(sys, "argv", ["table_8_imagenet", "--config", str(config_path), "--output-dir", str(tmp_path)])
    monkeypatch.setattr(table8_script, "load_config", lambda path: {})
    monkeypatch.setattr(table8_script, "build_model", lambda config: MarkerModel())
    monkeypatch.setattr(table8_script, "load_subset_images", lambda config: (clean_images, labels))
    monkeypatch.setattr(table8_script, "_article_model_inputs", lambda model, images: images)
    monkeypatch.setattr(
        table8_script,
        "adversarial_images_for_run",
        lambda config, model, images, override_path, selected_indices=None: adversarial_images,
    )

    assert table8_script.main() == 0

    status = json.loads((tmp_path / "table_8_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "parcial"
    assert status["limitation"] == "nenhum_adversarial_bem_sucedido"
    assert status["attack_success"] == 0
    assert status["disturbed_failure"] == 2
    assert not (tmp_path / "table_8_imagenet.csv").exists()
