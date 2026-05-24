from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from scripts.article_reproduction.inspect_imagenet_clock_predictions import (
    inspect_predictions,
    write_prediction_csv,
)
from deepdetector.evaluation.table4_imagenet import Table4Sample


class ConstantModel:
    """Predict a constant ImageNet label."""

    def __init__(self, label: int) -> None:
        self.label = int(label)

    def predict_label(self, images: np.ndarray) -> np.ndarray:
        return np.full((len(images),), self.label, dtype=np.int32)


def test_inspect_predictions_records_clean_labels() -> None:
    """Prediction inspection should not generate attacks or alter labels."""
    samples = [
        Table4Sample(
            image=np.zeros((2, 2, 3), dtype=np.float32),
            true_label=530,
            image_id="n03196217_1",
            class_name="digital_clock",
        )
    ]

    rows = inspect_predictions(model=ConstantModel(409), samples=samples)

    assert rows == [
        {
            "image_id": "n03196217_1",
            "class_name": "digital_clock",
            "configured_label": 530,
            "clean_pred": 409,
            "matches_configured_label": False,
        }
    ]


def test_write_prediction_csv_uses_stable_columns(tmp_path) -> None:
    """Prediction CSV should be simple to inspect manually."""
    output = write_prediction_csv(
        tmp_path / "digital_clock_predictions.csv",
        [
            {
                "image_id": "sample",
                "class_name": "digital_clock",
                "configured_label": 530,
                "clean_pred": 530,
                "matches_configured_label": True,
            }
        ],
    )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "image_id,class_name,configured_label,clean_pred,matches_configured_label",
        "sample,digital_clock,530,530,True",
    ]


def test_inspect_script_imports_when_executed_directly() -> None:
    """The script should resolve local imports when called by path."""
    result = subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "scripts"
                / "article_reproduction"
                / "inspect_imagenet_clock_predictions.py"
            ),
            "--help",
        ],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--class-name" in result.stdout
