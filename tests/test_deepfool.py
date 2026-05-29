from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.attacks.deepfool import generate_deepfool  # noqa: E402
from deepdetector.attacks.registry import ATTACK_REGISTRY, generate_attack  # noqa: E402


class LinearScoreModel:
    """Small differentiable model exposing the DeepFool wrapper contract."""

    def scores(self, images: np.ndarray) -> np.ndarray:
        batch = np.asarray(images, dtype=np.float32)
        flat_sum = np.sum(batch.reshape((len(batch), -1)), axis=1)
        return np.stack([flat_sum, 4.0 - flat_sum], axis=1).astype(np.float32)

    def gradient(self, image: np.ndarray, class_id: int) -> np.ndarray:
        if int(class_id) == 0:
            return -np.ones_like(image, dtype=np.float32)
        return np.ones_like(image, dtype=np.float32)


class NonDifferentiableModel:
    def scores(self, images: np.ndarray) -> np.ndarray:
        batch = np.asarray(images, dtype=np.float32)
        return np.zeros((len(batch), 2), dtype=np.float32)


class DualNetworkDeepFoolModel:
    """Expose prediction scores and attack gradients from separate logical nets."""

    def __init__(self):
        self.gradient_sources = []

    def predict_preprocessed_batch(self, images: np.ndarray) -> np.ndarray:
        batch = np.asarray(images, dtype=np.float32)
        flat_sum = np.sum(batch.reshape((len(batch), -1)), axis=1)
        return np.stack([flat_sum, 4.0 - flat_sum], axis=1).astype(np.float32)

    def gradient(self, image: np.ndarray, class_id: int) -> np.ndarray:
        self.gradient_sources.append(("deploy_removeSoftmax", int(class_id)))
        if int(class_id) == 0:
            return -np.ones_like(image, dtype=np.float32)
        return np.ones_like(image, dtype=np.float32)


def test_deepfool_attack_is_registered() -> None:
    assert "deepfool" in ATTACK_REGISTRY
    assert ATTACK_REGISTRY["deepfool"] is generate_deepfool


def test_deepfool_is_available_through_attack_dispatcher() -> None:
    images = np.full((1, 1, 2, 2), 0.8, dtype=np.float32)

    x_adv = generate_attack(
        "deepfool",
        model=LinearScoreModel(),
        images=images,
        max_iter=3,
        clip_min=0.0,
        clip_max=1.0,
    )

    assert x_adv.shape == images.shape


def test_deepfool_returns_same_shape_as_input() -> None:
    images = np.full((2, 1, 2, 2), 0.8, dtype=np.float32)
    labels = np.asarray([0, 0], dtype=np.int64)

    x_adv = generate_deepfool(
        model=LinearScoreModel(),
        images=images,
        labels=labels,
        max_iter=3,
        overshoot=0.02,
        clip_min=0.0,
        clip_max=1.0,
    )

    assert x_adv.shape == images.shape
    assert np.argmax(LinearScoreModel().scores(images), axis=1).tolist() == [0, 0]
    assert np.argmax(LinearScoreModel().scores(x_adv), axis=1).tolist() == [1, 1]


def test_deepfool_respects_clip_bounds() -> None:
    images = np.full((1, 1, 2, 2), 0.8, dtype=np.float32)

    x_adv = generate_deepfool(
        model=LinearScoreModel(),
        images=images,
        max_iter=3,
        overshoot=0.02,
        clip_min=0.0,
        clip_max=1.0,
    )

    assert float(x_adv.min()) >= 0.0
    assert float(x_adv.max()) <= 1.0


def test_deepfool_uses_wrapper_gradient_contract_for_attack_network() -> None:
    """DeepFool should rely on model.gradient without loading model files itself."""
    model = DualNetworkDeepFoolModel()
    images = np.full((1, 1, 2, 2), 0.8, dtype=np.float32)

    x_adv = generate_deepfool(
        model=model,
        images=images,
        max_iter=3,
        clip_min=0.0,
        clip_max=1.0,
    )

    assert x_adv.shape == images.shape
    assert model.gradient_sources
    assert {source for source, _class_id in model.gradient_sources} == {
        "deploy_removeSoftmax"
    }


def test_deepfool_requires_differentiable_model() -> None:
    images = np.full((1, 1, 2, 2), 0.8, dtype=np.float32)

    with pytest.raises(NotImplementedError, match="requires gradient"):
        generate_deepfool(model=NonDifferentiableModel(), images=images)


def test_table_10_googlenet_deepfool_row_has_config() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    rows = config["experiments"]["table_10_googlenet"]["rows"]
    row = next(item for item in rows if item["no"] == 7)

    assert row["status"] != "blocked"
    assert row["attack"]["name"] == "deepfool"
    assert row["attack"]["max_iter"] == 50
    assert row["attack"]["overshoot"] == 0.02
    assert row["attack"]["clip_min"] == 0.0
    assert row["attack"]["clip_max"] == 1.0
    assert (
        config["experiments"]["table_10_googlenet"]["model"]["deploy_proto"]
        == "artifacts/models/imagenet/googlenet/deploy_original.prototxt"
    )
    assert (
        config["experiments"]["table_10_googlenet"]["model"]["attack_deploy_proto"]
        == "artifacts/models/imagenet/googlenet/deploy_removeSoftmax.prototxt"
    )
