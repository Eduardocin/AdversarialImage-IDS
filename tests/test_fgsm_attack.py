from pathlib import Path
import sys
import types

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.attacks import fgsm as fgsm_module  # noqa: E402
from deepdetector.attacks.fgsm import _fgsm_logits_from_model  # noqa: E402


def test_fgsm_logits_fallback_handles_legacy_keras_tuple_error(monkeypatch) -> None:
    """FGSM should fall back to model probabilities when CleverHans fprop fails."""

    class FakeWrapper:
        def __init__(self, model) -> None:
            self.model = model

        def get_logits(self, x_placeholder):
            raise AttributeError("'tuple' object has no attribute 'layer'")

    class FakeModel:
        def __call__(self, x_placeholder):
            assert x_placeholder == "x"
            return np.asarray([[0.25, 0.75]], dtype=np.float32)

    cleverhans_module = types.ModuleType("cleverhans")
    utils_keras_module = types.ModuleType("cleverhans.utils_keras")
    utils_keras_module.KerasModelWrapper = FakeWrapper
    tensorflow_module = types.ModuleType("tensorflow")
    tensorflow_module.clip_by_value = np.clip
    tensorflow_module.math = types.SimpleNamespace(log=np.log)

    monkeypatch.setitem(sys.modules, "cleverhans", cleverhans_module)
    monkeypatch.setitem(sys.modules, "cleverhans.utils_keras", utils_keras_module)
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow_module)

    logits = _fgsm_logits_from_model(FakeModel(), "x")

    np.testing.assert_allclose(logits, np.log([[0.25, 0.75]]), rtol=1e-6)


def test_fgsm_logits_uses_provided_probability_output_without_wrapper(monkeypatch) -> None:
    """M3 Table 10 can pass graph predictions directly without rebuilding Keras."""
    tensorflow_module = types.ModuleType("tensorflow")
    tensorflow_module.clip_by_value = np.clip
    tensorflow_module.math = types.SimpleNamespace(log=np.log)
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow_module)

    probabilities = np.asarray([[0.4, 0.6]], dtype=np.float32)

    logits = _fgsm_logits_from_model(
        object(),
        "x",
        model_output=probabilities,
        output_is_probabilities=True,
    )

    np.testing.assert_allclose(logits, np.log([[0.4, 0.6]]), rtol=1e-6)


def test_generate_fgsm_feeds_keras_inference_phase(monkeypatch) -> None:
    """FGSM generation should run dropout models in inference mode."""
    captured = {}

    class FakePlaceholder:
        def __add__(self, other):
            return other

    class FakeSession:
        def run(self, tensor, feed_dict):
            captured["tensor"] = tensor
            captured["feed_dict"] = dict(feed_dict)
            return np.asarray([[[[0.2]]]], dtype=np.float32)

    tensorflow_module = types.ModuleType("tensorflow")
    tensorflow_module.float32 = np.float32
    tensorflow_module.clip_by_value = np.clip
    tensorflow_module.sign = np.sign
    tensorflow_module.reduce_max = lambda values, axis, keepdims: np.max(
        values, axis=axis, keepdims=keepdims
    )
    tensorflow_module.equal = np.equal
    tensorflow_module.cast = lambda values, dtype: values.astype(dtype)
    tensorflow_module.stop_gradient = lambda values: values
    tensorflow_module.compat = types.SimpleNamespace(
        v1=types.SimpleNamespace(gradients=lambda loss, x: [np.asarray([[[[1.0]]]])])
    )
    tensorflow_module.nn = types.SimpleNamespace(
        softmax_cross_entropy_with_logits_v2=lambda labels, logits: logits
    )
    tensorflow_module.math = types.SimpleNamespace(log=np.log)
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow_module)

    monkeypatch.setattr(
        fgsm_module,
        "_fgsm_logits_from_model",
        lambda *args, **kwargs: np.asarray([[0.1, 0.9]], dtype=np.float32),
    )
    monkeypatch.setattr(
        fgsm_module,
        "_keras_learning_phase_feed",
        lambda value=0: {"keras_learning_phase": value},
    )

    result = fgsm_module.generate_fgsm_examples(
        sess=FakeSession(),
        model=object(),
        x_placeholder=FakePlaceholder(),
        images=np.asarray([[[[0.1]]]], dtype=np.float32),
    )

    assert captured["feed_dict"]["keras_learning_phase"] == 0
    np.testing.assert_allclose(result, np.asarray([[[[0.2]]]], dtype=np.float32))
