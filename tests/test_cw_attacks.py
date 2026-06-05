from pathlib import Path
import sys
import types

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.attacks import cw_l2, cw_l2_nn_robust, cw_linf  # noqa: E402


class FakeGraph:
    def __init__(self) -> None:
        self.active = False

    def as_default(self):
        return self

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.active = False


class FakeSession:
    def __init__(self) -> None:
        self.graph = FakeGraph()


class FakeModel:
    def __init__(self) -> None:
        self.sess = FakeSession()
        self.num_labels = 1008


def test_cw_l2_attack_uses_model_session_graph(monkeypatch) -> None:
    """CW L2 ops must be created in the same graph used by the model session."""
    model = FakeModel()

    def fake_generate(**kwargs):
        assert model.sess.graph.active is True
        assert kwargs["sess"] is model.sess
        assert kwargs["nb_classes"] == 1008
        return kwargs["images"]

    monkeypatch.setattr(cw_l2, "generate_cw_l2_examples", fake_generate)

    images = np.zeros((1, 299, 299, 3), dtype=np.float32)
    result = cw_l2.generate_cw_l2_attack(model=model, images=images, labels=np.asarray([1]))

    np.testing.assert_array_equal(result, images)


def test_cw_linf_attack_uses_model_session_graph(monkeypatch) -> None:
    """CW Linf ops must be created in the same graph used by the model session."""
    model = FakeModel()

    def fake_generate(**kwargs):
        assert model.sess.graph.active is True
        assert kwargs["sess"] is model.sess
        assert kwargs["nb_classes"] == 1008
        return kwargs["images"]

    monkeypatch.setattr(cw_linf, "generate_cw_linf_examples", fake_generate)

    images = np.zeros((1, 299, 299, 3), dtype=np.float32)
    result = cw_linf.generate_cw_linf_attack(model=model, images=images, labels=np.asarray([1]))

    np.testing.assert_array_equal(result, images)


def test_nn_robust_patch_adds_legacy_keras_image_dim_ordering(monkeypatch) -> None:
    """TF2/Keras backends should expose the Keras 1 image ordering helper."""
    keras_module = types.ModuleType("keras")
    backend_module = types.ModuleType("keras.backend")
    calls = []

    def set_image_data_format(value):
        calls.append(value)

    backend_module.set_image_data_format = set_image_data_format
    backend_module.image_data_format = lambda: "channels_last"
    keras_module.backend = backend_module
    monkeypatch.setitem(sys.modules, "keras", keras_module)
    monkeypatch.setitem(sys.modules, "keras.backend", backend_module)

    cw_l2_nn_robust._patch_keras_backend_symbols()

    backend_module.set_image_dim_ordering("tf")
    assert calls == ["channels_last"]
    assert backend_module.image_dim_ordering() == "tf"


def test_nn_robust_patch_adds_modern_keras_image_data_format(monkeypatch) -> None:
    """Keras 1 backends should expose the modern data-format helper."""
    keras_module = types.ModuleType("keras")
    backend_module = types.ModuleType("keras.backend")
    calls = []

    def set_image_dim_ordering(value):
        calls.append(value)

    backend_module.set_image_dim_ordering = set_image_dim_ordering
    backend_module.image_dim_ordering = lambda: "tf"
    keras_module.backend = backend_module
    monkeypatch.setitem(sys.modules, "keras", keras_module)
    monkeypatch.setitem(sys.modules, "keras.backend", backend_module)

    cw_l2_nn_robust._patch_keras_backend_symbols()

    backend_module.set_image_data_format("channels_last")
    assert calls == ["tf"]
    assert backend_module.image_data_format() == "channels_last"


def test_nn_robust_patch_adds_tensorflow_v1_symbols(monkeypatch) -> None:
    """TF2 modules should expose the TF1 symbols used by nn_robust_attacks."""
    tensorflow_module = types.ModuleType("tensorflow")
    tensorflow_module.train = types.SimpleNamespace()
    tensorflow_module.compat = types.SimpleNamespace(
        v1=types.SimpleNamespace(
            disable_eager_execution=lambda: None,
            placeholder=object(),
            Session=object(),
            ConfigProto=object(),
            global_variables=object(),
            trainable_variables=object(),
            variables_initializer=object(),
            global_variables_initializer=object(),
            reset_default_graph=object(),
            get_default_graph=object(),
            assign=object(),
            gradients=object(),
            train=types.SimpleNamespace(AdamOptimizer=object()),
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow_module)

    cw_l2_nn_robust._patch_tensorflow_v1_symbols()

    assert tensorflow_module.placeholder is tensorflow_module.compat.v1.placeholder
    assert tensorflow_module.Session is tensorflow_module.compat.v1.Session
    assert (
        tensorflow_module.train.AdamOptimizer
        is tensorflow_module.compat.v1.train.AdamOptimizer
    )
