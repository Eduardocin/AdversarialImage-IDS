from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.attacks import cw_l2, cw_linf  # noqa: E402


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
