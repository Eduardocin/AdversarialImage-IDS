from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.attacks import nn_robust  # noqa: E402


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
        self.logit_inputs = []

    def get_logits(self, data):
        self.logit_inputs.append(data)
        return data


def test_nn_robust_cw_l2_attack_loads_local_backend_and_one_hot_labels(tmp_path) -> None:
    """The nn_robust L2 adapter should use local CarliniL2 and 1008-way labels."""
    attack_root = tmp_path / "nn_robust_attacks"
    attack_root.mkdir()
    (attack_root / "l2_attack.py").write_text(
        "\n".join(
            [
                "class CarliniL2:",
                "    def __init__(self, sess, model, batch_size=1, confidence=0, targeted=True, learning_rate=0.01, binary_search_steps=9, max_iterations=1000, abort_early=True, initial_const=0.001, boxmin=-0.5, boxmax=0.5):",
                "        assert model.image_size == 299",
                "        assert model.num_channels == 3",
                "        assert model.num_labels == 1008",
                "        assert batch_size == 1",
                "        assert confidence == 0.5",
                "        assert targeted is False",
                "        assert boxmin == -0.5",
                "        assert boxmax == 0.5",
                "        self.model = model",
                "    def attack(self, imgs, targets):",
                "        assert targets.shape == (1, 1008)",
                "        assert targets[0, 267] == 1.0",
                "        self.model.predict('attack_tensor')",
                "        return imgs + 0.25",
            ]
        ),
        encoding="utf-8",
    )
    model = FakeModel()
    images = np.zeros((1, 299, 299, 3), dtype=np.float32)

    result = nn_robust.generate_nn_robust_cw_l2_attack(
        model=model,
        images=images,
        labels=np.asarray([267]),
        nn_robust_attacks_root=str(attack_root),
        kappa=0.5,
        targeted=False,
    )

    assert model.logit_inputs == ["attack_tensor"]
    np.testing.assert_array_equal(result, images + 0.25)


def test_nn_robust_cw_linf_attack_loads_local_backend(tmp_path) -> None:
    """The nn_robust Linf adapter should use local CarliniLi."""
    attack_root = tmp_path / "nn_robust_attacks"
    attack_root.mkdir()
    (attack_root / "li_attack.py").write_text(
        "\n".join(
            [
                "class CarliniLi:",
                "    def __init__(self, sess, model, targeted=True, learning_rate=0.005, max_iterations=1000, abort_early=True, initial_const=1e-5, largest_const=20.0, reduce_const=False, decrease_factor=0.9, const_factor=2.0):",
                "        assert model.image_size == 299",
                "        assert model.num_channels == 3",
                "        assert model.num_labels == 1008",
                "        assert targeted is False",
                "        assert learning_rate == 0.005",
                "        self.model = model",
                "    def attack(self, imgs, targets):",
                "        assert targets.shape == (1, 1008)",
                "        assert targets[0, 80] == 1.0",
                "        self.model.predict('linf_tensor')",
                "        return imgs + 0.125",
            ]
        ),
        encoding="utf-8",
    )
    model = FakeModel()
    images = np.zeros((1, 299, 299, 3), dtype=np.float32)

    result = nn_robust.generate_nn_robust_cw_linf_attack(
        model=model,
        images=images,
        labels=np.asarray([80]),
        nn_robust_attacks_root=str(attack_root),
        targeted=False,
    )

    assert model.logit_inputs == ["linf_tensor"]
    np.testing.assert_array_equal(result, images + 0.125)
