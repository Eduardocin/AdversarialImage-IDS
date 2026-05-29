from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.experiments import adversarial_examples  # noqa: E402


def test_prepare_mnist_fgsm_adversarial_set_filters_and_generates_once(monkeypatch) -> None:
    """The materializer should prepare FGSM and base predictions once."""
    images = np.asarray(
        [
            np.full((2, 2, 1), 0.1, dtype=np.float32),
            np.full((2, 2, 1), 0.2, dtype=np.float32),
            np.full((2, 2, 1), 0.3, dtype=np.float32),
        ]
    )
    labels = np.asarray([1, 2, 3], dtype=np.int64)
    calls = {"fgsm": 0, "predict": 0}

    monkeypatch.setattr(
        adversarial_examples,
        "load_mnist_test_slice",
        lambda start, end: (images, labels),
    )
    monkeypatch.setattr(
        adversarial_examples,
        "one_d_entropy",
        lambda image: 6.0
        if float(image[0, 0, 0]) < 0.15 or float(image[0, 0, 0]) > 0.25
        else 4.0,
    )
    monkeypatch.setattr(
        adversarial_examples,
        "create_restored_mnist_graph",
        lambda train_dir: {
            "sess": "sess",
            "model": "model",
            "x": "x",
            "predictions": "predictions",
            "train_dir": train_dir,
        },
    )

    def fake_generate_fgsm_examples(**kwargs):
        calls["fgsm"] += 1
        assert kwargs["eps"] == 0.3
        assert kwargs["clip_min"] == 0.0
        assert kwargs["clip_max"] == 1.0
        return kwargs["images"] + 0.1

    def fake_predict_labels(sess, x_placeholder, predictions, batch_images, batch_size=256):
        del sess, x_placeholder, predictions, batch_size
        calls["predict"] += 1
        return np.full((len(batch_images),), calls["predict"], dtype=np.int64)

    monkeypatch.setattr(
        adversarial_examples,
        "generate_fgsm_examples",
        fake_generate_fgsm_examples,
    )
    monkeypatch.setattr(adversarial_examples, "predict_labels", fake_predict_labels)

    context = adversarial_examples.prepare_mnist_fgsm_adversarial_set(
        {
            "dataset": {
                "start": 0,
                "end": 3,
                "name": "mnist",
                "high_entropy_only": True,
                "entropy_threshold": {"min": 5.0},
            },
            "model": {"checkpoint_dir": "artifacts/model"},
            "attack": {
                "epsilon": 0.3,
                "clip_min": 0.0,
                "clip_max": 1.0,
                "cache": False,
            },
            "evaluation": {"batch_size": 2},
        }
    )

    assert calls == {"fgsm": 1, "predict": 2}
    assert context.images.shape == (2, 2, 2, 1)
    assert context.labels.tolist() == [1, 3]
    np.testing.assert_allclose(context.adversarial_images, context.images + 0.1)
    assert context.clean_predictions.tolist() == [1, 1]
    assert context.adversarial_predictions.tolist() == [2, 2]
    assert context.metadata["num_loaded"] == 3
    assert context.metadata["num_after_entropy_filter"] == 2
    assert context.metadata["num_examples"] == 2


def test_prepare_mnist_fgsm_adversarial_set_reuses_cache(monkeypatch, tmp_path) -> None:
    """A second materialization with identical inputs should not regenerate FGSM."""
    images = np.asarray(
        [
            np.full((2, 2, 1), 0.1, dtype=np.float32),
            np.full((2, 2, 1), 0.2, dtype=np.float32),
        ]
    )
    labels = np.asarray([1, 2], dtype=np.int64)
    graph = {
        "sess": "sess",
        "model": "model",
        "x": "x",
        "predictions": "predictions",
    }
    calls = {"fgsm": 0, "predict": 0}

    monkeypatch.setattr(
        adversarial_examples,
        "load_mnist_test_slice",
        lambda start, end: (images, labels),
    )
    monkeypatch.setattr(adversarial_examples, "one_d_entropy", lambda image: 6.0)

    def fake_generate_fgsm_examples(**kwargs):
        calls["fgsm"] += 1
        return kwargs["images"] + 0.1

    def fake_predict_labels(sess, x_placeholder, predictions, batch_images, batch_size=256):
        del sess, x_placeholder, predictions, batch_size
        calls["predict"] += 1
        return np.full((len(batch_images),), calls["predict"], dtype=np.int64)

    monkeypatch.setattr(
        adversarial_examples,
        "generate_fgsm_examples",
        fake_generate_fgsm_examples,
    )
    monkeypatch.setattr(adversarial_examples, "predict_labels", fake_predict_labels)

    config = {
        "dataset": {
            "start": 0,
            "end": 2,
            "name": "mnist",
            "high_entropy_only": True,
            "entropy_threshold": {"min": 5.0},
        },
        "model": {"checkpoint_dir": "artifacts/model"},
        "attack": {
            "name": "fgsm",
            "epsilon": 0.3,
            "clip_min": 0.0,
            "clip_max": 1.0,
            "cache_dir": str(tmp_path / "adversarial_examples"),
        },
        "evaluation": {"batch_size": 2},
    }

    first = adversarial_examples.prepare_mnist_fgsm_adversarial_set(config, graph=graph)

    assert calls == {"fgsm": 1, "predict": 2}
    assert first.metadata["cache_status"] == "miss"
    assert Path(first.metadata["cache_path"]).is_file()
    assert Path(first.metadata["cache_path"]).with_suffix(".json").is_file()

    monkeypatch.setattr(
        adversarial_examples,
        "generate_fgsm_examples",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    monkeypatch.setattr(
        adversarial_examples,
        "predict_labels",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )

    second = adversarial_examples.prepare_mnist_fgsm_adversarial_set(config, graph=graph)

    assert second.metadata["cache_status"] == "hit"
    np.testing.assert_allclose(second.adversarial_images, images + 0.1)
    assert second.clean_predictions.tolist() == [1, 1]
    assert second.adversarial_predictions.tolist() == [2, 2]


def test_adversarial_cache_artifacts_are_ignored() -> None:
    """Generated adversarial cache files should remain out of git."""
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/artifacts/adversarial_examples/" in gitignore


def test_legacy_fgsm_context_module_was_removed() -> None:
    """Adversarial generation should live in the central materializer."""
    assert not (
        PROJECT_ROOT / "src" / "deepdetector" / "experiments" / "fgsm_context.py"
    ).exists()
