from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.experiments import fgsm_context  # noqa: E402


def test_prepare_fgsm_context_filters_high_entropy_and_generates_once(monkeypatch) -> None:
    """The reusable context should prepare FGSM and base predictions once."""
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
        fgsm_context,
        "load_mnist_test_slice",
        lambda start, end: (images, labels),
    )
    monkeypatch.setattr(
        fgsm_context,
        "one_d_entropy",
        lambda image: 6.0 if float(image[0, 0, 0]) < 0.15 or float(image[0, 0, 0]) > 0.25 else 4.0,
    )
    monkeypatch.setattr(
        fgsm_context,
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

    monkeypatch.setattr(fgsm_context, "generate_fgsm_examples", fake_generate_fgsm_examples)
    monkeypatch.setattr(fgsm_context, "predict_labels", fake_predict_labels)

    context = fgsm_context.prepare_fgsm_context(
        {
            "dataset": {
                "start": 0,
                "end": 3,
                "high_entropy_only": True,
                "entropy_threshold": {"min": 5.0},
            },
            "model": {"checkpoint_dir": "artifacts/model"},
            "attack": {"epsilon": 0.3, "clip_min": 0.0, "clip_max": 1.0},
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
