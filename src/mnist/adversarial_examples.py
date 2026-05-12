"""Generate and persist MNIST adversarial examples."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Dict, Optional

import numpy as np

from .attacks import generate_cw_l2, generate_cw_linf, generate_fgsm
from .config import AttackConfig, MnistExperimentConfig
from .data import ensure_carlini_mnist_data, generate_untargeted_data, load_mnist_from_idx
from .models import build_m2_inference_model


@dataclass(frozen=True)
class AdversarialGenerationResult:
    """Summary of a generated adversarial example artifact."""

    attack_name: str
    output_npz: Path
    metadata_json: Path
    samples: int
    success_count: int


def _artifact_dir(config: MnistExperimentConfig, attack_name: str) -> Path:
    return config.paths.outputs_root / "adversarial" / attack_name


def _save_artifact(
    output_dir: Path,
    attack_name: str,
    clean_images: np.ndarray,
    adversarial_images: np.ndarray,
    labels: np.ndarray,
    clean_predictions: np.ndarray,
    adversarial_predictions: np.ndarray,
    source_indices: np.ndarray,
    metadata: Dict[str, object],
) -> AdversarialGenerationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / "examples.npz"
    metadata_json = output_dir / "metadata.json"

    clean_images = clean_images.astype(np.float32)
    adversarial_images = adversarial_images.astype(np.float32)
    labels = labels.astype(np.float32)
    clean_predictions = clean_predictions.astype(np.int64)
    adversarial_predictions = adversarial_predictions.astype(np.int64)
    source_indices = source_indices.astype(np.int64)

    np.savez_compressed(
        str(output_npz),
        clean_images=clean_images,
        adversarial_images=adversarial_images,
        labels=labels,
        clean_predictions=clean_predictions,
        adversarial_predictions=adversarial_predictions,
        source_indices=source_indices,
    )

    true_labels = labels.argmax(axis=1)
    success_count = int(np.sum(adversarial_predictions != true_labels))
    metadata = dict(metadata)
    metadata.update(
        {
            "attack_name": attack_name,
            "samples": int(len(adversarial_images)),
            "success_count": success_count,
            "output_npz": str(output_npz),
            "metadata_json": str(metadata_json),
        }
    )
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return AdversarialGenerationResult(
        attack_name=attack_name,
        output_npz=output_npz,
        metadata_json=metadata_json,
        samples=int(len(adversarial_images)),
        success_count=success_count,
    )


def _validate_slice(start: int, samples: int, total: int) -> None:
    if samples <= 0:
        raise ValueError("samples must be positive.")
    if start < 0:
        raise ValueError("start must be non-negative.")
    if start + samples > total:
        raise ValueError("Requested slice {}:{} exceeds dataset size {}.".format(start, start + samples, total))


def generate_fgsm_examples(
    config: Optional[MnistExperimentConfig] = None,
    samples: int = 10,
    start: Optional[int] = None,
    eps: Optional[float] = None,
    output_dir: Optional[Path] = None,
) -> AdversarialGenerationResult:
    """Generate M1/FGSM adversarial examples with CleverHans."""

    started_at = time.time()
    cfg = config or MnistExperimentConfig()
    sample_start = cfg.splits.detector_test_start if start is None else start
    attack_eps = cfg.attacks.fgsm_test_eps if eps is None else eps

    import keras
    from keras import backend as keras_backend
    import tensorflow as tf
    from cleverhans.utils_keras import cnn_model
    from cleverhans.utils_mnist import data_mnist

    print("[FGSM] samples={} start={} eps={}".format(samples, sample_start, attack_eps), flush=True)
    print("[FGSM] checkpoint_dir={}".format(cfg.paths.m1_checkpoint_dir), flush=True)

    keras_backend.set_image_dim_ordering("tf")
    sess = tf.Session()
    keras_backend.set_session(sess)

    _, _, test_images, test_labels = data_mnist(
        train_start=cfg.splits.train_start,
        train_end=cfg.splits.train_end,
        test_start=cfg.splits.test_start,
        test_end=cfg.splits.test_end,
    )
    _validate_slice(sample_start, samples, len(test_images))

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    model = cnn_model()
    predictions = model(x)
    checkpoint = tf.train.get_checkpoint_state(str(cfg.paths.m1_checkpoint_dir))
    checkpoint_path = None if checkpoint is None else checkpoint.model_checkpoint_path
    if checkpoint_path is None:
        raise FileNotFoundError(
            "M1 checkpoint not found in {}. Run scripts/mnist/run_m1.py first.".format(
                cfg.paths.m1_checkpoint_dir
            )
        )

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    print("[FGSM] loaded_checkpoint={}".format(checkpoint_path), flush=True)

    clean_images = test_images[sample_start : sample_start + samples]
    labels = test_labels[sample_start : sample_start + samples]
    adversarial_images = generate_fgsm(sess, x, model, clean_images, attack_eps)
    clean_predictions = sess.run(tf.argmax(predictions, axis=1), feed_dict={x: clean_images})
    adversarial_predictions = sess.run(tf.argmax(predictions, axis=1), feed_dict={x: adversarial_images})
    source_indices = np.arange(sample_start, sample_start + samples)

    result = _save_artifact(
        output_dir or _artifact_dir(cfg, "fgsm"),
        "fgsm",
        clean_images,
        adversarial_images,
        labels,
        clean_predictions,
        adversarial_predictions,
        source_indices,
        {
            "category": "faithful_reproduction",
            "model": "M1 CleverHans cnn_model",
            "attack": "CleverHans FastGradientMethod",
            "eps": attack_eps,
            "checkpoint_path": checkpoint_path,
            "pixel_range": "[0, 1]",
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
    )
    print(
        "[FGSM] saved={} success_count={}/{}".format(
            result.output_npz,
            result.success_count,
            result.samples,
        ),
        flush=True,
    )
    return result


def _cw_attack_config(
    config: AttackConfig,
    attack_name: str,
    max_iterations: Optional[int],
    binary_search_steps: Optional[int],
) -> AttackConfig:
    if attack_name == "cw_l2":
        return replace(
            config,
            cw_l2_max_iterations=config.cw_l2_max_iterations if max_iterations is None else max_iterations,
            cw_l2_binary_search_steps=(
                config.cw_l2_binary_search_steps if binary_search_steps is None else binary_search_steps
            ),
        )
    return replace(
        config,
        cw_linf_max_iterations=config.cw_linf_max_iterations if max_iterations is None else max_iterations,
    )


def generate_cw_examples(
    attack_name: str,
    config: Optional[MnistExperimentConfig] = None,
    samples: int = 10,
    start: Optional[int] = None,
    output_dir: Optional[Path] = None,
    max_iterations: Optional[int] = None,
    binary_search_steps: Optional[int] = None,
) -> AdversarialGenerationResult:
    """Generate M2/C&W L2 or Linf adversarial examples with Carlini's code."""

    if attack_name not in {"cw_l2", "cw_linf"}:
        raise ValueError("attack_name must be 'cw_l2' or 'cw_linf'.")

    started_at = time.time()
    cfg = config or MnistExperimentConfig()
    sample_start = cfg.splits.detector_test_start if start is None else start
    attack_cfg = _cw_attack_config(cfg.attacks, attack_name, max_iterations, binary_search_steps)

    import tensorflow as tf
    from keras import backend as keras_backend

    ensure_carlini_mnist_data(cfg.paths, source_dir=None)
    _, _, test_images, test_labels = load_mnist_from_idx(cfg.paths.carlini_data_dir, m2_range=True)
    _validate_slice(sample_start, samples, len(test_images))

    print("[{}] samples={} start={}".format(attack_name, samples, sample_start), flush=True)
    print("[{}] weights_path={}".format(attack_name, cfg.paths.m2_weights_path), flush=True)
    if attack_name == "cw_l2":
        print(
            "[cw_l2] max_iterations={} binary_search_steps={} learning_rate={} initial_const={}".format(
                attack_cfg.cw_l2_max_iterations,
                attack_cfg.cw_l2_binary_search_steps,
                attack_cfg.cw_l2_learning_rate,
                attack_cfg.cw_l2_initial_const,
            ),
            flush=True,
        )
    else:
        print("[cw_linf] max_iterations={}".format(attack_cfg.cw_linf_max_iterations), flush=True)

    sess = tf.Session()
    keras_backend.set_session(sess)
    model = build_m2_inference_model(cfg.paths.m2_weights_path)

    clean_images, labels = generate_untargeted_data(test_images, test_labels, samples, sample_start)
    clean_predictions = model.predict(clean_images).argmax(axis=1)
    if attack_name == "cw_l2":
        adversarial_images = generate_cw_l2(
            sess,
            model,
            clean_images,
            labels,
            cfg.paths.nn_robust_attacks_root,
            attack_cfg,
        )
    else:
        adversarial_images = generate_cw_linf(
            sess,
            model,
            clean_images,
            labels,
            cfg.paths.nn_robust_attacks_root,
            attack_cfg,
        )
    adversarial_predictions = model.predict(adversarial_images).argmax(axis=1)
    source_indices = np.arange(sample_start, sample_start + samples)

    result = _save_artifact(
        output_dir or _artifact_dir(cfg, attack_name),
        attack_name,
        clean_images,
        adversarial_images,
        labels,
        clean_predictions,
        adversarial_predictions,
        source_indices,
        {
            "category": "faithful_reproduction",
            "model": "M2 Carlini MNIST",
            "attack": "Carlini {}".format("L2" if attack_name == "cw_l2" else "Linf"),
            "pixel_range": "[-0.5, 0.5]",
            "weights_path": str(cfg.paths.m2_weights_path),
            "max_iterations": (
                attack_cfg.cw_l2_max_iterations if attack_name == "cw_l2" else attack_cfg.cw_linf_max_iterations
            ),
            "binary_search_steps": attack_cfg.cw_l2_binary_search_steps if attack_name == "cw_l2" else None,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
    )
    print(
        "[{}] saved={} success_count={}/{}".format(
            attack_name,
            result.output_npz,
            result.success_count,
            result.samples,
        ),
        flush=True,
    )
    return result
