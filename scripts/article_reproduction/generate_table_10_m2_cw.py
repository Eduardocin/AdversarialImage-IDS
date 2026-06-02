"""Generate and evaluate MNIST M2 CW detector metrics."""

from __future__ import print_function

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "article_reproduction" / "mnist_table_10_m2.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "mnist" / "article_reproduction" / "table_10_m2"

from deepdetector.data.mnist import load_mnist_data  # noqa: E402
from deepdetector.evaluation.article_reproduction import (  # noqa: E402
    adaptive_quantization_filter,
    apply_filter_batch,
    ensure_output_dir,
    format_percent,
    label_to_int,
    predict_labels,
    proposed_detection_filter,
    scalar_filter_for_intervals,
    write_csv,
    write_markdown_table,
)
from deepdetector.models.mnist_cnn import create_tf_session  # noqa: E402
from deepdetector.models.mnist_m2 import build_mnist_m2_model, load_mnist_m2_model  # noqa: E402
from deepdetector.paths import MNIST_M2_CHECKPOINT_DIR  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--generate-attacks",
        action="store_true",
        help="Regenerate configured CW adversarial .npy files before evaluation.",
    )
    parser.add_argument(
        "--overwrite-attacks",
        action="store_true",
        help="Allow --generate-attacks to overwrite existing adversarial .npy files.",
    )
    parser.add_argument(
        "--nn-robust-attacks-root",
        default=None,
        help="Path to a local carlini/nn_robust_attacks checkout. Required for CW generation.",
    )
    parser.add_argument(
        "--only-kappa",
        type=float,
        default=None,
        help="Run only the CW-L2 row with this kappa value.",
    )
    return parser


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve a config path relative to the project root."""
    if path_value in (None, ""):
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: Path) -> Dict[str, Any]:
    """Load the experiment YAML config."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")
    return config


def _set_keras_inference_phase() -> None:
    """Force Keras dropout/batch-norm layers to run in inference mode."""
    try:
        from keras import backend as K

        if hasattr(K, "set_learning_phase"):
            K.set_learning_phase(0)
    except Exception:
        pass


def format_kappa(kappa: float) -> str:
    """Format kappa for existing CW L2 result directory names."""
    value = float(kappa)
    if value.is_integer():
        return "{0:.1f}".format(value).replace(".", "p")
    return "{0:g}".format(value).replace(".", "p")


def format_attack_model_label(name: str, kappa: Optional[float]) -> str:
    """Format the displayed Attack/Model label without a separate kappa column."""
    if kappa in (None, ""):
        return name
    return "{0} (kappa={1:.1f})".format(name, float(kappa))


def create_restored_m2_graph(train_dir: str) -> Dict[str, Any]:
    """Create and restore the M2 TF1 graph."""
    import tensorflow as tf

    sess = create_tf_session()
    _set_keras_inference_phase()
    x_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        shape=(None, 28, 28, 1),
        name="x",
    )
    model, predictions = build_mnist_m2_model(x_placeholder)
    checkpoint = load_mnist_m2_model(sess, train_dir)
    if checkpoint is None:
        raise IOError("No M2 TensorFlow checkpoint found in {0}".format(train_dir))
    return {
        "sess": sess,
        "x": x_placeholder,
        "model": model,
        "predictions": predictions,
        "checkpoint": checkpoint,
    }


def select_filter(name: str):
    """Return the configured detection filter."""
    if name == "final":
        return proposed_detection_filter
    if name == "adaptive":
        return adaptive_quantization_filter
    if name == "scalar":
        return scalar_filter_for_intervals(6)
    raise ValueError("Unsupported filter: {0}".format(name))


def load_mnist_test_slice(start: int, samples: int) -> tuple:
    """Load one MNIST test slice for M2 evaluation."""
    _, _, x_test, y_test = load_mnist_data(test_start=start, test_end=start + samples)
    return np.asarray(x_test, dtype=np.float32), np.asarray(y_test)


def compute_detector_metrics(
    y_true: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
    filtered_clean_pred: np.ndarray,
    filtered_adv_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute Table 10 detector counts and percentage metrics."""
    true_labels = np.asarray(y_true, dtype=np.int64)
    clean_pred = np.asarray(clean_pred, dtype=np.int64)
    adv_pred = np.asarray(adv_pred, dtype=np.int64)
    filtered_clean_pred = np.asarray(filtered_clean_pred, dtype=np.int64)
    filtered_adv_pred = np.asarray(filtered_adv_pred, dtype=np.int64)

    clean_correct = clean_pred == true_labels
    attack_failed = clean_correct & (adv_pred == true_labels)
    effectual = clean_correct & (adv_pred != true_labels)
    detected = filtered_adv_pred != adv_pred
    true_positive = effectual & detected
    false_negative = effectual & ~detected
    false_positive = filtered_clean_pred != clean_pred
    recovered_true_positive = true_positive & (filtered_adv_pred == true_labels)

    tp = int(np.sum(true_positive))
    fn = int(np.sum(false_negative))
    fp = int(np.sum(false_positive))
    rtp = int(np.sum(recovered_true_positive))
    failed = int(np.sum(attack_failed))
    clean_wrong = int(np.sum(~clean_correct))

    recall = tp / float(tp + fn) if tp + fn else 0.0
    precision = tp / float(tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * recall / float(precision + recall) if precision + recall else 0.0
    rtp_rate = rtp / float(tp) if tp else 0.0

    return {
        "n_total": int(len(true_labels)),
        "F": failed,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "RTP": rtp,
        "RTP_percent": float(rtp_rate * 100.0),
        "recall_percent": float(recall * 100.0),
        "precision_percent": float(precision * 100.0),
        "f1_percent": float(f1 * 100.0),
        "n_clean_wrong": clean_wrong,
    }


def evaluate_adversarial_path(
    graph: Dict[str, Any],
    clean_images: np.ndarray,
    labels: np.ndarray,
    adversarial_path: Path,
    filter_fn: Any,
    batch_size: int,
) -> Dict[str, Any]:
    """Evaluate one saved adversarial array."""
    if not adversarial_path.exists():
        raise IOError("Adversarial examples not found: {0}".format(adversarial_path))

    adv_images = np.load(str(adversarial_path)).astype(np.float32)
    if adv_images.ndim != 4 or adv_images.shape[1:] != (28, 28, 1):
        raise ValueError("Expected adversarial array shape (N, 28, 28, 1).")

    sample_count = min(len(clean_images), len(adv_images), len(labels))
    clean = np.asarray(clean_images[:sample_count], dtype=np.float32)
    adv = np.asarray(adv_images[:sample_count], dtype=np.float32)
    y_true = label_to_int(labels[:sample_count])

    sess = graph["sess"]
    x_placeholder = graph["x"]
    predictions = graph["predictions"]

    clean_pred = predict_labels(sess, x_placeholder, predictions, clean, batch_size)
    adv_pred = predict_labels(sess, x_placeholder, predictions, adv, batch_size)
    filtered_clean = apply_filter_batch(filter_fn, clean)
    filtered_adv = apply_filter_batch(filter_fn, adv)
    filtered_clean_pred = predict_labels(
        sess,
        x_placeholder,
        predictions,
        filtered_clean,
        batch_size,
    )
    filtered_adv_pred = predict_labels(
        sess,
        x_placeholder,
        predictions,
        filtered_adv,
        batch_size,
    )

    return compute_detector_metrics(
        y_true=y_true,
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        filtered_clean_pred=filtered_clean_pred,
        filtered_adv_pred=filtered_adv_pred,
    )


class M2NnRobustAdapter(object):
    """Expose the M2 Keras model through the nn_robust_attacks model contract."""

    image_size = 28
    num_channels = 1
    num_labels = 10

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, data: Any) -> Any:
        """Return M2 logits for nn_robust_attacks centered input tensors."""
        return self.model(data + 0.5)


def _load_nn_robust_carlini_l2(root: str) -> Any:
    """Load CarliniL2 from a local nn_robust_attacks checkout."""
    attack_path = Path(str(root)).expanduser() / "l2_attack.py"
    if not attack_path.is_file():
        raise ImportError("Missing nn_robust_attacks l2_attack.py: {0}".format(attack_path))

    spec = importlib.util.spec_from_file_location(
        "deepdetector_nn_robust_l2_attack",
        str(attack_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load nn_robust_attacks from {0}".format(attack_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CarliniL2


def _load_nn_robust_carlini_li(root: str) -> Any:
    """Load CarliniLi from a local nn_robust_attacks checkout."""
    attack_path = Path(str(root)).expanduser() / "li_attack.py"
    if not attack_path.is_file():
        raise ImportError("Missing nn_robust_attacks li_attack.py: {0}".format(attack_path))

    spec = importlib.util.spec_from_file_location(
        "deepdetector_nn_robust_li_attack",
        str(attack_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load nn_robust_attacks from {0}".format(attack_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CarliniLi


def generate_nn_robust_cw_l2_examples(
    graph: Dict[str, Any],
    clean_images: np.ndarray,
    labels: np.ndarray,
    attack_row: Dict[str, Any],
    attack_config: Dict[str, Any],
    root: str,
) -> np.ndarray:
    """Generate CW-L2 examples with carlini/nn_robust_attacks."""
    CarliniL2 = _load_nn_robust_carlini_l2(root)
    kappa = attack_row.get("kappa")
    centered_images = np.asarray(clean_images, dtype=np.float32) - 0.5
    _set_keras_inference_phase()
    attack = CarliniL2(
        graph["sess"],
        M2NnRobustAdapter(graph["model"]),
        batch_size=int(attack_config.get("batch_size", 1)),
        max_iterations=int(attack_config.get("max_iterations", 2000)),
        confidence=float(kappa if kappa is not None else attack_config.get("confidence", 0.0)),
        binary_search_steps=int(attack_config.get("binary_search_steps", 5)),
        initial_const=float(attack_config.get("initial_const", 1.0)),
        learning_rate=float(attack_config.get("learning_rate", 0.1)),
        targeted=bool(attack_config.get("targeted", False)),
        abort_early=bool(attack_config.get("abort_early", True)),
        boxmin=-0.5,
        boxmax=0.5,
    )
    centered_adv = attack.attack(centered_images, labels)
    return np.clip(np.asarray(centered_adv, dtype=np.float32) + 0.5, 0.0, 1.0)


def generate_nn_robust_cw_linf_examples(
    graph: Dict[str, Any],
    clean_images: np.ndarray,
    labels: np.ndarray,
    attack_config: Dict[str, Any],
    root: str,
) -> np.ndarray:
    """Generate CW-Linf examples with carlini/nn_robust_attacks."""
    CarliniLi = _load_nn_robust_carlini_li(root)
    centered_images = np.asarray(clean_images, dtype=np.float32) - 0.5
    _set_keras_inference_phase()
    attack = CarliniLi(
        graph["sess"],
        M2NnRobustAdapter(graph["model"]),
        targeted=bool(attack_config.get("targeted", False)),
        learning_rate=float(attack_config.get("learning_rate", 0.005)),
        max_iterations=int(attack_config.get("max_iterations", 1000)),
        abort_early=bool(attack_config.get("abort_early", True)),
        initial_const=float(attack_config.get("initial_const", 1e-5)),
        largest_const=float(attack_config.get("largest_const", 20.0)),
        reduce_const=bool(attack_config.get("reduce_const", False)),
        decrease_factor=float(attack_config.get("decrease_factor", 0.9)),
        const_factor=float(attack_config.get("const_factor", 2.0)),
    )
    centered_adv = attack.attack(centered_images, labels)
    return np.clip(np.asarray(centered_adv, dtype=np.float32) + 0.5, 0.0, 1.0)


def generate_adversarial_path(
    graph: Dict[str, Any],
    clean_images: np.ndarray,
    labels: np.ndarray,
    attack_row: Dict[str, Any],
    attack_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    evaluation_config: Dict[str, Any],
    overwrite: bool,
    nn_robust_attacks_root: Optional[str] = None,
) -> Path:
    """Generate and save one configured M2 CW adversarial array."""
    adversarial_path = attack_row["adversarial_path"]
    if adversarial_path is None:
        raise ValueError("Attack row must define an adversarial output path.")
    if adversarial_path.exists() and not overwrite:
        raise IOError(
            "Adversarial examples already exist: {0}. Use --overwrite-attacks to regenerate.".format(
                adversarial_path
            )
        )

    norm = str(attack_row["norm"]).lower()
    if not nn_robust_attacks_root:
        raise ValueError("--nn-robust-attacks-root is required for M2 CW generation.")

    if norm == "l2":
        adv_images = generate_nn_robust_cw_l2_examples(
            graph=graph,
            clean_images=clean_images,
            labels=labels,
            attack_row=attack_row,
            attack_config=attack_config,
            root=nn_robust_attacks_root,
        )
        backend = "nn_robust_attacks.CarliniL2"
    elif norm == "linf":
        adv_images = generate_nn_robust_cw_linf_examples(
            graph=graph,
            clean_images=clean_images,
            labels=labels,
            attack_config=attack_config,
            root=nn_robust_attacks_root,
        )
        backend = "nn_robust_attacks.CarliniLi"
    else:
        raise ValueError("Unsupported CW norm for M2 generation: {0}".format(attack_row["norm"]))

    adversarial_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(adversarial_path), adv_images.astype(np.float32))
    write_adversarial_manifest(
        adversarial_path=adversarial_path,
        attack_row=attack_row,
        attack_config=attack_config,
        dataset_config=dataset_config,
        sample_count=int(len(clean_images)),
        backend=backend,
    )
    print("generated_adversarial={0}".format(adversarial_path))
    return adversarial_path


def write_adversarial_manifest(
    *,
    adversarial_path: Path,
    attack_row: Dict[str, Any],
    attack_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    sample_count: int,
    backend: str,
) -> Path:
    """Write metadata beside one generated M2 adversarial array."""
    manifest_path = adversarial_path.parent / "manifest.json"
    manifest = {
        "dataset": dataset_config.get("name", "mnist"),
        "split": dataset_config.get("split", "test"),
        "dataset_start": int(dataset_config.get("start", 5500)),
        "samples": sample_count,
        "model": "M2",
        "attack": attack_row.get("attack", "CW"),
        "norm": attack_row.get("norm"),
        "kappa": attack_row.get("kappa"),
        "backend": backend,
        "adversarial_examples": adversarial_path.name,
        "parameters": {
            key: attack_config[key]
            for key in sorted(attack_config)
            if key not in {"adversarial_path", "adversarial_template", "kappas"}
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def configured_attack_rows(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield concrete attack rows from the YAML config."""
    for attack_config in config.get("attacks", []):
        kappas = attack_config.get("kappas", [None])
        for kappa in kappas:
            if attack_config.get("adversarial_template"):
                kappa_key = format_kappa(float(kappa))
                path = str(attack_config["adversarial_template"]).format(kappa=kappa_key)
            else:
                path = attack_config["adversarial_path"]
            yield {
                "name": attack_config["name"],
                "attack": attack_config["attack"],
                "norm": attack_config["norm"],
                "kappa": kappa,
                "adversarial_path": _resolve_path(path),
                "config": attack_config,
            }


def filter_attack_rows(
    rows: Iterable[Dict[str, Any]],
    only_kappa: Optional[float],
) -> Iterable[Dict[str, Any]]:
    """Filter configured rows for targeted M2 runs."""
    if only_kappa is None:
        for row in rows:
            yield row
        return

    target = float(only_kappa)
    for row in rows:
        kappa = row.get("kappa")
        if kappa is not None and abs(float(kappa) - target) < 1e-9:
            yield row


def main() -> int:
    """Run Table 10 M2 evaluation using existing adversarial examples."""
    args = build_parser().parse_args()
    config = load_config(_resolve_path(args.config) or DEFAULT_CONFIG)

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    detection_config = config.get("detection", {})
    evaluation_config = config.get("evaluation", {})
    metrics_config = config.get("metrics", {})
    output_config = config.get("output", {})

    output_dir = ensure_output_dir(
        str(
            _resolve_path(args.output_dir or output_config.get("results_dir"))
            or DEFAULT_OUTPUT_DIR
        )
    )
    train_dir = str(
        _resolve_path(args.train_dir or model_config.get("checkpoint_dir"))
        or MNIST_M2_CHECKPOINT_DIR
    )
    start = int(dataset_config.get("start", 9000))
    samples = int(dataset_config.get("samples", 1000))
    batch_size = int(evaluation_config.get("batch_size", 256))
    filter_name = str(detection_config.get("filter", "final"))
    filter_fn = select_filter(filter_name)

    clean_images, labels = load_mnist_test_slice(start, samples)
    graph = create_restored_m2_graph(train_dir)
    rows: List[Dict[str, Any]] = []

    try:
        for attack_row in filter_attack_rows(
            configured_attack_rows(config),
            args.only_kappa,
        ):
            adversarial_path = attack_row["adversarial_path"]
            if args.generate_attacks:
                adversarial_path = generate_adversarial_path(
                    graph=graph,
                    clean_images=clean_images,
                    labels=labels,
                    attack_row=attack_row,
                    attack_config=attack_row["config"],
                    dataset_config=dataset_config,
                    evaluation_config=evaluation_config,
                    overwrite=bool(args.overwrite_attacks),
                    nn_robust_attacks_root=args.nn_robust_attacks_root,
                )
            metrics = evaluate_adversarial_path(
                graph=graph,
                clean_images=clean_images,
                labels=labels,
                adversarial_path=adversarial_path,
                filter_fn=filter_fn,
                batch_size=batch_size,
            )
            kappa = attack_row["kappa"]
            row = {
                "attack_model": attack_row["name"],
                "attack": attack_row["attack"],
                "norm": attack_row["norm"],
                "dataset": dataset_config.get("name", "mnist").upper(),
                "model": model_config.get("name", "M2"),
                "kappa": "" if kappa is None else float(kappa),
                "filter": filter_name,
                "adversarial_examples_path": str(adversarial_path),
                "notes": "Evaluated from saved adversarial examples.",
            }
            row.update(metrics)
            rows.append(row)
    finally:
        graph["sess"].close()

    columns = metrics_config.get(
        "columns",
        [
            "Attack/Model",
            "Dataset",
            "#F",
            "TP",
            "FN",
            "FP",
            "RTP",
            "RTP%",
            "Recall",
            "Precision",
            "F1",
        ],
    )
    table_rows: List[Dict[str, Any]] = []
    for row in rows:
        table_rows.append(
            {
                "Attack/Model": format_attack_model_label(row["attack_model"], row["kappa"]),
                "Dataset": row["dataset"],
                "#F": row["F"],
                "TP": row["TP"],
                "FN": row["FN"],
                "FP": row["FP"],
                "RTP": row["RTP"],
                "RTP%": format_percent(row["RTP_percent"]),
                "Recall": format_percent(row["recall_percent"]),
                "Precision": format_percent(row["precision_percent"]),
                "F1": format_percent(row["f1_percent"]),
            }
        )

    csv_path = write_csv(
        str(Path(output_dir) / str(output_config.get("csv", "table_10_m2_mnist_cw.csv"))),
        table_rows,
        metrics_config.get("raw_fields", columns),
    )

    md_path = write_markdown_table(
        str(Path(output_dir) / str(output_config.get("markdown", "table_10_m2_mnist_cw.md"))),
        "MNIST M2 CW Detector Metrics",
        columns,
        [[row[column] for column in columns] for row in table_rows],
    )

    print("results_csv={0}".format(csv_path))
    print("results_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
