"""Generate MNIST CW Linf adversarial examples for the M2 model."""

from __future__ import print_function

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List

import numpy as np


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())

from deepdetector.attacks.cw_linf import generate_cw_linf_examples
from deepdetector.data.mnist import load_mnist_data
from deepdetector.evaluation.adversarial import evaluate_attack_success
from deepdetector.models.mnist_cnn import create_tf_session
from deepdetector.models.mnist_m2 import build_mnist_m2_model
from deepdetector.paths import (
    MNIST_M2_ADVERSARIAL_DIR,
    MNIST_M2_CHECKPOINT_DIR,
    MNIST_M2_RESULTS_DIR,
)
from deepdetector.training.train_mnist_m2 import train_or_load_mnist_m2_model


SEED_TF = 1234
SEED_NUMPY = 20170830
CW_LINF_RESULTS_DIR = MNIST_M2_RESULTS_DIR / "cw_linf"
CW_LINF_ADVERSARIAL_DIR = MNIST_M2_ADVERSARIAL_DIR / "cw_linf"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=9000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--initial-tau", type=float, default=1.0)
    parser.add_argument("--const", type=float, default=1.0)
    parser.add_argument("--tau-decay", type=float, default=0.9)
    parser.add_argument("--tau-check-interval", type=int, default=50)
    parser.add_argument(
        "--train-dir",
        default=str(MNIST_M2_CHECKPOINT_DIR),
        help="Directory containing the M2 TensorFlow checkpoint.",
    )
    parser.add_argument(
        "--adversarial-dir",
        default=str(CW_LINF_ADVERSARIAL_DIR),
        help="Directory where CW Linf adversarial .npy arrays are stored.",
    )
    parser.add_argument("--output-dir", default=str(CW_LINF_RESULTS_DIR))
    return parser


def _timestamp() -> str:
    """Return a human-readable local timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _linf_distortions(clean: np.ndarray, adv: np.ndarray) -> np.ndarray:
    """Return per-example Linf distortions."""
    return np.max(np.abs(adv - clean).reshape((len(clean), -1)), axis=1)


def append_progress_csv(
    output_dir: Path,
    batch_start: int,
    batch_end: int,
    total: int,
    elapsed_seconds: float,
) -> Path:
    """Append one CW Linf batch progress row."""
    path = output_dir / "progress.csv"
    write_header = not path.exists()
    fieldnames = [
        "timestamp",
        "batch_start",
        "batch_end",
        "n_done",
        "n_total",
        "percent_done",
        "elapsed_seconds",
    ]
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": _timestamp(),
                "batch_start": int(batch_start),
                "batch_end": int(batch_end),
                "n_done": int(batch_end),
                "n_total": int(total),
                "percent_done": float(100.0 * batch_end / float(total)),
                "elapsed_seconds": float(elapsed_seconds),
            }
        )
    return path


def write_summary_csv(output_dir: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    """Write CW Linf summary CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"
    fieldnames = [
        "experiment",
        "dataset",
        "model",
        "attack",
        "norm",
        "n_total",
        "n_clean_correct",
        "n_clean_wrong",
        "n_attack_success",
        "n_attack_failed",
        "clean_accuracy",
        "adversarial_accuracy",
        "attack_success_rate",
        "mean_linf_distortion",
        "median_linf_distortion",
        "adv_examples_path",
        "seed_tf",
        "seed_numpy",
        "status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_summary_md(output_dir: Path, row: Dict[str, Any], args: argparse.Namespace) -> Path:
    """Write CW Linf summary Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.md"
    lines = [
        "# MNIST M2 CW Linf",
        "",
        "## Configuration",
        "",
        "- samples: {0}".format(args.samples),
        "- start_index: {0}".format(args.start_index),
        "- batch_size: {0}".format(args.batch_size),
        "- max_iterations: {0}".format(args.max_iterations),
        "- learning_rate: {0}".format(args.learning_rate),
        "- confidence: {0}".format(args.confidence),
        "- initial_tau: {0}".format(args.initial_tau),
        "- const: {0}".format(args.const),
        "- tau_decay: {0}".format(args.tau_decay),
        "- tau_check_interval: {0}".format(args.tau_check_interval),
        "",
        "| n_total | clean_accuracy | adversarial_accuracy | attack_success_rate | mean_linf | median_linf | status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        "| {n_total} | {clean_accuracy:.6f} | {adversarial_accuracy:.6f} | "
        "{attack_success_rate:.6f} | {mean_linf_distortion:.6f} | "
        "{median_linf_distortion:.6f} | {status} |".format(**row),
        "",
        "The attack optimizes a margin objective with a shrinking Linf threshold.",
        "",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    """Run manual CW Linf generation and evaluation."""
    args = build_parser().parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")

    try:
        from keras import backend as K

        if hasattr(K, "set_learning_phase"):
            K.set_learning_phase(0)
    except Exception:
        pass

    import tensorflow as tf

    tf.compat.v1.set_random_seed(SEED_TF)
    rng = np.random.RandomState([2017, 8, 30])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adversarial_dir = Path(args.adversarial_dir)
    adversarial_dir.mkdir(parents=True, exist_ok=True)

    end_index = args.start_index + args.samples
    sess = create_tf_session()
    X_train, Y_train, X_test, Y_test = load_mnist_data(
        test_start=args.start_index,
        test_end=end_index,
        rng=rng,
    )
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 28, 28, 1), name="x")
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name="y")
    model, predictions = build_mnist_m2_model(x)

    train_or_load_mnist_m2_model(
        sess=sess,
        x=x,
        y=y,
        predictions=predictions,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        config={
            "nb_epochs": 10,
            "batch_size": 128,
            "learning_rate": 0.001,
            "train_dir": args.train_dir,
            "filename": "mnist_m2.ckpt",
            "load_model": args.load_model,
            "rng": rng,
        },
    )

    started = time.time()
    print(
        "[{0}] START cw_linf samples={1} batch_size={2} max_iterations={3}".format(
            _timestamp(),
            len(X_test),
            args.batch_size,
            args.max_iterations,
        ),
        flush=True,
    )

    def on_batch_done(batch_start: int, batch_end: int, total: int) -> None:
        elapsed = time.time() - started
        append_progress_csv(output_dir, batch_start, batch_end, total, elapsed)
        print(
            "[{0}] PROGRESS cw_linf n_done={1}/{2} percent={3:.1f}% "
            "elapsed_seconds={4:.1f}".format(
                _timestamp(),
                batch_end,
                total,
                100.0 * batch_end / float(total),
                elapsed,
            ),
            flush=True,
        )

    adv_examples = generate_cw_linf_examples(
        sess=sess,
        model=model,
        x_placeholder=x,
        images=X_test,
        labels=Y_test,
        batch_size=args.batch_size,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        confidence=args.confidence,
        initial_tau=args.initial_tau,
        const=args.const,
        tau_decay=args.tau_decay,
        tau_check_interval=args.tau_check_interval,
        clip_min=0.0,
        clip_max=1.0,
        progress_callback=on_batch_done,
    )

    adv_path = adversarial_dir / "adversarial_examples.npy"
    np.save(str(adv_path), adv_examples)

    metrics = evaluate_attack_success(
        sess=sess,
        x=x,
        predictions=predictions,
        X_clean=X_test,
        X_adv=adv_examples,
        Y_true=Y_test,
    )
    distortions = _linf_distortions(X_test, adv_examples)
    row = {
        "experiment": "mnist_m2_cw_linf",
        "dataset": "mnist",
        "model": "M2",
        "attack": "CW",
        "norm": "Linf",
        "n_total": int(metrics["total_examples"]),
        "n_clean_correct": int(metrics["valid_attack_candidates"]),
        "n_clean_wrong": int(metrics["original_classified_wrong_number"]),
        "n_attack_success": int(metrics["successful_attacks"]),
        "n_attack_failed": int(metrics["disturbed_failure_number"]),
        "clean_accuracy": float(metrics["clean_accuracy"]),
        "adversarial_accuracy": float(metrics["adversarial_accuracy"]),
        "attack_success_rate": float(metrics["attack_success_rate"]),
        "mean_linf_distortion": float(np.mean(distortions)),
        "median_linf_distortion": float(np.median(distortions)),
        "adv_examples_path": str(adv_path),
        "seed_tf": SEED_TF,
        "seed_numpy": SEED_NUMPY,
        "status": "executed",
        "notes": "Uses a TF1 margin objective with a shrinking Linf threshold.",
    }
    csv_path = write_summary_csv(output_dir, [row])
    md_path = write_summary_md(output_dir, row, args)
    print(
        "[{0}] DONE cw_linf adversarial_accuracy={1:.6f} "
        "attack_success_rate={2:.6f} elapsed_seconds={3:.1f}".format(
            _timestamp(),
            row["adversarial_accuracy"],
            row["attack_success_rate"],
            time.time() - started,
        ),
        flush=True,
    )
    print("adv_examples_path={0}".format(adv_path))
    print("summary_csv={0}".format(csv_path))
    print("summary_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

