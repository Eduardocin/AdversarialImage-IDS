"""Generate MNIST CW L2 adversarial examples for the M2 model."""

from __future__ import print_function

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List

import numpy as np


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())

from deepdetector.attacks.cw_l2 import generate_cw_l2_examples
from deepdetector.data.mnist import load_mnist_data
from deepdetector.evaluation.adversarial import evaluate_attack_success
from deepdetector.models.mnist_cnn import create_tf_session
from deepdetector.models.mnist_m2 import build_mnist_m2_model
from deepdetector.training.train_mnist_m2 import train_or_load_mnist_m2_model


SEED_TF = 1234
SEED_NUMPY = 20170830
CLEAN_BASELINE_DIR = PROJECT_ROOT / "results" / "mnist" / "m2_cw" / "clean_baseline"
CW_L2_RESULTS_DIR = PROJECT_ROOT / "results" / "mnist" / "m2_cw" / "cw_l2"


def parse_float_list(value: str) -> List[float]:
    """Parse a comma-separated float list."""
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required.")
    return values


def format_kappa(kappa: float) -> str:
    """Format kappa for stable result directory names."""
    value = float(kappa)
    if value.is_integer():
        return "{0:.1f}".format(value).replace(".", "p")
    return "{0:g}".format(value).replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--kappas", default="0.0,0.5,1.0,2.0,4.0")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=9000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--binary-search-steps", type=int, default=5)
    parser.add_argument(
        "--train-dir",
        default=str(CLEAN_BASELINE_DIR / "checkpoints"),
        help="Directory containing the M2 TensorFlow checkpoint.",
    )
    parser.add_argument("--output-dir", default=str(CW_L2_RESULTS_DIR))
    return parser


def _distortions(clean: np.ndarray, adv: np.ndarray) -> np.ndarray:
    """Return per-example L2 distortions."""
    flat = (adv - clean).reshape((len(clean), -1))
    return np.sqrt(np.sum(flat * flat, axis=1))


def _timestamp() -> str:
    """Return a human-readable local timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _progress_kappas(kappas: List[float]) -> Iterable[float]:
    """Return a tqdm progress iterator when tqdm is available."""
    try:
        from tqdm import tqdm

        return tqdm(
            kappas,
            desc="CW L2 kappas",
            unit="kappa",
            dynamic_ncols=True,
            file=sys.stdout,
        )
    except Exception:
        return kappas


def write_summary_csv(output_dir: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    """Write aggregate CW L2 metrics as CSV."""
    path = output_dir / "summary.csv"
    fieldnames = [
        "experiment",
        "dataset",
        "model",
        "attack",
        "norm",
        "kappa",
        "n_total",
        "n_clean_correct",
        "n_clean_wrong",
        "n_attack_success",
        "n_attack_failed",
        "clean_accuracy",
        "adversarial_accuracy",
        "attack_success_rate",
        "mean_l2_distortion",
        "median_l2_distortion",
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


def append_progress_csv(
    output_dir: Path,
    kappa: float,
    batch_start: int,
    batch_end: int,
    total: int,
    elapsed_seconds: float,
) -> Path:
    """Append one CW L2 batch progress row."""
    path = output_dir / "progress.csv"
    write_header = not path.exists()
    fieldnames = [
        "timestamp",
        "kappa",
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
                "kappa": float(kappa),
                "batch_start": int(batch_start),
                "batch_end": int(batch_end),
                "n_done": int(batch_end),
                "n_total": int(total),
                "percent_done": float(100.0 * batch_end / float(total)),
                "elapsed_seconds": float(elapsed_seconds),
            }
        )
    return path


def write_summary_md(output_dir: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> Path:
    """Write aggregate CW L2 metrics as Markdown."""
    path = output_dir / "summary.md"
    lines = [
        "# MNIST M2 CW L2",
        "",
        "## Configuration",
        "",
        "- samples: {0}".format(args.samples),
        "- start_index: {0}".format(args.start_index),
        "- kappas: {0}".format(", ".join("{0:g}".format(k) for k in args.kappas)),
        "- batch_size: {0}".format(args.batch_size),
        "- max_iterations: {0}".format(args.max_iterations),
        "- learning_rate: {0}".format(args.learning_rate),
        "- binary_search_steps: {0}".format(args.binary_search_steps),
        "- train_dir: `{0}`".format(args.train_dir),
        "",
        "| kappa | n_total | clean_accuracy | adversarial_accuracy | attack_success_rate | mean_l2 | median_l2 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {kappa:g} | {n_total} | {clean_accuracy:.6f} | {adversarial_accuracy:.6f} | "
            "{attack_success_rate:.6f} | {mean_l2_distortion:.6f} | "
            "{median_l2_distortion:.6f} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    """Run CW L2 generation and evaluation."""
    args = build_parser().parse_args()
    args.kappas = parse_float_list(args.kappas)
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

    rows = []
    for kappa in _progress_kappas(args.kappas):
        kappa_started = time.time()
        kappa_dir = output_dir / "kappa_{0}".format(format_kappa(kappa))
        kappa_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[{0}] START cw_l2 kappa={1:g} samples={2} batch_size={3} "
            "max_iterations={4} binary_search_steps={5}".format(
                _timestamp(),
                kappa,
                len(X_test),
                args.batch_size,
                args.max_iterations,
                args.binary_search_steps,
            ),
            flush=True,
        )

        def on_batch_done(batch_start: int, batch_end: int, total: int) -> None:
            elapsed = time.time() - kappa_started
            append_progress_csv(output_dir, kappa, batch_start, batch_end, total, elapsed)
            print(
                "[{0}] PROGRESS cw_l2 kappa={1:g} n_done={2}/{3} "
                "percent={4:.1f}% elapsed_seconds={5:.1f}".format(
                    _timestamp(),
                    kappa,
                    batch_end,
                    total,
                    100.0 * batch_end / float(total),
                    elapsed,
                ),
                flush=True,
            )

        adv_examples = generate_cw_l2_examples(
            sess=sess,
            model=model,
            x_placeholder=x,
            images=X_test,
            labels=Y_test,
            confidence=kappa,
            batch_size=args.batch_size,
            max_iterations=args.max_iterations,
            learning_rate=args.learning_rate,
            binary_search_steps=args.binary_search_steps,
            clip_min=0.0,
            clip_max=1.0,
            progress_callback=on_batch_done,
        )
        adv_path = kappa_dir / "adversarial_examples.npy"
        np.save(str(adv_path), adv_examples)
        print(
            "[{0}] SAVED cw_l2 kappa={1:g} path={2}".format(
                _timestamp(),
                kappa,
                adv_path,
            ),
            flush=True,
        )

        metrics = evaluate_attack_success(
            sess=sess,
            x=x,
            predictions=predictions,
            X_clean=X_test,
            X_adv=adv_examples,
            Y_true=Y_test,
        )
        distortions = _distortions(X_test, adv_examples)
        n_clean_wrong = int(metrics["original_classified_wrong_number"])
        n_clean_correct = int(metrics["valid_attack_candidates"])
        row = {
            "experiment": "mnist_m2_cw_l2",
            "dataset": "mnist",
            "model": "M2",
            "attack": "CW",
            "norm": "L2",
            "kappa": float(kappa),
            "n_total": int(metrics["total_examples"]),
            "n_clean_correct": n_clean_correct,
            "n_clean_wrong": n_clean_wrong,
            "n_attack_success": int(metrics["successful_attacks"]),
            "n_attack_failed": int(metrics["disturbed_failure_number"]),
            "clean_accuracy": float(metrics["clean_accuracy"]),
            "adversarial_accuracy": float(metrics["adversarial_accuracy"]),
            "attack_success_rate": float(metrics["attack_success_rate"]),
            "mean_l2_distortion": float(np.mean(distortions)),
            "median_l2_distortion": float(np.median(distortions)),
            "adv_examples_path": str(adv_path),
            "seed_tf": SEED_TF,
            "seed_numpy": SEED_NUMPY,
            "status": "executed",
            "notes": "Last 1000 MNIST test digits by default: indices 9000-9999.",
        }
        rows.append(row)
        print(
            "[{0}] DONE cw_l2 kappa={1:g} adversarial_accuracy={2:.6f} "
            "attack_success_rate={3:.6f} elapsed_seconds={4:.1f}".format(
                _timestamp(),
                kappa,
                row["adversarial_accuracy"],
                row["attack_success_rate"],
                time.time() - kappa_started,
            ),
            flush=True,
        )

    csv_path = write_summary_csv(output_dir, rows)
    md_path = write_summary_md(output_dir, rows, args)
    print("summary_csv={0}".format(csv_path))
    print("summary_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

