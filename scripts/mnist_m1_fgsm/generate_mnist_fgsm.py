"""Generate MNIST FGSM adversarial examples with CleverHans."""

from __future__ import print_function

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import numpy as np


SCRIPTS_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "_project_root.py").is_file()
)
sys.path.insert(0, str(SCRIPTS_ROOT))
from _project_root import configure_project_paths

PROJECT_ROOT = configure_project_paths(__file__)

from deepdetector.attacks.fgsm import generate_fgsm_examples
from deepdetector.data.mnist import load_mnist_data
from deepdetector.evaluation.adversarial import evaluate_attack_success
from deepdetector.models.mnist_cnn import build_mnist_model, create_tf_session
from deepdetector.training.train_mnist import train_or_load_mnist_model


CLEAN_BASELINE_DIR = PROJECT_ROOT / "results" / "mnist" / "clean_baseline"
FGSM_RESULTS_DIR = PROJECT_ROOT / "results" / "mnist" / "fgsm"


def parse_epsilons(value: str) -> List[float]:
    """Parse a comma-separated epsilon list."""
    epsilons = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        eps = float(item)
        if eps < 0:
            raise argparse.ArgumentTypeError("epsilons must be non-negative.")
        epsilons.append(eps)
    if not epsilons:
        raise argparse.ArgumentTypeError("at least one epsilon is required.")
    return epsilons


def format_epsilon(eps: float) -> str:
    """Format epsilon for stable result directory names."""
    return ("{0:g}".format(eps)).replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epsilons",
        nargs="+",
        default=["0.2"],
        help="Comma-separated or space-separated epsilon values.",
    )
    parser.add_argument("--samples", type=int, default=4500)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--train-dir",
        default=str(CLEAN_BASELINE_DIR / "checkpoints"),
        help="Directory used for TensorFlow checkpoint files.",
    )
    parser.add_argument("--filename", default="mnist.ckpt")
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Restore an existing checkpoint from --train-dir when available.",
    )
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(FGSM_RESULTS_DIR))
    return parser


def write_summary_csv(output_dir: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    """Write aggregate FGSM metrics as CSV."""
    path = output_dir / "summary.csv"
    fieldnames = [
        "epsilon",
        "total_examples",
        "clean_accuracy",
        "adversarial_accuracy",
        "original_classified_wrong_number",
        "disturbed_failure_number",
        "successful_attacks",
        "valid_attack_candidates",
        "attack_success_rate",
        "adv_min",
        "adv_max",
        "adversarial_examples_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_summary_md(output_dir: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> Path:
    """Write aggregate FGSM metrics as Markdown."""
    path = output_dir / "summary.md"
    lines = [
        "# MNIST FGSM",
        "",
        "## Configuration",
        "",
        "- samples: {0}".format(args.samples),
        "- epsilons: {0}".format(", ".join("{0:g}".format(eps) for eps in args.epsilons)),
        "- clip_min: {0}".format(args.clip_min),
        "- clip_max: {0}".format(args.clip_max),
        "- train_dir: `{0}`".format(args.train_dir),
        "- load_model: {0}".format(args.load_model),
        "",
        "## Metrics",
        "",
        "| epsilon | clean_accuracy | adversarial_accuracy | attack_success_rate | clean_errors | attack_failures | successful_attacks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {epsilon:g} | {clean_accuracy:.6f} | {adversarial_accuracy:.6f} | "
            "{attack_success_rate:.6f} | {original_classified_wrong_number} | "
            "{disturbed_failure_number} | {successful_attacks} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Semantics",
            "",
            "`adversarial_accuracy` is the fraction of adversarial images still "
            "classified as the true label. `attack_success_rate` is computed only "
            "over clean images classified correctly before perturbation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    """Run FGSM generation and evaluation."""
    args = build_parser().parse_args()
    args.epsilons = parse_epsilons(",".join(args.epsilons))
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")

    import tensorflow as tf

    tf.compat.v1.set_random_seed(1234)
    rng = np.random.RandomState([2017, 8, 30])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sess = create_tf_session()
    X_train, Y_train, X_test, Y_test = load_mnist_data(rng=rng)
    sample_count = min(args.samples, len(X_test))
    X_eval = X_test[:sample_count]
    Y_eval = Y_test[:sample_count]

    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 28, 28, 1), name="x")
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name="y")
    model, predictions = build_mnist_model(x)

    train_or_load_mnist_model(
        sess=sess,
        x=x,
        y=y,
        predictions=predictions,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        config={
            "nb_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_dir": args.train_dir,
            "filename": args.filename,
            "load_model": args.load_model,
            "rng": rng,
        },
    )

    rows = []
    for eps in args.epsilons:
        eps_dir = output_dir / "eps_{0}".format(format_epsilon(eps))
        eps_dir.mkdir(parents=True, exist_ok=True)
        adv_examples = generate_fgsm_examples(
            sess=sess,
            model=model,
            x_placeholder=x,
            images=X_eval,
            eps=eps,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
        )
        adv_path = eps_dir / "adversarial_examples.npy"
        np.save(str(adv_path), adv_examples)

        metrics = evaluate_attack_success(
            sess=sess,
            x=x,
            predictions=predictions,
            X_clean=X_eval,
            X_adv=adv_examples,
            Y_true=Y_eval,
        )
        row = dict(metrics)
        row.update(
            {
                "epsilon": eps,
                "adv_min": float(adv_examples.min()),
                "adv_max": float(adv_examples.max()),
                "adversarial_examples_path": str(adv_path),
            }
        )
        rows.append(row)
        print(
            "eps={0:g} adversarial_accuracy={1:.6f} attack_success_rate={2:.6f}".format(
                eps,
                row["adversarial_accuracy"],
                row["attack_success_rate"],
            )
        )

    csv_path = write_summary_csv(output_dir, rows)
    md_path = write_summary_md(output_dir, rows, args)
    print("summary_csv={0}".format(csv_path))
    print("summary_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
