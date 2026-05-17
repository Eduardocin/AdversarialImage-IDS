"""Prepare or generate MNIST CW Linf adversarial examples for the M2 model."""

from __future__ import print_function

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List


SCRIPTS_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "_project_root.py").is_file()
)
sys.path.insert(0, str(SCRIPTS_ROOT))
from _project_root import configure_project_paths

PROJECT_ROOT = configure_project_paths(__file__)

from deepdetector.attacks.cw_linf import CwLinfUnavailableError


SEED_TF = 1234
SEED_NUMPY = 20170830
CLEAN_BASELINE_DIR = PROJECT_ROOT / "results" / "mnist" / "m2_cw" / "clean_baseline"
CW_LINF_RESULTS_DIR = PROJECT_ROOT / "results" / "mnist" / "m2_cw" / "cw_linf"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=9000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--train-dir",
        default=str(CLEAN_BASELINE_DIR / "checkpoints"),
        help="Directory containing the M2 TensorFlow checkpoint.",
    )
    parser.add_argument("--output-dir", default=str(CW_LINF_RESULTS_DIR))
    return parser


def _not_executed_row(reason: str) -> Dict[str, Any]:
    """Return a stable not-executed summary row."""
    return {
        "experiment": "mnist_m2_cw_linf",
        "dataset": "mnist",
        "model": "M2",
        "attack": "CW",
        "norm": "Linf",
        "n_total": 0,
        "n_clean_correct": 0,
        "n_clean_wrong": 0,
        "n_attack_success": 0,
        "n_attack_failed": 0,
        "clean_accuracy": "",
        "adversarial_accuracy": "",
        "attack_success_rate": "",
        "mean_linf_distortion": "",
        "median_linf_distortion": "",
        "adv_examples_path": "",
        "seed_tf": SEED_TF,
        "seed_numpy": SEED_NUMPY,
        "status": "not_executed",
        "reason": reason,
        "notes": "No results are invented for CW Linf.",
    }


def write_summary_csv(output_dir: Path, rows: List[Dict[str, Any]]) -> Path:
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
        "reason",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_summary_md(output_dir: Path, reason: str) -> Path:
    """Write CW Linf limitation summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.md"
    lines = [
        "# MNIST M2 CW Linf",
        "",
        "status: not_executed",
        "",
        "## Reason",
        "",
        reason,
        "",
        "No adversarial examples or detector metrics were invented.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    """Record the CW Linf status for this legacy stack."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    del args
    try:
        raise CwLinfUnavailableError(
            "CleverHans 3.1.0 in this project provides CW L2 but no compatible CW Linf attack API."
        )
    except CwLinfUnavailableError as exc:
        reason = str(exc)
        row = _not_executed_row(reason)
        csv_path = write_summary_csv(output_dir, [row])
        md_path = write_summary_md(output_dir, reason)
        print("status=not_executed")
        print("reason={0}".format(reason))
        print("summary_csv={0}".format(csv_path))
        print("summary_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
