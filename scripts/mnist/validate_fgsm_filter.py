"""Validate M1/FGSM examples with adaptive noise reduction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mnist.config import MnistExperimentConfig
from src.mnist.fgsm_filter_validation import validate_fgsm_filter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = validate_fgsm_filter(
        config=MnistExperimentConfig(),
        input_npz=args.input_npz,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print("fgsm_filter_samples={}".format(result.samples))
    print("fgsm_filter_detected_count={}".format(result.detected_count))
    print("fgsm_filter_restored_count={}".format(result.restored_count))
    print("fgsm_filter_metrics_json={}".format(result.metrics_json))
    print("fgsm_filter_output_npz={}".format(result.output_npz))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
