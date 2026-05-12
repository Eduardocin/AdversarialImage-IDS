"""Train/load the project-owned MNIST M1 implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mnist.config import M1Config, MnistExperimentConfig
from src.mnist.m1 import run_m1


def build_parser() -> argparse.ArgumentParser:
    config = MnistExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.m1.epochs)
    parser.add_argument("--batch-size", type=int, default=config.m1.batch_size)
    parser.add_argument("--learning-rate", type=float, default=config.m1.learning_rate)
    parser.add_argument("--filename", default=config.m1.filename)
    parser.add_argument(
        "--load-model",
        action="store_true",
        default=config.m1.load_model,
        help="Load checkpoint from outputs/mnist/m1 when available.",
    )
    parser.add_argument(
        "--no-load-model",
        action="store_false",
        dest="load_model",
        help="Force training from scratch.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch for environment/pipeline validation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    m1_config = M1Config(
        epochs=1 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        filename=args.filename,
        load_model=args.load_model,
    )
    result = run_m1(m1_config=m1_config)
    print("m1_clean_accuracy={:.4f}".format(result.clean_accuracy))
    print("m1_checkpoint_dir={}".format(result.checkpoint_dir))
    print("m1_checkpoint_path={}".format(result.checkpoint_path))
    print("m1_trained_from_scratch={}".format(result.trained_from_scratch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
