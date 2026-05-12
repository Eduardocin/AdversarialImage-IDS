"""Generate MNIST adversarial examples for M1/FGSM and M2/C&W."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mnist.adversarial_examples import generate_cw_examples, generate_fgsm_examples
from src.mnist.config import MnistExperimentConfig


def _default_samples(attack: str, full: bool, config: MnistExperimentConfig) -> int:
    if not full:
        return 10
    if attack == "fgsm":
        return config.splits.test_end - config.splits.detector_test_start
    return config.source_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attack", choices=["fgsm", "cw-l2", "cw-linf", "all"])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Use reference sample counts.")
    parser.add_argument("--eps", type=float, default=None, help="Override FGSM epsilon.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override C&W max iterations.")
    parser.add_argument(
        "--binary-search-steps",
        type=int,
        default=None,
        help="Override C&W L2 binary search steps.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = MnistExperimentConfig()
    attacks = ["fgsm", "cw-l2", "cw-linf"] if args.attack == "all" else [args.attack]

    for attack in attacks:
        samples = args.samples if args.samples is not None else _default_samples(attack, args.full, config)
        if attack == "fgsm":
            generate_fgsm_examples(config=config, samples=samples, start=args.start, eps=args.eps)
        elif attack == "cw-l2":
            generate_cw_examples(
                "cw_l2",
                config=config,
                samples=samples,
                start=args.start,
                max_iterations=args.max_iterations,
                binary_search_steps=args.binary_search_steps,
            )
        elif attack == "cw-linf":
            generate_cw_examples(
                "cw_linf",
                config=config,
                samples=samples,
                start=args.start,
                max_iterations=args.max_iterations,
                binary_search_steps=args.binary_search_steps,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

