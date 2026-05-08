"""Select MNIST detector parameters using FGSM, matching DeepDetector's Train phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.datasets.mnist import MnistNpzDataset
from src.detector.config import default_mnist_detector_candidates
from src.detector.deepdetector import evaluate_mnist_fgsm_detector
from src.detector.selection import (
    MnistDetectorCandidateResult,
    select_best_mnist_detector,
)
from src.models.target_models import load_torchscript_target
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "mnist" / "mnist_splits.npz"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "detectors" / "mnist_fgsm_detector.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--target-model-path",
        type=Path,
        required=True,
        help="Path to an already trained TorchScript MNIST target model.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    import torch
    from torch.utils.data import DataLoader, Subset

    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MnistNpzDataset(args.dataset_path, "train")
    if args.debug:
        dataset = Subset(dataset, range(min(64, len(dataset))))

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    model = load_torchscript_target(args.target_model_path, device=device)

    results: list[MnistDetectorCandidateResult] = []
    for config in default_mnist_detector_candidates():
        counts = evaluate_mnist_fgsm_detector(
            model,
            dataloader,
            epsilon=args.epsilon,
            transform=config,
            device=device,
        )
        result = MnistDetectorCandidateResult(config=config, counts=counts)
        results.append(result)
        print(
            "candidate="
            f"{config.to_dict()} "
            f"precision={counts.precision:.4f} "
            f"recall={counts.recall:.4f} "
            f"f1={counts.f1:.4f} "
            f"fp={counts.fp}"
        )

    best = select_best_mnist_detector(results)
    payload = {
        "phase": "train",
        "dataset": "mnist",
        "attack": "fgsm",
        "epsilon": args.epsilon,
        "split": "train",
        "selection_rule": "max_f1_then_precision_then_recall_then_lower_fp",
        "selected": best.to_dict(),
        "candidates": [result.to_dict() for result in results],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"selected_config={best.config.to_dict()}")
    print(f"detector_config_saved={args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
