"""Test a selected MNIST detector configuration with FGSM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.datasets.mnist import MnistNpzDataset
from src.detector.config import MnistDetectorConfig
from src.detector.deepdetector import evaluate_mnist_fgsm_detector
from src.models.target_models import load_torchscript_target
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "mnist" / "mnist_splits.npz"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "outputs" / "detectors" / "mnist_fgsm_detector.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--detector-config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--target-model-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    import torch
    from torch.utils.data import DataLoader, Subset

    args = parse_args()
    set_seed(args.seed)
    payload = json.loads(args.detector_config_path.read_text(encoding="utf-8"))
    config = MnistDetectorConfig.from_dict(payload["selected"]["config"])
    epsilon = float(payload["epsilon"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MnistNpzDataset(args.dataset_path, "test")
    if args.debug:
        dataset = Subset(dataset, range(min(64, len(dataset))))

    model = load_torchscript_target(args.target_model_path, device=device)
    counts = evaluate_mnist_fgsm_detector(
        model,
        DataLoader(dataset, batch_size=args.batch_size),
        epsilon=epsilon,
        transform=config,
        device=device,
    )

    print(f"phase=test")
    print(f"config={config.to_dict()}")
    print(json.dumps(counts.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
