#!/usr/bin/env python3
"""Sweep FGSM epsilon values for the audio DeepDetector PoC.

This script reuses a trained SpeechResCNN checkpoint and the calibrated audio
filter from audio_adversarial_poc.py. It evaluates several epsilon values in one
run and saves a consolidated CSV/JSON table.

Example:
    python scripts/dev/audio_epsilon_sweep_poc.py \
        --device cuda \
        --data-dir data/raw/speech_commands \
        --checkpoint artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt \
        --max-test 1200 \
        --batch-size 64 \
        --epsilons "0.25,0.5,1.0,1.5,2.0,2.5,3.0,4.0"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from audio_adversarial_poc import (
    DEFAULT_CLASSES,
    N_MELS,
    SAMPLE_RATE,
    AudioFilterConfig,
    PaperStyleMetrics,
    SpeechCNN,
    SpeechCommandsSpectrogramDataset,
    configure_backend,
    evaluate,
    log_environment,
    resolve_device,
    safe_torch_load,
    set_seed,
)


LOGGER = logging.getLogger("audio_epsilon_sweep_poc")


def parse_epsilons(value: str) -> List[float]:
    epsilons = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not epsilons:
        raise ValueError("At least one epsilon value is required")
    if any(eps < 0 for eps in epsilons):
        raise ValueError("Epsilon values must be non-negative")
    return epsilons


def limit_dataset(dataset: Dataset, max_items: int, seed: int) -> Dataset:
    if max_items <= 0 or max_items >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator).tolist()
    LOGGER.info("Limiting test dataset from %d to %d samples", len(dataset), max_items)
    return Subset(dataset, perm[:max_items])


def make_test_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    class_names = [item.strip() for item in args.classes.split(",") if item.strip()]
    test_ds = SpeechCommandsSpectrogramDataset(
        root=Path(args.data_dir),
        subset="testing",
        classes=class_names,
        download=not args.no_download,
    )
    test_ds = limit_dataset(test_ds, args.max_test, args.seed + 1)
    return DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def build_filter_config(args: argparse.Namespace) -> AudioFilterConfig:
    if args.filter_threshold_low >= args.filter_threshold_high:
        raise ValueError("--filter-threshold-low must be smaller than --filter-threshold-high")
    for value_name in ["filter_q_low", "filter_q_mid", "filter_q_high"]:
        if getattr(args, value_name) < 2:
            raise ValueError(f"--{value_name.replace('_', '-')} must be >= 2")

    return AudioFilterConfig(
        normalization=args.filter_normalization,
        threshold_low=args.filter_threshold_low,
        threshold_high=args.filter_threshold_high,
        q_low=args.filter_q_low,
        q_mid=args.filter_q_mid,
        q_high=args.filter_q_high,
        smoothing=args.filter_smoothing,
    )


def metrics_to_row(epsilon: float, metrics: PaperStyleMetrics, filter_config: AudioFilterConfig) -> Dict[str, object]:
    row: Dict[str, object] = {
        "epsilon": epsilon,
        "filter_config_name": filter_config.name,
        "filter_normalization": filter_config.normalization,
        "filter_threshold_low": filter_config.threshold_low,
        "filter_threshold_high": filter_config.threshold_high,
        "filter_q_low": filter_config.q_low,
        "filter_q_mid": filter_config.q_mid,
        "filter_q_high": filter_config.q_high,
        "filter_smoothing": filter_config.smoothing,
    }
    row.update(asdict(metrics))
    return row


def save_sweep_results(
    rows: Sequence[Dict[str, object]],
    csv_path: Path,
    json_path: Path,
    extra: Dict[str, object],
) -> None:
    if not rows:
        raise RuntimeError("No epsilon sweep results to save")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "extra": extra,
        "results": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    LOGGER.info("Saved epsilon sweep CSV to %s", csv_path)
    LOGGER.info("Saved epsilon sweep JSON to %s", json_path)


def print_summary(rows: Sequence[Dict[str, object]]) -> None:
    print("\n=== Audio FGSM Epsilon Sweep ===")
    print("epsilon,attack_success_rate_on_eligible,recall,precision,f1,false_positive_rate,rtp_percent,filtered_adversarial_accuracy_all,tp,fn,fp,tn")
    for row in rows:
        print(
            f"{row['epsilon']},"
            f"{row['attack_success_rate_on_eligible']:.2f},"
            f"{row['recall']:.2f},"
            f"{row['precision']:.2f},"
            f"{row['f1']:.2f},"
            f"{row['false_positive_rate']:.2f},"
            f"{row['rtp_percent']:.2f},"
            f"{row['filtered_adversarial_accuracy_all']:.2f},"
            f"{row['tp']},"
            f"{row['fn']},"
            f"{row['fp']},"
            f"{row['tn']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep FGSM epsilon values for the audio DeepDetector PoC.")
    parser.add_argument("--data-dir", default="data/raw/speech_commands", help="Speech Commands data directory.")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated class labels to use.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt",
        help="Trained SpeechResCNN checkpoint.",
    )
    parser.add_argument(
        "--epsilons",
        default="0.25,0.5,1.0,1.5,2.0,2.5,3.0,4.0",
        help="Comma-separated FGSM epsilon values in dB units.",
    )
    parser.add_argument("--max-test", type=int, default=1200, help="Limit test samples. Use <=0 for all.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device selection.")
    parser.add_argument("--disable-cudnn", action="store_true", help="Use CUDA without cuDNN.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout used by checkpoint architecture.")
    parser.add_argument("--no-download", action="store_true", help="Disable dataset download.")
    parser.add_argument("--eval-log-interval", type=int, default=5, help="Evaluation log interval in batches.")
    parser.add_argument("--csv-out", default="artifacts/audio_poc/epsilon_sweep.csv")
    parser.add_argument("--json-out", default="artifacts/audio_poc/epsilon_sweep.json")

    # Same calibrated audio filter defaults as audio_adversarial_poc.py.
    parser.add_argument("--filter-normalization", choices=["sample", "global"], default="global")
    parser.add_argument("--filter-threshold-low", type=float, default=3.0)
    parser.add_argument("--filter-threshold-high", type=float, default=4.0)
    parser.add_argument("--filter-q-low", type=int, default=2)
    parser.add_argument("--filter-q-mid", type=int, default=4)
    parser.add_argument("--filter-q-high", type=int, default=6)
    parser.add_argument(
        "--filter-smoothing",
        choices=["none", "cross3", "cross5", "mean3"],
        default="mean3",
    )

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    configure_backend(disable_cudnn=args.disable_cudnn)
    set_seed(args.seed)

    epsilons = parse_epsilons(args.epsilons)
    class_names = [item.strip() for item in args.classes.split(",") if item.strip()]
    filter_config = build_filter_config(args)
    device = resolve_device(args.device)
    log_environment(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the model first with audio_adversarial_poc.py."
        )

    LOGGER.info("Loading checkpoint from %s", checkpoint_path)
    model = SpeechCNN(num_classes=len(class_names), dropout=args.dropout).to(device)
    payload = safe_torch_load(checkpoint_path, device)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()

    LOGGER.info("Using filter config: %s", filter_config.name)
    LOGGER.info("Epsilons: %s", epsilons)

    test_loader = make_test_loader(args, device=device)
    LOGGER.info("Test batches: %d", len(test_loader))

    rows: List[Dict[str, object]] = []
    start = time.time()
    for idx, epsilon in enumerate(epsilons, start=1):
        eps_start = time.time()
        LOGGER.info("[%d/%d] Evaluating epsilon=%.4f", idx, len(epsilons), epsilon)
        metrics = evaluate(
            model=model,
            test_loader=test_loader,
            device=device,
            epsilon=epsilon,
            filter_config=filter_config,
            save_debug=False,
            debug_dir=Path("artifacts/audio/debug"),
            class_names=class_names,
            log_interval=max(args.eval_log_interval, 1),
        )
        row = metrics_to_row(epsilon, metrics, filter_config)
        rows.append(row)
        LOGGER.info(
            "[%d/%d] epsilon=%.4f done - attack_success=%.2f recall=%.2f precision=%.2f f1=%.2f fpr=%.2f elapsed=%.1fs",
            idx,
            len(epsilons),
            epsilon,
            metrics.attack_success_rate_on_eligible,
            metrics.recall,
            metrics.precision,
            metrics.f1,
            metrics.false_positive_rate,
            time.time() - eps_start,
        )

    save_sweep_results(
        rows,
        csv_path=Path(args.csv_out),
        json_path=Path(args.json_out),
        extra={
            "classes": class_names,
            "architecture": "SpeechResCNN",
            "checkpoint": str(checkpoint_path),
            "device": device.type,
            "max_test": args.max_test,
            "filter_config": asdict(filter_config),
            "filter_config_name": filter_config.name,
            "sample_rate": SAMPLE_RATE,
            "n_mels": N_MELS,
        },
    )
    print_summary(rows)
    LOGGER.info("Epsilon sweep finished in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
