#!/usr/bin/env python3
"""Audio filter V2 sweep: fixed quantization vs entropy-adaptive filter.

This experiment checks whether the image-inspired entropy branching used in the
DeepDetector-style audio PoC is really useful for log-Mel spectrograms.

It compares:
    1. Current calibrated entropy-adaptive filter from audio_adversarial_poc.py.
    2. Simpler fixed quantization filters with optional smoothing.

The main question is:
    Can global dB normalization + fixed scalar quantization match or beat the
    entropy-based filter while keeping false positives low?

Example:
    python scripts/dev/audio_filter_v2_fixed_quantization_sweep_poc.py \
        --device cuda \
        --data-dir data/raw/speech_commands \
        --checkpoint artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt \
        --max-test 1200 \
        --batch-size 64 \
        --epsilon 2.0 \
        --quantizations "2,3,4,5,6,8,12,16" \
        --smoothings "none,mean3,cross3,cross5" \
        --normalizations "global"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from audio_adversarial_poc import (
    DB_MAX,
    DB_MIN,
    DEFAULT_CLASSES,
    AudioFilterConfig,
    PaperStyleMetrics,
    SpeechCNN,
    SpeechCommandsSpectrogramDataset,
    apply_smoothing_uint8,
    configure_backend,
    deepdetector_filter_batch,
    denormalize_spectrogram_from_filters,
    fgsm_attack,
    log_environment,
    normalize_spectrogram_for_filters,
    predict,
    quantize_uint8,
    resolve_device,
    safe_div,
    safe_torch_load,
    set_seed,
)

LOGGER = logging.getLogger("audio_filter_v2_fixed_quantization_sweep_poc")


@dataclass(frozen=True)
class FixedFilterConfig:
    normalization: str
    quantization: int
    smoothing: str
    smoothing_combine: str

    @property
    def name(self) -> str:
        return (
            f"fixed_q_{self.quantization}__"
            f"smooth_{self.smoothing}__"
            f"combine_{self.smoothing_combine}__"
            f"norm_{self.normalization}"
        )


@dataclass
class EntropyStats:
    mean: float
    std: float
    min: float
    p25: float
    median: float
    p75: float
    max: float


@dataclass
class CachedBatch:
    x: torch.Tensor
    y: torch.Tensor
    x_adv: torch.Tensor
    pred_clean: torch.Tensor
    pred_adv: torch.Tensor


def parse_csv_options(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one value")
    return items


def parse_csv_ints(value: str) -> List[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer value")
    if any(item < 2 for item in items):
        raise ValueError("Quantization values must be >= 2")
    return items


def entropy_uint8(image: np.ndarray) -> float:
    hist = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    probs = hist / max(float(hist.sum()), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def entropy_stats(values: Sequence[float]) -> EntropyStats:
    if not values:
        return EntropyStats(mean=0.0, std=0.0, min=0.0, p25=0.0, median=0.0, p75=0.0, max=0.0)
    arr = np.asarray(values, dtype=np.float64)
    return EntropyStats(
        mean=float(arr.mean()),
        std=float(arr.std()),
        min=float(arr.min()),
        p25=float(np.percentile(arr, 25)),
        median=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
        max=float(arr.max()),
    )


def build_fixed_configs(args: argparse.Namespace) -> List[FixedFilterConfig]:
    quantizations = parse_csv_ints(args.quantizations)
    smoothings = parse_csv_options(args.smoothings)
    normalizations = parse_csv_options(args.normalizations)
    combines = parse_csv_options(args.smoothing_combines)

    valid_smoothings = {"none", "mean3", "cross3", "cross5"}
    valid_normalizations = {"global", "sample"}
    valid_combines = {"conservative", "replace"}

    invalid_smoothings = sorted(set(smoothings) - valid_smoothings)
    invalid_normalizations = sorted(set(normalizations) - valid_normalizations)
    invalid_combines = sorted(set(combines) - valid_combines)
    if invalid_smoothings:
        raise ValueError(f"Invalid smoothing modes: {invalid_smoothings}")
    if invalid_normalizations:
        raise ValueError(f"Invalid normalization modes: {invalid_normalizations}")
    if invalid_combines:
        raise ValueError(f"Invalid smoothing combine modes: {invalid_combines}")

    configs: List[FixedFilterConfig] = []
    seen = set()
    for normalization in normalizations:
        for quantization in quantizations:
            for smoothing in smoothings:
                for combine in combines:
                    effective_combine = "none" if smoothing == "none" else combine
                    key = (normalization, quantization, smoothing, effective_combine)
                    if key in seen:
                        continue
                    seen.add(key)
                    configs.append(
                        FixedFilterConfig(
                            normalization=normalization,
                            quantization=quantization,
                            smoothing=smoothing,
                            smoothing_combine=effective_combine,
                        )
                    )
    return configs


def fixed_filter_single(spec_db: torch.Tensor, config: FixedFilterConfig) -> Tuple[torch.Tensor, float, float]:
    if spec_db.ndim != 3 or spec_db.size(0) != 1:
        raise ValueError(f"Expected [1, n_mels, time], got {tuple(spec_db.shape)}")

    spec_np = spec_db.squeeze(0).detach().cpu().numpy().astype(np.float32)
    spec_uint8, spec_min, spec_max = normalize_spectrogram_for_filters(
        spec_np,
        normalization=config.normalization,
    )

    entropy_before = entropy_uint8(spec_uint8)
    quantized = quantize_uint8(spec_uint8, intervals=config.quantization)

    if config.smoothing == "none":
        filtered_uint8 = quantized
    else:
        smoothed = apply_smoothing_uint8(quantized, config.smoothing)
        if config.smoothing_combine == "replace":
            filtered_uint8 = smoothed
        elif config.smoothing_combine == "conservative":
            diff_quant = np.abs(quantized.astype(np.int16) - spec_uint8.astype(np.int16))
            diff_smooth = np.abs(smoothed.astype(np.int16) - spec_uint8.astype(np.int16))
            filtered_uint8 = np.where(diff_quant <= diff_smooth, quantized, smoothed).astype(np.uint8)
        else:
            raise ValueError(f"Unknown smoothing_combine: {config.smoothing_combine}")

    entropy_after = entropy_uint8(filtered_uint8)
    filtered_db = denormalize_spectrogram_from_filters(filtered_uint8, spec_min, spec_max)
    filtered_db = np.clip(filtered_db, DB_MIN, DB_MAX).astype(np.float32)
    return torch.from_numpy(filtered_db).unsqueeze(0), entropy_before, entropy_after


def fixed_filter_batch(
    x_db: torch.Tensor,
    config: FixedFilterConfig,
) -> Tuple[torch.Tensor, List[float], List[float]]:
    device = x_db.device
    filtered_items: List[torch.Tensor] = []
    entropy_before: List[float] = []
    entropy_after: List[float] = []

    for sample in x_db.detach().cpu():
        filtered, before, after = fixed_filter_single(sample, config)
        filtered_items.append(filtered)
        entropy_before.append(before)
        entropy_after.append(after)

    return torch.stack(filtered_items, dim=0).to(device=device, dtype=x_db.dtype), entropy_before, entropy_after


def baseline_filter_batch(
    x_db: torch.Tensor,
    config: AudioFilterConfig,
) -> Tuple[torch.Tensor, List[float], List[float]]:
    """Run the current entropy-adaptive baseline and log approximate entropies.

    The baseline filter already computes entropy internally. Here we recompute
    before/after only for reporting consistency with the fixed-filter rows.
    """
    filtered = deepdetector_filter_batch(x_db, config)
    before_values: List[float] = []
    after_values: List[float] = []
    for original, transformed in zip(x_db.detach().cpu(), filtered.detach().cpu()):
        original_np = original.squeeze(0).numpy().astype(np.float32)
        transformed_np = transformed.squeeze(0).numpy().astype(np.float32)
        original_uint8, _, _ = normalize_spectrogram_for_filters(original_np, normalization=config.normalization)
        transformed_uint8, _, _ = normalize_spectrogram_for_filters(transformed_np, normalization=config.normalization)
        before_values.append(entropy_uint8(original_uint8))
        after_values.append(entropy_uint8(transformed_uint8))
    return filtered, before_values, after_values


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


def cache_attack_batches(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    epsilon: float,
    log_interval: int,
) -> List[CachedBatch]:
    cached: List[CachedBatch] = []
    LOGGER.info("Caching clean/adversarial batches with epsilon=%.4f", epsilon)

    for batch_idx, (x, y) in enumerate(test_loader, start=1):
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")

        pred_clean = predict(model, x)
        x_adv = fgsm_attack(model, x, y, epsilon=epsilon)
        pred_adv = predict(model, x_adv)

        cached.append(
            CachedBatch(
                x=x.detach(),
                y=y.detach(),
                x_adv=x_adv.detach(),
                pred_clean=pred_clean.detach(),
                pred_adv=pred_adv.detach(),
            )
        )

        if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(test_loader):
            LOGGER.info("Cached batch %d/%d", batch_idx, len(test_loader))

    return cached


def compute_metrics_from_filtered_predictions(
    y: torch.Tensor,
    pred_clean: torch.Tensor,
    pred_adv: torch.Tensor,
    pred_clean_filtered: torch.Tensor,
    pred_adv_filtered: torch.Tensor,
) -> Dict[str, int]:
    clean_correct_mask = pred_clean == y
    benign_detected_as_adv = pred_clean != pred_clean_filtered
    attack_success = clean_correct_mask & (pred_adv != y)
    attack_failure = clean_correct_mask & (pred_adv == y)
    detected_adv = pred_adv != pred_adv_filtered
    tp_mask = attack_success & detected_adv
    fn_mask = attack_success & ~detected_adv

    return {
        "num_clean_total": int(y.numel()),
        "num_clean_correct": int(clean_correct_mask.sum().item()),
        "num_clean_misclassified": int((~clean_correct_mask).sum().item()),
        "num_eligible_for_attack": int(clean_correct_mask.sum().item()),
        "num_failures": int(attack_failure.sum().item()),
        "num_successful_adversarial": int(attack_success.sum().item()),
        "clean_correct_total": int(clean_correct_mask.sum().item()),
        "adv_correct_total": int((pred_adv == y).sum().item()),
        "adv_correct_on_eligible": int((clean_correct_mask & (pred_adv == y)).sum().item()),
        "filtered_adv_correct_total": int((pred_adv_filtered == y).sum().item()),
        "tp": int(tp_mask.sum().item()),
        "fn": int(fn_mask.sum().item()),
        "fp": int(benign_detected_as_adv.sum().item()),
        "tn": int((~benign_detected_as_adv).sum().item()),
        "rtp": int((tp_mask & (pred_adv_filtered == y)).sum().item()),
    }


def evaluate_filter_config(
    model: torch.nn.Module,
    cached_batches: Sequence[CachedBatch],
    config_name: str,
    filter_mode: str,
    fixed_config: FixedFilterConfig | None,
    baseline_config: AudioFilterConfig | None,
    log_interval: int,
) -> Dict[str, object]:
    totals = {
        "num_clean_total": 0,
        "num_clean_correct": 0,
        "num_clean_misclassified": 0,
        "num_eligible_for_attack": 0,
        "num_failures": 0,
        "num_successful_adversarial": 0,
        "clean_correct_total": 0,
        "adv_correct_total": 0,
        "adv_correct_on_eligible": 0,
        "filtered_adv_correct_total": 0,
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "tn": 0,
        "rtp": 0,
    }
    entropy_clean_values: List[float] = []
    entropy_adv_values: List[float] = []
    entropy_filtered_clean_values: List[float] = []
    entropy_filtered_adv_values: List[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(cached_batches, start=1):
            if filter_mode == "fixed":
                assert fixed_config is not None
                clean_filtered, clean_entropy, clean_filtered_entropy = fixed_filter_batch(batch.x, fixed_config)
                adv_filtered, adv_entropy, adv_filtered_entropy = fixed_filter_batch(batch.x_adv, fixed_config)
            elif filter_mode == "adaptive_entropy_baseline":
                assert baseline_config is not None
                clean_filtered, clean_entropy, clean_filtered_entropy = baseline_filter_batch(batch.x, baseline_config)
                adv_filtered, adv_entropy, adv_filtered_entropy = baseline_filter_batch(batch.x_adv, baseline_config)
            else:
                raise ValueError(f"Unknown filter mode: {filter_mode}")

            pred_clean_filtered = predict(model, clean_filtered)
            pred_adv_filtered = predict(model, adv_filtered)
            batch_counts = compute_metrics_from_filtered_predictions(
                y=batch.y,
                pred_clean=batch.pred_clean,
                pred_adv=batch.pred_adv,
                pred_clean_filtered=pred_clean_filtered,
                pred_adv_filtered=pred_adv_filtered,
            )
            for key, value in batch_counts.items():
                totals[key] += value

            entropy_clean_values.extend(clean_entropy)
            entropy_adv_values.extend(adv_entropy)
            entropy_filtered_clean_values.extend(clean_filtered_entropy)
            entropy_filtered_adv_values.extend(adv_filtered_entropy)

            if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(cached_batches):
                LOGGER.debug("%s - batch %d/%d", config_name, batch_idx, len(cached_batches))

    recall = safe_div(totals["tp"], totals["tp"] + totals["fn"])
    precision = safe_div(totals["tp"], totals["tp"] + totals["fp"])
    f1 = safe_div(2.0 * recall * precision, recall + precision)
    rtp_percent = 100.0 * safe_div(totals["rtp"], totals["tp"])
    false_positive_rate = safe_div(totals["fp"], totals["fp"] + totals["tn"])

    metrics = PaperStyleMetrics(
        num_clean_total=totals["num_clean_total"],
        num_clean_correct=totals["num_clean_correct"],
        num_clean_misclassified=totals["num_clean_misclassified"],
        num_eligible_for_attack=totals["num_eligible_for_attack"],
        num_failures=totals["num_failures"],
        num_successful_adversarial=totals["num_successful_adversarial"],
        tp=totals["tp"],
        fn=totals["fn"],
        fp=totals["fp"],
        tn=totals["tn"],
        rtp=totals["rtp"],
        rtp_percent=rtp_percent,
        recall=100.0 * recall,
        precision=100.0 * precision,
        f1=100.0 * f1,
        false_positive_rate=100.0 * false_positive_rate,
        clean_accuracy=100.0 * safe_div(totals["clean_correct_total"], totals["num_clean_total"]),
        adversarial_accuracy_all=100.0 * safe_div(totals["adv_correct_total"], totals["num_clean_total"]),
        adversarial_accuracy_on_eligible=100.0
        * safe_div(totals["adv_correct_on_eligible"], totals["num_eligible_for_attack"]),
        filtered_adversarial_accuracy_all=100.0
        * safe_div(totals["filtered_adv_correct_total"], totals["num_clean_total"]),
        attack_success_rate_on_eligible=100.0
        * safe_div(totals["num_successful_adversarial"], totals["num_eligible_for_attack"]),
        attack_success_rate_all=100.0
        * safe_div(totals["num_successful_adversarial"], totals["num_clean_total"]),
    )

    row: Dict[str, object] = {
        "config_name": config_name,
        "filter_mode": filter_mode,
        "normalization": fixed_config.normalization if fixed_config else baseline_config.normalization,
        "quantization": fixed_config.quantization if fixed_config else "adaptive_2_4_6",
        "smoothing": fixed_config.smoothing if fixed_config else baseline_config.smoothing,
        "smoothing_combine": fixed_config.smoothing_combine if fixed_config else "conservative",
    }
    row.update(asdict(metrics))

    for prefix, stats in [
        ("entropy_clean", entropy_stats(entropy_clean_values)),
        ("entropy_adversarial", entropy_stats(entropy_adv_values)),
        ("entropy_filtered_clean", entropy_stats(entropy_filtered_clean_values)),
        ("entropy_filtered_adversarial", entropy_stats(entropy_filtered_adv_values)),
    ]:
        for key, value in asdict(stats).items():
            row[f"{prefix}_{key}"] = value

    return row


def save_rows(rows: Sequence[Dict[str, object]], csv_path: Path, json_path: Path, extra: Dict[str, object]) -> None:
    if not rows:
        raise RuntimeError("No rows to save")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"extra": extra, "results": list(rows)}, f, indent=2)

    LOGGER.info("Saved CSV to %s", csv_path)
    LOGGER.info("Saved JSON to %s", json_path)


def print_top_results(rows: Sequence[Dict[str, object]], top_k: int) -> None:
    by_f1 = sorted(rows, key=lambda row: float(row["f1"]), reverse=True)
    by_recall_low_fpr = sorted(
        rows,
        key=lambda row: (float(row["recall"]), -float(row["false_positive_rate"]), float(row["f1"])),
        reverse=True,
    )

    def print_block(title: str, selected: Sequence[Dict[str, object]]) -> None:
        print(f"\n=== {title} ===")
        print("rank,filter_mode,f1,recall,precision,fpr,rtp_percent,tp,fn,fp,tn,config")
        for rank, row in enumerate(selected, start=1):
            print(
                f"{rank},"
                f"{row['filter_mode']},"
                f"{row['f1']:.2f},"
                f"{row['recall']:.2f},"
                f"{row['precision']:.2f},"
                f"{row['false_positive_rate']:.2f},"
                f"{row['rtp_percent']:.2f},"
                f"{row['tp']},"
                f"{row['fn']},"
                f"{row['fp']},"
                f"{row['tn']},"
                f"{row['config_name']}"
            )

    print_block(f"Top {top_k} by F1", by_f1[:top_k])
    print_block(f"Top {top_k} by recall", by_recall_low_fpr[:top_k])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep fixed quantization filters for audio spectrograms.")
    parser.add_argument("--data-dir", default="data/raw/speech_commands", help="Speech Commands data directory.")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated class labels to use.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt",
        help="Trained SpeechResCNN checkpoint.",
    )
    parser.add_argument("--epsilon", type=float, default=2.0, help="FGSM epsilon in dB units.")
    parser.add_argument("--max-test", type=int, default=1200, help="Limit test samples. Use <=0 for all.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device selection.")
    parser.add_argument("--disable-cudnn", action="store_true", help="Use CUDA without cuDNN.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout used by checkpoint architecture.")
    parser.add_argument("--no-download", action="store_true", help="Disable dataset download.")
    parser.add_argument("--quantizations", default="2,3,4,5,6,8,12,16")
    parser.add_argument("--smoothings", default="none,mean3,cross3,cross5")
    parser.add_argument("--normalizations", default="global")
    parser.add_argument("--smoothing-combines", default="conservative,replace")
    parser.add_argument("--include-baseline", action="store_true", default=True)
    parser.add_argument("--csv-out", default="artifacts/audio_poc/filter_v2_fixed_quantization_sweep.csv")
    parser.add_argument("--json-out", default="artifacts/audio_poc/filter_v2_fixed_quantization_sweep.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=5)
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

    class_names = [item.strip() for item in args.classes.split(",") if item.strip()]
    device = resolve_device(args.device)
    log_environment(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the model first with audio_adversarial_poc.py."
        )

    model = SpeechCNN(num_classes=len(class_names), dropout=args.dropout).to(device)
    payload = safe_torch_load(checkpoint_path, device)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    LOGGER.info("Loaded checkpoint: %s", checkpoint_path)

    test_loader = make_test_loader(args, device=device)
    cached_batches = cache_attack_batches(
        model=model,
        test_loader=test_loader,
        device=device,
        epsilon=args.epsilon,
        log_interval=max(args.log_interval, 1),
    )

    rows: List[Dict[str, object]] = []
    start = time.time()

    if args.include_baseline:
        baseline_config = AudioFilterConfig(
            normalization="global",
            threshold_low=3.0,
            threshold_high=4.0,
            q_low=2,
            q_mid=4,
            q_high=6,
            smoothing="mean3",
        )
        LOGGER.info("Evaluating current entropy-adaptive baseline: %s", baseline_config.name)
        rows.append(
            evaluate_filter_config(
                model=model,
                cached_batches=cached_batches,
                config_name=baseline_config.name,
                filter_mode="adaptive_entropy_baseline",
                fixed_config=None,
                baseline_config=baseline_config,
                log_interval=max(args.log_interval, 1),
            )
        )

    fixed_configs = build_fixed_configs(args)
    LOGGER.info("Running %d fixed quantization configurations", len(fixed_configs))
    for idx, config in enumerate(fixed_configs, start=1):
        config_start = time.time()
        LOGGER.info("[%d/%d] Evaluating %s", idx, len(fixed_configs), config.name)
        row = evaluate_filter_config(
            model=model,
            cached_batches=cached_batches,
            config_name=config.name,
            filter_mode="fixed",
            fixed_config=config,
            baseline_config=None,
            log_interval=max(args.log_interval, 1),
        )
        rows.append(row)
        LOGGER.info(
            "[%d/%d] Done %s - f1=%.2f recall=%.2f precision=%.2f fpr=%.2f rtp=%.2f elapsed=%.1fs",
            idx,
            len(fixed_configs),
            config.name,
            row["f1"],
            row["recall"],
            row["precision"],
            row["false_positive_rate"],
            row["rtp_percent"],
            time.time() - config_start,
        )

    save_rows(
        rows,
        csv_path=Path(args.csv_out),
        json_path=Path(args.json_out),
        extra={
            "classes": class_names,
            "architecture": "SpeechResCNN",
            "checkpoint": str(checkpoint_path),
            "epsilon_db": args.epsilon,
            "device": device.type,
            "max_test": args.max_test,
            "experiment": "fixed_quantization_filter_v2",
            "db_min": DB_MIN,
            "db_max": DB_MAX,
        },
    )
    print_top_results(rows, top_k=args.top_k)
    LOGGER.info("Filter V2 sweep finished in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
