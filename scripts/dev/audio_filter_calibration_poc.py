#!/usr/bin/env python3
"""Audio filter calibration sweep for the DeepDetector PoC.

This script is intentionally separate from the main audio_adversarial_poc.py.
It reuses the trained SpeechResCNN checkpoint and sweeps filter parameters for
log-Mel spectrograms:

    - entropy thresholds;
    - scalar quantization intervals;
    - smoothing strategy;
    - normalization mode before filtering.

The goal is to verify whether the original DeepDetector image thresholds
(entropy < 4, entropy < 5, otherwise high entropy) are appropriate for audio
spectrograms, or whether audio needs different thresholds/quantization.

Example:
    python scripts/dev/audio_filter_calibration_poc.py \
        --device cuda \
        --data-dir data/raw/speech_commands \
        --checkpoint artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt \
        --max-test 1200 \
        --epsilon 2.0 \
        --thresholds "3,4;4,5;5,6;6,7" \
        --quantizations "2,4,6;4,6,8;6,8,16" \
        --smoothings "none,cross3,cross5,mean3" \
        --normalizations "sample,global"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from audio_adversarial_poc import (
    DB_MAX,
    DB_MIN,
    DEFAULT_CLASSES,
    PaperStyleMetrics,
    SpeechCNN,
    configure_backend,
    fgsm_attack,
    log_environment,
    make_loaders,
    predict,
    resolve_device,
    safe_div,
    safe_torch_load,
    set_seed,
)


LOGGER = logging.getLogger("audio_filter_calibration_poc")


@dataclass(frozen=True)
class FilterConfig:
    threshold_low: float
    threshold_high: float
    q_low: int
    q_mid: int
    q_high: int
    smoothing: str
    normalization: str

    @property
    def name(self) -> str:
        return (
            f"thr_{self.threshold_low:g}_{self.threshold_high:g}__"
            f"q_{self.q_low}_{self.q_mid}_{self.q_high}__"
            f"smooth_{self.smoothing}__norm_{self.normalization}"
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
class CalibrationResult:
    config_name: str
    threshold_low: float
    threshold_high: float
    q_low: int
    q_mid: int
    q_high: int
    smoothing: str
    normalization: str
    metrics: PaperStyleMetrics
    entropy_clean: EntropyStats
    entropy_adversarial: EntropyStats
    entropy_filtered_clean: EntropyStats
    entropy_filtered_adversarial: EntropyStats


def parse_thresholds(value: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        left, right = [part.strip() for part in item.split(",")]
        low = float(left)
        high = float(right)
        if low >= high:
            raise ValueError(f"Invalid threshold pair {item!r}: low must be < high")
        pairs.append((low, high))
    if not pairs:
        raise ValueError("At least one threshold pair is required")
    return pairs


def parse_quantizations(value: str) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [int(part.strip()) for part in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid quantization triple {item!r}")
        if any(part < 2 for part in parts):
            raise ValueError(f"Invalid quantization triple {item!r}: all values must be >= 2")
        triples.append((parts[0], parts[1], parts[2]))
    if not triples:
        raise ValueError("At least one quantization triple is required")
    return triples


def parse_csv_options(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one option")
    return items


def build_filter_configs(args: argparse.Namespace) -> List[FilterConfig]:
    thresholds = parse_thresholds(args.thresholds)
    quantizations = parse_quantizations(args.quantizations)
    smoothings = parse_csv_options(args.smoothings)
    normalizations = parse_csv_options(args.normalizations)

    valid_smoothings = {"none", "cross3", "cross5", "mean3"}
    valid_normalizations = {"sample", "global"}

    invalid_smoothings = sorted(set(smoothings) - valid_smoothings)
    invalid_normalizations = sorted(set(normalizations) - valid_normalizations)
    if invalid_smoothings:
        raise ValueError(f"Invalid smoothing options: {invalid_smoothings}")
    if invalid_normalizations:
        raise ValueError(f"Invalid normalization options: {invalid_normalizations}")

    configs: List[FilterConfig] = []
    for threshold_low, threshold_high in thresholds:
        for q_low, q_mid, q_high in quantizations:
            for smoothing in smoothings:
                for normalization in normalizations:
                    configs.append(
                        FilterConfig(
                            threshold_low=threshold_low,
                            threshold_high=threshold_high,
                            q_low=q_low,
                            q_mid=q_mid,
                            q_high=q_high,
                            smoothing=smoothing,
                            normalization=normalization,
                        )
                    )
    return configs


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


def normalize_spectrogram_for_filters(spec_db: np.ndarray, normalization: str) -> Tuple[np.ndarray, float, float]:
    """Map a 2D dB spectrogram to uint8 [0, 255].

    sample:
        Uses each sample's own min/max, which may increase local contrast.
    global:
        Uses the fixed dB range [-80, 0], closer to an image-like fixed input
        scale and usually better aligned with the original DeepDetector setup.
    """
    spec = np.clip(spec_db.astype(np.float32), DB_MIN, DB_MAX)

    if normalization == "global":
        original_min = DB_MIN
        original_max = DB_MAX
    elif normalization == "sample":
        original_min = float(spec.min())
        original_max = float(spec.max())
    else:
        raise ValueError(f"Unknown normalization mode: {normalization}")

    denom = original_max - original_min
    if denom < 1e-8:
        return np.zeros_like(spec, dtype=np.uint8), original_min, original_max

    normalized = (spec - original_min) / denom
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8), original_min, original_max


def denormalize_spectrogram_from_filters(
    filtered_uint8: np.ndarray,
    original_min: float,
    original_max: float,
) -> np.ndarray:
    x = filtered_uint8.astype(np.float32) / 255.0
    return x * (original_max - original_min) + original_min


def entropy_uint8(image: np.ndarray) -> float:
    hist = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    probs = hist / max(float(hist.sum()), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def quantize_uint8(image: np.ndarray, intervals: int) -> np.ndarray:
    image_f = image.astype(np.float32)
    step = 255.0 / float(intervals - 1)
    quantized = np.round(image_f / step) * step
    return np.clip(quantized, 0, 255).astype(np.uint8)


def smooth_cross_uint8(image: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    mask[center, :] = 1.0
    mask[:, center] = 1.0
    mask /= mask.sum()
    return smooth_with_kernel_uint8(image, mask)


def smooth_mean_uint8(image: np.ndarray, size: int) -> np.ndarray:
    mask = np.ones((size, size), dtype=np.float32) / float(size * size)
    return smooth_with_kernel_uint8(image, mask)


def smooth_with_kernel_uint8(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    size = kernel.shape[0]
    pad = size // 2
    padded = np.pad(image.astype(np.float32), pad_width=pad, mode="edge")
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(size):
        for j in range(size):
            weight = kernel[i, j]
            if weight == 0:
                continue
            output += weight * padded[i : i + image.shape[0], j : j + image.shape[1]]

    return np.clip(np.round(output), 0, 255).astype(np.uint8)


def apply_smoothing(image: np.ndarray, smoothing: str) -> np.ndarray:
    if smoothing == "none":
        return image
    if smoothing == "cross3":
        return smooth_cross_uint8(image, size=3)
    if smoothing == "cross5":
        return smooth_cross_uint8(image, size=5)
    if smoothing == "mean3":
        return smooth_mean_uint8(image, size=3)
    raise ValueError(f"Unknown smoothing mode: {smoothing}")


def filter_single(spec_db: torch.Tensor, config: FilterConfig) -> Tuple[torch.Tensor, float, float]:
    """Filter one [1, n_mels, time] spectrogram.

    Returns:
        filtered tensor, entropy_before, entropy_after
    """
    spec_np = spec_db.squeeze(0).detach().cpu().numpy().astype(np.float32)
    spec_uint8, spec_min, spec_max = normalize_spectrogram_for_filters(spec_np, config.normalization)

    entropy_before = entropy_uint8(spec_uint8)
    if entropy_before < config.threshold_low:
        intervals = config.q_low
    elif entropy_before < config.threshold_high:
        intervals = config.q_mid
    else:
        intervals = config.q_high

    quantized = quantize_uint8(spec_uint8, intervals=intervals)
    smoothed = apply_smoothing(quantized, config.smoothing)

    if config.smoothing == "none":
        filtered_uint8 = quantized
    else:
        # Keep the DeepDetector-inspired conservative combination rule:
        # choose the transformed value that changes each cell less.
        diff_quant = np.abs(quantized.astype(np.int16) - spec_uint8.astype(np.int16))
        diff_smooth = np.abs(smoothed.astype(np.int16) - spec_uint8.astype(np.int16))
        filtered_uint8 = np.where(diff_quant <= diff_smooth, quantized, smoothed).astype(np.uint8)

    entropy_after = entropy_uint8(filtered_uint8)
    filtered_db = denormalize_spectrogram_from_filters(filtered_uint8, spec_min, spec_max)
    filtered_db = np.clip(filtered_db, DB_MIN, DB_MAX).astype(np.float32)
    return torch.from_numpy(filtered_db).unsqueeze(0), entropy_before, entropy_after


def filter_batch(x_db: torch.Tensor, config: FilterConfig) -> Tuple[torch.Tensor, List[float], List[float]]:
    device = x_db.device
    filtered_items: List[torch.Tensor] = []
    entropy_before: List[float] = []
    entropy_after: List[float] = []

    for sample in x_db.detach().cpu():
        filtered, before, after = filter_single(sample, config)
        filtered_items.append(filtered)
        entropy_before.append(before)
        entropy_after.append(after)

    return torch.stack(filtered_items, dim=0).to(device=device, dtype=x_db.dtype), entropy_before, entropy_after


@dataclass
class CachedBatch:
    x: torch.Tensor
    y: torch.Tensor
    x_adv: torch.Tensor
    pred_clean: torch.Tensor
    pred_adv: torch.Tensor


def cache_attack_batches(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    epsilon: float,
    log_interval: int,
) -> List[CachedBatch]:
    """Generate adversarial examples once and reuse them for all filter configs."""
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
            LOGGER.info("Cached attack batch %d/%d", batch_idx, len(test_loader))

    return cached


def evaluate_config(
    model: torch.nn.Module,
    cached_batches: Sequence[CachedBatch],
    config: FilterConfig,
    log_interval: int,
) -> CalibrationResult:
    model.eval()

    num_clean_total = 0
    num_clean_correct = 0
    num_clean_misclassified = 0
    num_eligible_for_attack = 0
    num_failures = 0
    num_successful_adversarial = 0
    clean_correct_total = 0
    adv_correct_total = 0
    adv_correct_on_eligible = 0
    filtered_adv_correct_total = 0
    tp = fp = fn = tn = rtp = 0

    entropy_clean_values: List[float] = []
    entropy_adv_values: List[float] = []
    entropy_filtered_clean_values: List[float] = []
    entropy_filtered_adv_values: List[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(cached_batches, start=1):
            x = batch.x
            y = batch.y
            x_adv = batch.x_adv
            pred_clean = batch.pred_clean
            pred_adv = batch.pred_adv

            clean_filtered, clean_entropy, clean_filtered_entropy = filter_batch(x, config)
            adv_filtered, adv_entropy, adv_filtered_entropy = filter_batch(x_adv, config)

            pred_clean_filtered = predict(model, clean_filtered)
            pred_adv_filtered = predict(model, adv_filtered)

            batch_size = x.size(0)
            num_clean_total += batch_size

            clean_correct_mask = pred_clean == y
            clean_correct_count = clean_correct_mask.sum().item()
            num_clean_correct += clean_correct_count
            num_clean_misclassified += batch_size - clean_correct_count
            num_eligible_for_attack += clean_correct_count

            clean_correct_total += clean_correct_count
            adv_correct_total += (pred_adv == y).sum().item()
            filtered_adv_correct_total += (pred_adv_filtered == y).sum().item()
            adv_correct_on_eligible += (clean_correct_mask & (pred_adv == y)).sum().item()

            benign_detected_as_adv = pred_clean != pred_clean_filtered
            fp += benign_detected_as_adv.sum().item()
            tn += (~benign_detected_as_adv).sum().item()

            attack_success = clean_correct_mask & (pred_adv != y)
            attack_failure = clean_correct_mask & (pred_adv == y)
            num_successful_adversarial += attack_success.sum().item()
            num_failures += attack_failure.sum().item()

            detected_adv = pred_adv != pred_adv_filtered
            tp_mask = attack_success & detected_adv
            fn_mask = attack_success & ~detected_adv
            tp += tp_mask.sum().item()
            fn += fn_mask.sum().item()
            rtp += (tp_mask & (pred_adv_filtered == y)).sum().item()

            entropy_clean_values.extend(clean_entropy)
            entropy_adv_values.extend(adv_entropy)
            entropy_filtered_clean_values.extend(clean_filtered_entropy)
            entropy_filtered_adv_values.extend(adv_filtered_entropy)

            if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(cached_batches):
                LOGGER.debug(
                    "%s - batch %d/%d - tp=%d fn=%d fp=%d rtp=%d",
                    config.name,
                    batch_idx,
                    len(cached_batches),
                    tp,
                    fn,
                    fp,
                    rtp,
                )

    recall = safe_div(tp, tp + fn)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2.0 * recall * precision, recall + precision)
    rtp_percent = 100.0 * safe_div(rtp, tp)
    false_positive_rate = safe_div(fp, fp + tn)

    metrics = PaperStyleMetrics(
        num_clean_total=num_clean_total,
        num_clean_correct=num_clean_correct,
        num_clean_misclassified=num_clean_misclassified,
        num_eligible_for_attack=num_eligible_for_attack,
        num_failures=num_failures,
        num_successful_adversarial=num_successful_adversarial,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        rtp=rtp,
        rtp_percent=rtp_percent,
        recall=100.0 * recall,
        precision=100.0 * precision,
        f1=100.0 * f1,
        false_positive_rate=100.0 * false_positive_rate,
        clean_accuracy=100.0 * safe_div(clean_correct_total, num_clean_total),
        adversarial_accuracy_all=100.0 * safe_div(adv_correct_total, num_clean_total),
        adversarial_accuracy_on_eligible=100.0 * safe_div(adv_correct_on_eligible, num_eligible_for_attack),
        filtered_adversarial_accuracy_all=100.0 * safe_div(filtered_adv_correct_total, num_clean_total),
        attack_success_rate_on_eligible=100.0 * safe_div(num_successful_adversarial, num_eligible_for_attack),
        attack_success_rate_all=100.0 * safe_div(num_successful_adversarial, num_clean_total),
    )

    return CalibrationResult(
        config_name=config.name,
        threshold_low=config.threshold_low,
        threshold_high=config.threshold_high,
        q_low=config.q_low,
        q_mid=config.q_mid,
        q_high=config.q_high,
        smoothing=config.smoothing,
        normalization=config.normalization,
        metrics=metrics,
        entropy_clean=entropy_stats(entropy_clean_values),
        entropy_adversarial=entropy_stats(entropy_adv_values),
        entropy_filtered_clean=entropy_stats(entropy_filtered_clean_values),
        entropy_filtered_adversarial=entropy_stats(entropy_filtered_adv_values),
    )


def result_to_row(result: CalibrationResult) -> Dict[str, object]:
    metrics = result.metrics
    row: Dict[str, object] = {
        "config_name": result.config_name,
        "threshold_low": result.threshold_low,
        "threshold_high": result.threshold_high,
        "q_low": result.q_low,
        "q_mid": result.q_mid,
        "q_high": result.q_high,
        "smoothing": result.smoothing,
        "normalization": result.normalization,
        "num_clean_total": metrics.num_clean_total,
        "num_clean_correct": metrics.num_clean_correct,
        "num_eligible_for_attack": metrics.num_eligible_for_attack,
        "num_failures": metrics.num_failures,
        "num_successful_adversarial": metrics.num_successful_adversarial,
        "tp": metrics.tp,
        "fn": metrics.fn,
        "fp": metrics.fp,
        "tn": metrics.tn,
        "rtp": metrics.rtp,
        "rtp_percent": metrics.rtp_percent,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "f1": metrics.f1,
        "false_positive_rate": metrics.false_positive_rate,
        "clean_accuracy": metrics.clean_accuracy,
        "adversarial_accuracy_all": metrics.adversarial_accuracy_all,
        "filtered_adversarial_accuracy_all": metrics.filtered_adversarial_accuracy_all,
        "attack_success_rate_on_eligible": metrics.attack_success_rate_on_eligible,
    }

    for prefix, stats in [
        ("entropy_clean", result.entropy_clean),
        ("entropy_adversarial", result.entropy_adversarial),
        ("entropy_filtered_clean", result.entropy_filtered_clean),
        ("entropy_filtered_adversarial", result.entropy_filtered_adversarial),
    ]:
        stats_dict = asdict(stats)
        for key, value in stats_dict.items():
            row[f"{prefix}_{key}"] = value

    return row


def save_results(results: Sequence[CalibrationResult], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [result_to_row(result) for result in results]
    if not rows:
        raise RuntimeError("No calibration results to save")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = [
        {
            "config": {
                "name": result.config_name,
                "threshold_low": result.threshold_low,
                "threshold_high": result.threshold_high,
                "q_low": result.q_low,
                "q_mid": result.q_mid,
                "q_high": result.q_high,
                "smoothing": result.smoothing,
                "normalization": result.normalization,
            },
            "metrics": asdict(result.metrics),
            "entropy": {
                "clean": asdict(result.entropy_clean),
                "adversarial": asdict(result.entropy_adversarial),
                "filtered_clean": asdict(result.entropy_filtered_clean),
                "filtered_adversarial": asdict(result.entropy_filtered_adversarial),
            },
        }
        for result in results
    ]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    LOGGER.info("Saved CSV results to %s", csv_path)
    LOGGER.info("Saved JSON results to %s", json_path)


def print_top_results(results: Sequence[CalibrationResult], top_k: int) -> None:
    sorted_by_f1 = sorted(results, key=lambda item: item.metrics.f1, reverse=True)
    sorted_by_recall_low_fpr = sorted(
        results,
        key=lambda item: (item.metrics.recall, -item.metrics.false_positive_rate, item.metrics.f1),
        reverse=True,
    )

    def print_block(title: str, selected: Sequence[CalibrationResult]) -> None:
        print(f"\n=== {title} ===")
        print("rank,f1,recall,precision,fpr,rtp_percent,tp,fn,fp,tn,config")
        for rank, result in enumerate(selected, start=1):
            metrics = result.metrics
            print(
                f"{rank},"
                f"{metrics.f1:.2f},"
                f"{metrics.recall:.2f},"
                f"{metrics.precision:.2f},"
                f"{metrics.false_positive_rate:.2f},"
                f"{metrics.rtp_percent:.2f},"
                f"{metrics.tp},"
                f"{metrics.fn},"
                f"{metrics.fp},"
                f"{metrics.tn},"
                f"{result.config_name}"
            )

    print_block(f"Top {top_k} by F1", sorted_by_f1[:top_k])
    print_block(f"Top {top_k} by recall", sorted_by_recall_low_fpr[:top_k])

    baseline = next(
        (
            result
            for result in results
            if result.threshold_low == 4.0
            and result.threshold_high == 5.0
            and result.q_low == 2
            and result.q_mid == 4
            and result.q_high == 6
            and result.smoothing == "cross5"
        ),
        None,
    )
    if baseline is not None:
        print("\n=== Baseline-like DeepDetector image setting found ===")
        print("config,f1,recall,precision,fpr,rtp_percent")
        print(
            f"{baseline.config_name},"
            f"{baseline.metrics.f1:.2f},"
            f"{baseline.metrics.recall:.2f},"
            f"{baseline.metrics.precision:.2f},"
            f"{baseline.metrics.false_positive_rate:.2f},"
            f"{baseline.metrics.rtp_percent:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep audio DeepDetector filter parameters.")
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
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout used by the checkpoint architecture.")
    parser.add_argument(
        "--thresholds",
        default="3,4;4,5;5,6;6,7",
        help="Semicolon-separated threshold pairs, e.g. '3,4;4,5'.",
    )
    parser.add_argument(
        "--quantizations",
        default="2,4,6;4,6,8;6,8,16",
        help="Semicolon-separated quantization triples, e.g. '2,4,6;4,6,8'.",
    )
    parser.add_argument(
        "--smoothings",
        default="none,cross3,cross5,mean3",
        help="Comma-separated smoothing modes: none,cross3,cross5,mean3.",
    )
    parser.add_argument(
        "--normalizations",
        default="sample,global",
        help="Comma-separated normalization modes: sample,global.",
    )
    parser.add_argument("--csv-out", default="artifacts/audio_poc/filter_calibration.csv")
    parser.add_argument("--json-out", default="artifacts/audio_poc/filter_calibration.json")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top configs to print.")
    parser.add_argument("--log-interval", type=int, default=5, help="Batch/config log interval.")
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

    device = resolve_device(args.device)
    log_environment(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the model first with audio_adversarial_poc.py."
        )

    class_names = [item.strip() for item in args.classes.split(",") if item.strip()]
    model = SpeechCNN(num_classes=len(class_names), dropout=args.dropout).to(device)
    payload = safe_torch_load(checkpoint_path, device)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    LOGGER.info("Loaded checkpoint: %s", checkpoint_path)

    # Reuse make_loaders from the main PoC. We do not need training data here.
    args.max_train = 1
    args.epochs = 0
    args.no_download = False
    _, test_loader = make_loaders(args, device=device)

    configs = build_filter_configs(args)
    LOGGER.info("Running %d filter configurations", len(configs))

    cached_batches = cache_attack_batches(
        model=model,
        test_loader=test_loader,
        device=device,
        epsilon=args.epsilon,
        log_interval=max(args.log_interval, 1),
    )

    results: List[CalibrationResult] = []
    start = time.time()
    for idx, config in enumerate(configs, start=1):
        config_start = time.time()
        LOGGER.info("[%d/%d] Evaluating %s", idx, len(configs), config.name)
        result = evaluate_config(
            model=model,
            cached_batches=cached_batches,
            config=config,
            log_interval=max(args.log_interval, 1),
        )
        results.append(result)
        LOGGER.info(
            "[%d/%d] Done %s - f1=%.2f recall=%.2f precision=%.2f fpr=%.2f rtp=%.2f elapsed=%.1fs",
            idx,
            len(configs),
            config.name,
            result.metrics.f1,
            result.metrics.recall,
            result.metrics.precision,
            result.metrics.false_positive_rate,
            result.metrics.rtp_percent,
            time.time() - config_start,
        )

    save_results(results, Path(args.csv_out), Path(args.json_out))
    print_top_results(results, top_k=args.top_k)
    LOGGER.info("Calibration sweep finished in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
