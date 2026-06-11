#!/usr/bin/env python3
"""Audio filter V3 sweep: spectrogram-aware smoothing geometry.

This experiment starts from the best V2 filter:

    global normalization + fixed q=5 + cross5 smoothing + replace

and tests whether spectrogram-aware smoothing can improve it. Unlike images,
log-Mel spectrogram axes have different meanings:

    X axis: time frames
    Y axis: Mel frequency bins

So this script evaluates anisotropic kernels such as time-only smoothing
(1x3, 1x5, 1x7) and frequency-only smoothing (3x1, 5x1, 7x1). It also tests
weighted combinations between quantized and smoothed maps and optional dB noise
gates.

The script reuses the trained SpeechResCNN checkpoint and FGSM generation logic
from audio_adversarial_poc.py.

Example:
    python scripts/dev/audio_filter_v3_geometry_sweep_poc.py \
        --device cuda \
        --data-dir data/raw/speech_commands \
        --checkpoint artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt \
        --max-test 1200 \
        --batch-size 64 \
        --epsilon 2.0 \
        --quantizations "4,5,6" \
        --smoothings "none,time3,time5,time7,freq3,freq5,freq7,cross5,mean3" \
        --combine-modes "replace,weighted" \
        --weighted-alphas "0.25,0.5,0.75" \
        --noise-gates "none,-75,-70,-65,-60"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from audio_adversarial_poc import (
    DB_MAX,
    DB_MIN,
    DEFAULT_CLASSES,
    PaperStyleMetrics,
    SpeechCNN,
    SpeechCommandsSpectrogramDataset,
    configure_backend,
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

LOGGER = logging.getLogger("audio_filter_v3_geometry_sweep_poc")


@dataclass(frozen=True)
class GeometryFilterConfig:
    normalization: str
    quantization: int
    smoothing: str
    combine_mode: str
    weighted_alpha: Optional[float]
    noise_gate_db: Optional[float]

    @property
    def name(self) -> str:
        gate = "none" if self.noise_gate_db is None else f"{self.noise_gate_db:g}db"
        alpha = "none" if self.weighted_alpha is None else f"{self.weighted_alpha:g}"
        return (
            f"v3_q_{self.quantization}__"
            f"smooth_{self.smoothing}__"
            f"combine_{self.combine_mode}__"
            f"alpha_{alpha}__"
            f"gate_{gate}__"
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


def parse_csv_floats(value: str) -> List[float]:
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one float value")
    return items


def parse_noise_gates(value: str) -> List[Optional[float]]:
    gates: List[Optional[float]] = []
    for item in parse_csv_options(value):
        if item.lower() == "none":
            gates.append(None)
        else:
            gate = float(item)
            if gate < DB_MIN or gate > DB_MAX:
                raise ValueError(f"Noise gate must be within [{DB_MIN}, {DB_MAX}], got {gate}")
            gates.append(gate)
    return gates


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


def smoothing_kernel(mode: str) -> np.ndarray:
    """Return spectrogram smoothing kernel.

    timeK means 1xK over the time axis.
    freqK means Kx1 over the frequency axis.
    """
    if mode == "none":
        return np.asarray([[1.0]], dtype=np.float32)

    if mode.startswith("time"):
        size = int(mode.replace("time", ""))
        return np.ones((1, size), dtype=np.float32) / float(size)

    if mode.startswith("freq"):
        size = int(mode.replace("freq", ""))
        return np.ones((size, 1), dtype=np.float32) / float(size)

    if mode == "mean3":
        return np.ones((3, 3), dtype=np.float32) / 9.0

    if mode == "mean5":
        return np.ones((5, 5), dtype=np.float32) / 25.0

    if mode == "cross3":
        kernel = np.zeros((3, 3), dtype=np.float32)
        kernel[1, :] = 1.0
        kernel[:, 1] = 1.0
        return kernel / kernel.sum()

    if mode == "cross5":
        kernel = np.zeros((5, 5), dtype=np.float32)
        kernel[2, :] = 1.0
        kernel[:, 2] = 1.0
        return kernel / kernel.sum()

    raise ValueError(f"Unknown smoothing mode: {mode}")


def smooth_uint8(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image

    kernel = smoothing_kernel(mode)
    height, width = image.shape
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(image.astype(np.float32), ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            weight = kernel[i, j]
            if weight == 0:
                continue
            output += weight * padded[i : i + height, j : j + width]

    return np.clip(np.round(output), 0, 255).astype(np.uint8)


def apply_noise_gate_db(spec_db: np.ndarray, gate_db: Optional[float]) -> np.ndarray:
    if gate_db is None:
        return spec_db
    gated = spec_db.copy()
    gated[gated < gate_db] = DB_MIN
    return gated


def build_configs(args: argparse.Namespace) -> List[GeometryFilterConfig]:
    quantizations = parse_csv_ints(args.quantizations)
    smoothings = parse_csv_options(args.smoothings)
    combine_modes = parse_csv_options(args.combine_modes)
    alphas = parse_csv_floats(args.weighted_alphas)
    noise_gates = parse_noise_gates(args.noise_gates)

    valid_smoothings = {
        "none",
        "time3",
        "time5",
        "time7",
        "freq3",
        "freq5",
        "freq7",
        "cross3",
        "cross5",
        "mean3",
        "mean5",
    }
    valid_combine_modes = {"quantize_only", "replace", "weighted"}

    invalid_smoothings = sorted(set(smoothings) - valid_smoothings)
    invalid_combine_modes = sorted(set(combine_modes) - valid_combine_modes)
    if invalid_smoothings:
        raise ValueError(f"Invalid smoothing modes: {invalid_smoothings}")
    if invalid_combine_modes:
        raise ValueError(f"Invalid combine modes: {invalid_combine_modes}")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("Weighted alphas must be within [0, 1]")

    configs: List[GeometryFilterConfig] = []
    seen = set()

    for q in quantizations:
        for smoothing in smoothings:
            for gate_db in noise_gates:
                for combine_mode in combine_modes:
                    if smoothing == "none":
                        # Smoothing is absent, so all smoothing-based combine modes collapse to quantize_only.
                        key = (q, smoothing, "quantize_only", None, gate_db)
                        if key not in seen:
                            seen.add(key)
                            configs.append(
                                GeometryFilterConfig(
                                    normalization="global",
                                    quantization=q,
                                    smoothing=smoothing,
                                    combine_mode="quantize_only",
                                    weighted_alpha=None,
                                    noise_gate_db=gate_db,
                                )
                            )
                        continue

                    if combine_mode == "quantize_only":
                        key = (q, smoothing, combine_mode, None, gate_db)
                        if key in seen:
                            continue
                        seen.add(key)
                        configs.append(
                            GeometryFilterConfig(
                                normalization="global",
                                quantization=q,
                                smoothing=smoothing,
                                combine_mode=combine_mode,
                                weighted_alpha=None,
                                noise_gate_db=gate_db,
                            )
                        )
                    elif combine_mode == "replace":
                        key = (q, smoothing, combine_mode, None, gate_db)
                        if key in seen:
                            continue
                        seen.add(key)
                        configs.append(
                            GeometryFilterConfig(
                                normalization="global",
                                quantization=q,
                                smoothing=smoothing,
                                combine_mode=combine_mode,
                                weighted_alpha=None,
                                noise_gate_db=gate_db,
                            )
                        )
                    else:
                        for alpha in alphas:
                            key = (q, smoothing, combine_mode, alpha, gate_db)
                            if key in seen:
                                continue
                            seen.add(key)
                            configs.append(
                                GeometryFilterConfig(
                                    normalization="global",
                                    quantization=q,
                                    smoothing=smoothing,
                                    combine_mode=combine_mode,
                                    weighted_alpha=alpha,
                                    noise_gate_db=gate_db,
                                )
                            )

    return configs


def filter_single(spec_db: torch.Tensor, config: GeometryFilterConfig) -> Tuple[torch.Tensor, float, float]:
    if spec_db.ndim != 3 or spec_db.size(0) != 1:
        raise ValueError(f"Expected [1, n_mels, time], got {tuple(spec_db.shape)}")

    spec_np = spec_db.squeeze(0).detach().cpu().numpy().astype(np.float32)
    spec_np = np.clip(spec_np, DB_MIN, DB_MAX)
    spec_np = apply_noise_gate_db(spec_np, config.noise_gate_db)

    spec_uint8, spec_min, spec_max = normalize_spectrogram_for_filters(
        spec_np,
        normalization=config.normalization,
    )
    entropy_before = entropy_uint8(spec_uint8)

    quantized = quantize_uint8(spec_uint8, intervals=config.quantization)
    smoothed = smooth_uint8(quantized, config.smoothing)

    if config.combine_mode == "quantize_only" or config.smoothing == "none":
        filtered_uint8 = quantized
    elif config.combine_mode == "replace":
        filtered_uint8 = smoothed
    elif config.combine_mode == "weighted":
        assert config.weighted_alpha is not None
        alpha = config.weighted_alpha
        filtered_float = (1.0 - alpha) * quantized.astype(np.float32) + alpha * smoothed.astype(np.float32)
        filtered_uint8 = np.clip(np.round(filtered_float), 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown combine mode: {config.combine_mode}")

    entropy_after = entropy_uint8(filtered_uint8)
    filtered_db = denormalize_spectrogram_from_filters(filtered_uint8, spec_min, spec_max)
    filtered_db = np.clip(filtered_db, DB_MIN, DB_MAX).astype(np.float32)
    return torch.from_numpy(filtered_db).unsqueeze(0), entropy_before, entropy_after


def filter_batch(
    x_db: torch.Tensor,
    config: GeometryFilterConfig,
) -> Tuple[torch.Tensor, List[float], List[float]]:
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


def compute_batch_counts(
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


def evaluate_config(
    model: torch.nn.Module,
    cached_batches: Sequence[CachedBatch],
    config: GeometryFilterConfig,
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
            clean_filtered, clean_entropy, clean_filtered_entropy = filter_batch(batch.x, config)
            adv_filtered, adv_entropy, adv_filtered_entropy = filter_batch(batch.x_adv, config)

            pred_clean_filtered = predict(model, clean_filtered)
            pred_adv_filtered = predict(model, adv_filtered)

            counts = compute_batch_counts(
                y=batch.y,
                pred_clean=batch.pred_clean,
                pred_adv=batch.pred_adv,
                pred_clean_filtered=pred_clean_filtered,
                pred_adv_filtered=pred_adv_filtered,
            )
            for key, value in counts.items():
                totals[key] += value

            entropy_clean_values.extend(clean_entropy)
            entropy_adv_values.extend(adv_entropy)
            entropy_filtered_clean_values.extend(clean_filtered_entropy)
            entropy_filtered_adv_values.extend(adv_filtered_entropy)

            if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(cached_batches):
                LOGGER.debug("%s - batch %d/%d", config.name, batch_idx, len(cached_batches))

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
        "config_name": config.name,
        "normalization": config.normalization,
        "quantization": config.quantization,
        "smoothing": config.smoothing,
        "combine_mode": config.combine_mode,
        "weighted_alpha": config.weighted_alpha if config.weighted_alpha is not None else "none",
        "noise_gate_db": config.noise_gate_db if config.noise_gate_db is not None else "none",
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


def print_rankings(rows: Sequence[Dict[str, object]], top_k: int) -> None:
    by_f1 = sorted(rows, key=lambda row: float(row["f1"]), reverse=True)
    by_constraint = sorted(
        [
            row
            for row in rows
            if float(row["false_positive_rate"]) <= 3.0
            and float(row["precision"]) >= 88.0
            and float(row["rtp_percent"]) >= 85.0
        ],
        key=lambda row: (float(row["f1"]), float(row["recall"])),
        reverse=True,
    )
    by_low_fpr = sorted(rows, key=lambda row: (float(row["false_positive_rate"]), -float(row["f1"])))

    def print_block(title: str, selected: Sequence[Dict[str, object]]) -> None:
        print(f"\n=== {title} ===")
        print("rank,f1,recall,precision,fpr,rtp_percent,tp,fn,fp,tn,q,smoothing,combine,alpha,gate,config")
        for rank, row in enumerate(selected, start=1):
            print(
                f"{rank},"
                f"{row['f1']:.2f},"
                f"{row['recall']:.2f},"
                f"{row['precision']:.2f},"
                f"{row['false_positive_rate']:.2f},"
                f"{row['rtp_percent']:.2f},"
                f"{row['tp']},"
                f"{row['fn']},"
                f"{row['fp']},"
                f"{row['tn']},"
                f"{row['quantization']},"
                f"{row['smoothing']},"
                f"{row['combine_mode']},"
                f"{row['weighted_alpha']},"
                f"{row['noise_gate_db']},"
                f"{row['config_name']}"
            )

    print_block(f"Top {top_k} by F1", by_f1[:top_k])
    print_block(f"Top {top_k} under practical constraints", by_constraint[:top_k])
    print_block(f"Top {top_k} lowest FPR", by_low_fpr[:top_k])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep spectrogram-aware filter geometry for audio adversarial detection.")
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

    parser.add_argument("--quantizations", default="4,5,6")
    parser.add_argument("--smoothings", default="none,time3,time5,time7,freq3,freq5,freq7,cross5,mean3")
    parser.add_argument("--combine-modes", default="replace,weighted")
    parser.add_argument("--weighted-alphas", default="0.25,0.5,0.75")
    parser.add_argument("--noise-gates", default="none,-75,-70,-65,-60")

    parser.add_argument("--csv-out", default="artifacts/audio_poc/filter_v3_geometry_sweep.csv")
    parser.add_argument("--json-out", default="artifacts/audio_poc/filter_v3_geometry_sweep.json")
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

    configs = build_configs(args)
    LOGGER.info("Running %d V3 filter configurations", len(configs))

    rows: List[Dict[str, object]] = []
    start = time.time()
    for idx, config in enumerate(configs, start=1):
        config_start = time.time()
        LOGGER.info("[%d/%d] Evaluating %s", idx, len(configs), config.name)
        row = evaluate_config(
            model=model,
            cached_batches=cached_batches,
            config=config,
            log_interval=max(args.log_interval, 1),
        )
        rows.append(row)
        LOGGER.info(
            "[%d/%d] Done %s - f1=%.2f recall=%.2f precision=%.2f fpr=%.2f rtp=%.2f elapsed=%.1fs",
            idx,
            len(configs),
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
            "experiment": "spectrogram_geometry_filter_v3",
            "db_min": DB_MIN,
            "db_max": DB_MAX,
            "baseline_to_beat": {
                "name": "fixed_q_5__smooth_cross5__combine_replace__norm_global",
                "recall": 56.85,
                "precision": 90.03,
                "f1": 69.69,
                "false_positive_rate": 2.83,
                "rtp_percent": 89.25,
            },
        },
    )
    print_rankings(rows, top_k=args.top_k)
    LOGGER.info("Filter V3 sweep finished in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
