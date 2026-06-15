#!/usr/bin/env python3
"""Audio adversarial attack suite for the best filter profiles.

This script extends the audio PoC beyond FGSM and evaluates stronger/different
attack families against the best filters found in the previous sweeps.

Supported attacks:
    - fgsm: fast gradient sign method in log-Mel dB space.
    - deepfool: untargeted DeepFool-style iterative attack in log-Mel dB space.
    - cw_l2: untargeted CW-L2-style projected optimization attack.

Supported filter profiles:
    - v4_fixed_high_f1:
        q=7 + freq11 + replace + global normalization.
    - v4_entropy_balanced:
        entropy thresholds 6.1/6.5, q=6/7/8, smoothing=freq7/freq9/freq11.
    - v2_q5_cross5:
        q=5 + cross5 + replace + global normalization.

Notes:
    CW-L2 is much slower than FGSM and DeepFool. Start with --max-test 200 or
    --max-test 300 before running a larger evaluation.

Example FGSM + DeepFool:
    python scripts/dev/audio_attack_suite_poc.py \
        --device cuda \
        --data-dir data/raw/speech_commands \
        --checkpoint artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt \
        --max-test 1200 \
        --batch-size 32 \
        --attacks "fgsm,deepfool"

Example CW-L2 smoke test:
    python scripts/dev/audio_attack_suite_poc.py \
        --device cuda \
        --data-dir data/raw/speech_commands \
        --checkpoint artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt \
        --max-test 300 \
        --batch-size 16 \
        --attacks "cw_l2" \
        --cw-max-iterations 200 \
        --cw-binary-search-steps 3
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from audio_adversarial_poc import (
    DB_MAX,
    DB_MIN,
    DEFAULT_CLASSES,
    PaperStyleMetrics,
    SpeechCNN,
    SpeechCommandsSpectrogramDataset,
    configure_backend,
    fgsm_attack,
    log_environment,
    predict,
    resolve_device,
    safe_div,
    safe_torch_load,
    set_seed,
)

LOGGER = logging.getLogger("audio_attack_suite_poc")


@dataclass(frozen=True)
class FilterProfile:
    name: str
    mode: str
    quantization: int
    smoothing: str
    entropy_low: Optional[float] = None
    entropy_high: Optional[float] = None
    q_low: Optional[int] = None
    q_mid: Optional[int] = None
    q_high: Optional[int] = None
    smoothing_low: Optional[str] = None
    smoothing_mid: Optional[str] = None
    smoothing_high: Optional[str] = None


@dataclass
class AttackBatch:
    x: torch.Tensor
    y: torch.Tensor
    pred_clean: torch.Tensor
    pred_clean_filtered: Dict[str, torch.Tensor]
    x_adv: torch.Tensor
    pred_adv: torch.Tensor
    pred_adv_filtered: Dict[str, torch.Tensor]


FILTER_PROFILES: Dict[str, FilterProfile] = {
    "v4_fixed_high_f1": FilterProfile(
        name="v4_fixed_high_f1",
        mode="fixed",
        quantization=7,
        smoothing="freq11",
    ),
    "v4_entropy_balanced": FilterProfile(
        name="v4_entropy_balanced",
        mode="entropy",
        quantization=0,
        smoothing="adaptive",
        entropy_low=6.1,
        entropy_high=6.5,
        q_low=6,
        q_mid=7,
        q_high=8,
        smoothing_low="freq7",
        smoothing_mid="freq9",
        smoothing_high="freq11",
    ),
    "v2_q5_cross5": FilterProfile(
        name="v2_q5_cross5",
        mode="fixed",
        quantization=5,
        smoothing="cross5",
    ),
}


def parse_csv_options(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def entropy_uint8(image: np.ndarray) -> float:
    hist = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    probs = hist / max(float(hist.sum()), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def spec_to_uint8_global(spec_db: np.ndarray) -> np.ndarray:
    spec = np.clip(spec_db.astype(np.float32), DB_MIN, DB_MAX)
    normalized = (spec - DB_MIN) / (DB_MAX - DB_MIN)
    return np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)


def uint8_to_spec_global(image: np.ndarray) -> np.ndarray:
    spec = image.astype(np.float32) / 255.0
    return np.clip(spec * (DB_MAX - DB_MIN) + DB_MIN, DB_MIN, DB_MAX).astype(np.float32)


def quantize_uint8(image: np.ndarray, intervals: int) -> np.ndarray:
    if intervals <= 1:
        return np.zeros_like(image, dtype=np.uint8)
    step = 255.0 / float(intervals - 1)
    quantized = np.round(image.astype(np.float32) / step) * step
    return np.clip(quantized, 0, 255).astype(np.uint8)


def smoothing_kernel(mode: str) -> np.ndarray:
    if mode == "none":
        return np.asarray([[1.0]], dtype=np.float32)

    if mode.startswith("freq"):
        size = int(mode.replace("freq", ""))
        return np.ones((size, 1), dtype=np.float32) / float(size)

    if mode.startswith("time"):
        size = int(mode.replace("time", ""))
        return np.ones((1, size), dtype=np.float32) / float(size)

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


def choose_profile_branch(entropy: float, profile: FilterProfile) -> Tuple[int, str]:
    if profile.mode == "fixed":
        return profile.quantization, profile.smoothing

    if profile.entropy_low is None or profile.entropy_high is None:
        raise ValueError(f"Entropy profile is missing thresholds: {profile}")

    assert profile.q_low is not None and profile.q_mid is not None and profile.q_high is not None
    assert profile.smoothing_low is not None and profile.smoothing_mid is not None and profile.smoothing_high is not None

    if entropy < profile.entropy_low:
        return profile.q_low, profile.smoothing_low
    if entropy < profile.entropy_high:
        return profile.q_mid, profile.smoothing_mid
    return profile.q_high, profile.smoothing_high


def apply_filter_single(spec_db: torch.Tensor, profile: FilterProfile) -> torch.Tensor:
    if spec_db.ndim != 3 or spec_db.size(0) != 1:
        raise ValueError(f"Expected [1, n_mels, time], got {tuple(spec_db.shape)}")

    spec_np = spec_db.squeeze(0).detach().cpu().numpy().astype(np.float32)
    image = spec_to_uint8_global(spec_np)
    entropy = entropy_uint8(image)
    q, smoothing = choose_profile_branch(entropy, profile)
    quantized = quantize_uint8(image, intervals=q)
    filtered = smooth_uint8(quantized, mode=smoothing)
    filtered_db = uint8_to_spec_global(filtered)
    return torch.from_numpy(filtered_db).unsqueeze(0)


def apply_filter_batch(x_db: torch.Tensor, profile: FilterProfile) -> torch.Tensor:
    device = x_db.device
    filtered = [apply_filter_single(sample, profile) for sample in x_db.detach().cpu()]
    return torch.stack(filtered, dim=0).to(device=device, dtype=x_db.dtype)


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


def deepfool_attack_single(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_iter: int,
    overshoot: float,
    clip_min: float = DB_MIN,
    clip_max: float = DB_MAX,
) -> torch.Tensor:
    """Untargeted DeepFool-style attack for one sample.

    The implementation operates in log-Mel dB space and uses all output classes.
    """
    was_training = model.training
    model.eval()

    x_adv = x.detach().clone()
    y_int = int(y.item())

    for _ in range(max_iter):
        x_adv = x_adv.detach().clone().requires_grad_(True)
        logits = model(x_adv)
        current_pred = int(logits.argmax(dim=1).item())
        if current_pred != y_int:
            break

        logit_y = logits[0, y_int]
        grad_y = torch.autograd.grad(logit_y, x_adv, retain_graph=True)[0]

        best_distance = float("inf")
        best_perturbation: Optional[torch.Tensor] = None
        num_classes = logits.size(1)

        for class_idx in range(num_classes):
            if class_idx == y_int:
                continue

            logit_k = logits[0, class_idx]
            grad_k = torch.autograd.grad(logit_k, x_adv, retain_graph=True)[0]
            w = grad_k - grad_y
            f = logit_k - logit_y
            w_norm = torch.norm(w.reshape(-1), p=2) + 1e-8
            distance = abs(float(f.item())) / float(w_norm.item())
            perturbation = ((abs(f) + 1e-4) / (w_norm**2)) * w

            if distance < best_distance:
                best_distance = distance
                best_perturbation = perturbation.detach()

        if best_perturbation is None:
            break

        x_adv = torch.clamp(x_adv.detach() + (1.0 + overshoot) * best_perturbation, clip_min, clip_max)

    if was_training:
        model.train()

    return x_adv.detach()


def deepfool_attack_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    clean_correct_mask: torch.Tensor,
    max_iter: int,
    overshoot: float,
) -> torch.Tensor:
    adv_items: List[torch.Tensor] = []
    for idx in range(x.size(0)):
        sample = x[idx : idx + 1]
        label = y[idx : idx + 1]
        if not bool(clean_correct_mask[idx].item()):
            adv_items.append(sample.detach())
            continue
        adv_items.append(
            deepfool_attack_single(
                model=model,
                x=sample,
                y=label,
                max_iter=max_iter,
                overshoot=overshoot,
            )
        )
    return torch.cat(adv_items, dim=0).to(device=x.device, dtype=x.dtype)


def cw_l2_attack_single(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_iterations: int,
    binary_search_steps: int,
    initial_const: float,
    learning_rate: float,
    confidence: float,
    abort_early: bool,
    clip_min: float = DB_MIN,
    clip_max: float = DB_MAX,
) -> torch.Tensor:
    """Untargeted CW-L2-style projected optimization attack for one sample.

    This is a practical PoC implementation. It optimizes an additive delta in dB
    space with projection to [DB_MIN, DB_MAX], rather than the original tanh-space
    parameterization.
    """
    was_training = model.training
    model.eval()

    y_int = int(y.item())
    best_adv = x.detach().clone()
    best_l2 = float("inf")
    best_success = False

    lower_bound = 0.0
    upper_bound = float("inf")
    const = initial_const

    for _ in range(binary_search_steps):
        delta = torch.zeros_like(x, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=learning_rate)
        prev_loss: Optional[float] = None

        for iteration in range(max_iterations):
            adv = torch.clamp(x + delta, clip_min, clip_max)
            logits = model(adv)
            real = logits[0, y_int]
            other_logits = logits.clone()
            other_logits[0, y_int] = -1e9
            other = other_logits.max(dim=1).values[0]

            # Untargeted CW objective: make some other class exceed the true class.
            attack_loss = torch.clamp(real - other + confidence, min=0.0)
            l2 = torch.sum((adv - x) ** 2)
            loss = l2 + const * attack_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pred = int(logits.argmax(dim=1).item())
            success = pred != y_int
            l2_value = float(l2.detach().item())
            if success and l2_value < best_l2:
                best_l2 = l2_value
                best_adv = adv.detach().clone()
                best_success = True

            if abort_early and iteration % max(max_iterations // 10, 1) == 0:
                loss_value = float(loss.detach().item())
                if prev_loss is not None and loss_value > prev_loss * 0.999:
                    break
                prev_loss = loss_value

        if best_success:
            upper_bound = min(upper_bound, const)
            const = (lower_bound + upper_bound) / 2.0
        else:
            lower_bound = max(lower_bound, const)
            const = const * 10.0 if upper_bound == float("inf") else (lower_bound + upper_bound) / 2.0

    if was_training:
        model.train()

    return best_adv.detach()


def cw_l2_attack_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    clean_correct_mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    adv_items: List[torch.Tensor] = []
    for idx in range(x.size(0)):
        sample = x[idx : idx + 1]
        label = y[idx : idx + 1]
        if not bool(clean_correct_mask[idx].item()):
            adv_items.append(sample.detach())
            continue
        adv_items.append(
            cw_l2_attack_single(
                model=model,
                x=sample,
                y=label,
                max_iterations=args.cw_max_iterations,
                binary_search_steps=args.cw_binary_search_steps,
                initial_const=args.cw_initial_const,
                learning_rate=args.cw_learning_rate,
                confidence=args.cw_confidence,
                abort_early=args.cw_abort_early,
            )
        )
    return torch.cat(adv_items, dim=0).to(device=x.device, dtype=x.dtype)


def generate_attack_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    pred_clean: torch.Tensor,
    attack_name: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    clean_correct_mask = pred_clean == y

    if attack_name == "fgsm":
        return fgsm_attack(model, x, y, epsilon=args.fgsm_epsilon)

    if attack_name == "deepfool":
        return deepfool_attack_batch(
            model=model,
            x=x,
            y=y,
            clean_correct_mask=clean_correct_mask,
            max_iter=args.deepfool_max_iter,
            overshoot=args.deepfool_overshoot,
        )

    if attack_name == "cw_l2":
        return cw_l2_attack_batch(
            model=model,
            x=x,
            y=y,
            clean_correct_mask=clean_correct_mask,
            args=args,
        )

    raise ValueError(f"Unsupported attack: {attack_name}")


def compute_metrics_from_batches(
    batches: Sequence[AttackBatch],
    filter_profile: str,
) -> PaperStyleMetrics:
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

    for batch in batches:
        y = batch.y
        pred_clean = batch.pred_clean
        pred_adv = batch.pred_adv
        pred_clean_filtered = batch.pred_clean_filtered[filter_profile]
        pred_adv_filtered = batch.pred_adv_filtered[filter_profile]

        clean_correct_mask = pred_clean == y
        benign_detected_as_adv = pred_clean != pred_clean_filtered
        attack_success = clean_correct_mask & (pred_adv != y)
        attack_failure = clean_correct_mask & (pred_adv == y)
        detected_adv = pred_adv != pred_adv_filtered
        tp_mask = attack_success & detected_adv
        fn_mask = attack_success & ~detected_adv

        totals["num_clean_total"] += int(y.numel())
        totals["num_clean_correct"] += int(clean_correct_mask.sum().item())
        totals["num_clean_misclassified"] += int((~clean_correct_mask).sum().item())
        totals["num_eligible_for_attack"] += int(clean_correct_mask.sum().item())
        totals["num_failures"] += int(attack_failure.sum().item())
        totals["num_successful_adversarial"] += int(attack_success.sum().item())
        totals["clean_correct_total"] += int(clean_correct_mask.sum().item())
        totals["adv_correct_total"] += int((pred_adv == y).sum().item())
        totals["adv_correct_on_eligible"] += int((clean_correct_mask & (pred_adv == y)).sum().item())
        totals["filtered_adv_correct_total"] += int((pred_adv_filtered == y).sum().item())
        totals["tp"] += int(tp_mask.sum().item())
        totals["fn"] += int(fn_mask.sum().item())
        totals["fp"] += int(benign_detected_as_adv.sum().item())
        totals["tn"] += int((~benign_detected_as_adv).sum().item())
        totals["rtp"] += int((tp_mask & (pred_adv_filtered == y)).sum().item())

    recall = safe_div(totals["tp"], totals["tp"] + totals["fn"])
    precision = safe_div(totals["tp"], totals["tp"] + totals["fp"])
    f1 = safe_div(2.0 * recall * precision, recall + precision)
    rtp_percent = 100.0 * safe_div(totals["rtp"], totals["tp"])
    false_positive_rate = safe_div(totals["fp"], totals["fp"] + totals["tn"])

    return PaperStyleMetrics(
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


def evaluate_attack(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    attack_name: str,
    profiles: Dict[str, FilterProfile],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    model.eval()
    attack_batches: List[AttackBatch] = []
    start = time.time()

    LOGGER.info("Evaluating attack=%s over %d batches", attack_name, len(test_loader))
    for batch_idx, (x, y) in enumerate(test_loader, start=1):
        batch_start = time.time()
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")

        with torch.no_grad():
            pred_clean = predict(model, x)

        x_adv = generate_attack_batch(
            model=model,
            x=x,
            y=y,
            pred_clean=pred_clean,
            attack_name=attack_name,
            args=args,
        )

        with torch.no_grad():
            pred_adv = predict(model, x_adv)
            pred_clean_filtered: Dict[str, torch.Tensor] = {}
            pred_adv_filtered: Dict[str, torch.Tensor] = {}

            for profile_name, profile in profiles.items():
                clean_filtered = apply_filter_batch(x, profile)
                adv_filtered = apply_filter_batch(x_adv, profile)
                pred_clean_filtered[profile_name] = predict(model, clean_filtered)
                pred_adv_filtered[profile_name] = predict(model, adv_filtered)

        attack_batches.append(
            AttackBatch(
                x=x.detach(),
                y=y.detach(),
                pred_clean=pred_clean.detach(),
                pred_clean_filtered={k: v.detach() for k, v in pred_clean_filtered.items()},
                x_adv=x_adv.detach(),
                pred_adv=pred_adv.detach(),
                pred_adv_filtered={k: v.detach() for k, v in pred_adv_filtered.items()},
            )
        )

        if batch_idx == 1 or batch_idx % args.log_interval == 0 or batch_idx == len(test_loader):
            elapsed = time.time() - batch_start
            running_adv_success = sum(
                int(((batch.pred_clean == batch.y) & (batch.pred_adv != batch.y)).sum().item())
                for batch in attack_batches
            )
            running_eligible = sum(int((batch.pred_clean == batch.y).sum().item()) for batch in attack_batches)
            LOGGER.info(
                "attack=%s batch %d/%d done - running_attack_success=%.2f%% - elapsed_batch=%.1fs",
                attack_name,
                batch_idx,
                len(test_loader),
                100.0 * safe_div(running_adv_success, running_eligible),
                elapsed,
            )

    rows: List[Dict[str, object]] = []
    for profile_name, profile in profiles.items():
        metrics = compute_metrics_from_batches(attack_batches, filter_profile=profile_name)
        row: Dict[str, object] = {
            "attack": attack_name,
            "filter_profile": profile_name,
            "filter_config": asdict(profile),
        }
        row.update(asdict(metrics))
        rows.append(row)

    LOGGER.info("Finished attack=%s in %.1fs", attack_name, time.time() - start)
    return rows


def save_results(rows: Sequence[Dict[str, object]], csv_path: Path, json_path: Path, extra: Dict[str, object]) -> None:
    if not rows:
        raise RuntimeError("No results to save")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    csv_rows: List[Dict[str, object]] = []
    for row in rows:
        csv_row = dict(row)
        csv_row["filter_config"] = json.dumps(csv_row["filter_config"], sort_keys=True)
        csv_rows.append(csv_row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"extra": extra, "results": list(rows)}, f, indent=2)

    LOGGER.info("Saved CSV to %s", csv_path)
    LOGGER.info("Saved JSON to %s", json_path)


def print_summary(rows: Sequence[Dict[str, object]]) -> None:
    print("\n=== Audio Attack Suite Results ===")
    print(
        "attack,filter_profile,attack_success_rate_on_eligible,recall,precision,f1,"
        "false_positive_rate,rtp_percent,tp,fn,fp,tn,num_failures"
    )
    for row in rows:
        print(
            f"{row['attack']},"
            f"{row['filter_profile']},"
            f"{row['attack_success_rate_on_eligible']:.2f},"
            f"{row['recall']:.2f},"
            f"{row['precision']:.2f},"
            f"{row['f1']:.2f},"
            f"{row['false_positive_rate']:.2f},"
            f"{row['rtp_percent']:.2f},"
            f"{row['tp']},"
            f"{row['fn']},"
            f"{row['fp']},"
            f"{row['tn']},"
            f"{row['num_failures']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FGSM, DeepFool and CW-L2-style attacks on audio filters.")
    parser.add_argument("--data-dir", default="data/raw/speech_commands", help="Speech Commands data directory.")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated class labels to use.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt",
        help="Trained SpeechResCNN checkpoint.",
    )
    parser.add_argument("--attacks", default="fgsm,deepfool", help="Comma-separated attacks: fgsm,deepfool,cw_l2")
    parser.add_argument(
        "--filter-profiles",
        default="v4_fixed_high_f1,v4_entropy_balanced,v2_q5_cross5",
        help="Comma-separated filter profiles.",
    )
    parser.add_argument("--max-test", type=int, default=1200, help="Limit test samples. Use <=0 for all.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size. CW-L2 usually needs a smaller value.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device selection.")
    parser.add_argument("--disable-cudnn", action="store_true", help="Use CUDA without cuDNN.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout used by checkpoint architecture.")
    parser.add_argument("--no-download", action="store_true", help="Disable dataset download.")

    parser.add_argument("--fgsm-epsilon", type=float, default=2.0, help="FGSM epsilon in dB units.")
    parser.add_argument("--deepfool-max-iter", type=int, default=30)
    parser.add_argument("--deepfool-overshoot", type=float, default=0.02)

    parser.add_argument("--cw-max-iterations", type=int, default=200)
    parser.add_argument("--cw-binary-search-steps", type=int, default=3)
    parser.add_argument("--cw-initial-const", type=float, default=1.0)
    parser.add_argument("--cw-learning-rate", type=float, default=0.05)
    parser.add_argument("--cw-confidence", type=float, default=0.0)
    parser.add_argument("--cw-abort-early", action="store_true")

    parser.add_argument("--csv-out", default="artifacts/audio_poc/attack_suite_results.csv")
    parser.add_argument("--json-out", default="artifacts/audio_poc/attack_suite_results.json")
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
    attack_names = parse_csv_options(args.attacks)
    profile_names = parse_csv_options(args.filter_profiles)

    valid_attacks = {"fgsm", "deepfool", "cw_l2"}
    invalid_attacks = sorted(set(attack_names) - valid_attacks)
    invalid_profiles = sorted(set(profile_names) - set(FILTER_PROFILES))
    if invalid_attacks:
        raise ValueError(f"Invalid attacks: {invalid_attacks}")
    if invalid_profiles:
        raise ValueError(f"Invalid filter profiles: {invalid_profiles}")

    selected_profiles = {name: FILTER_PROFILES[name] for name in profile_names}

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
    LOGGER.info("Attacks: %s", attack_names)
    LOGGER.info("Filter profiles: %s", profile_names)

    test_loader = make_test_loader(args, device=device)
    all_rows: List[Dict[str, object]] = []
    start = time.time()

    for attack_name in attack_names:
        rows = evaluate_attack(
            model=model,
            test_loader=test_loader,
            device=device,
            attack_name=attack_name,
            profiles=selected_profiles,
            args=args,
        )
        all_rows.extend(rows)

    save_results(
        all_rows,
        csv_path=Path(args.csv_out),
        json_path=Path(args.json_out),
        extra={
            "classes": class_names,
            "architecture": "SpeechResCNN",
            "checkpoint": str(checkpoint_path),
            "device": device.type,
            "max_test": args.max_test,
            "attacks": attack_names,
            "filter_profiles": {name: asdict(profile) for name, profile in selected_profiles.items()},
            "fgsm_epsilon": args.fgsm_epsilon,
            "deepfool_max_iter": args.deepfool_max_iter,
            "deepfool_overshoot": args.deepfool_overshoot,
            "cw_max_iterations": args.cw_max_iterations,
            "cw_binary_search_steps": args.cw_binary_search_steps,
            "cw_initial_const": args.cw_initial_const,
            "cw_learning_rate": args.cw_learning_rate,
            "cw_confidence": args.cw_confidence,
        },
    )
    print_summary(all_rows)
    LOGGER.info("Audio attack suite finished in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
