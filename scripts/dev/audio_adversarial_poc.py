#!/usr/bin/env python3
"""Standalone PoC: DeepDetector-style adversarial detection for audio.

This script intentionally does NOT integrate with the project runner/YAML flow.
It validates the idea quickly:

1. Load Speech Commands with torchaudio.
2. Keep a small subset of commands.
3. Convert waveforms to log-Mel spectrograms in dB.
4. Train or load a small CNN classifier.
5. Generate FGSM adversarial examples directly in the spectrogram domain.
6. Apply a DeepDetector-inspired adaptive filter:
   - normalize spectrogram to uint8;
   - compute entropy;
   - apply adaptive scalar quantization;
   - optionally apply spatial smoothing;
   - denormalize back to dB before model inference.
7. Compare C(x) and C(T(x)) to detect adversarial samples.

Example:
    python scripts/dev/audio_adversarial_poc.py \
        --data-dir data/raw/speech_commands \
        --epochs 5 \
        --max-train 3000 \
        --max-test 600 \
        --epsilon 2.0 \
        --save-debug

Notes:
    - The model receives spectrograms in dB scale externally.
    - The CNN normalizes dB values internally for numerical stability.
    - FGSM epsilon is expressed in dB units. Start with 1.0 to 3.0.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import torchaudio
except ImportError as exc:  # pragma: no cover - user-facing runtime check
    raise SystemExit(
        "torchaudio is required for this PoC. Install it with a PyTorch-compatible build."
    ) from exc


LOGGER = logging.getLogger("audio_adversarial_poc")

DEFAULT_CLASSES = ["yes", "no", "up", "down", "left", "right"]
SAMPLE_RATE = 16_000
NUM_SAMPLES = 16_000
N_MELS = 64
DB_MIN = -80.0
DB_MAX = 0.0


@dataclass
class Metrics:
    clean_accuracy: float
    adversarial_accuracy: float
    filtered_adversarial_accuracy: float
    attack_success_rate: float
    tp: int
    fp: int
    fn: int
    tn: int
    recall: float
    precision: float
    f1: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pad_or_truncate(waveform: torch.Tensor, num_samples: int = NUM_SAMPLES) -> torch.Tensor:
    """Ensure every waveform has exactly num_samples samples."""
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with shape [channels, samples], got {tuple(waveform.shape)}")

    # Convert to mono if necessary.
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    current = waveform.size(1)
    if current > num_samples:
        return waveform[:, :num_samples]
    if current < num_samples:
        pad_amount = num_samples - current
        return F.pad(waveform, (0, pad_amount))
    return waveform


class SpeechCommandsSpectrogramDataset(Dataset):
    """Speech Commands subset converted to log-Mel spectrograms."""

    def __init__(
        self,
        root: Path,
        subset: str,
        classes: Sequence[str],
        download: bool = True,
        sample_rate: int = SAMPLE_RATE,
        num_samples: int = NUM_SAMPLES,
        n_mels: int = N_MELS,
    ) -> None:
        self.root = Path(root)
        self.classes = list(classes)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.sample_rate = sample_rate
        self.num_samples = num_samples

        self.base = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(self.root),
            url="speech_commands_v0.02",
            folder_in_archive="SpeechCommands",
            download=download,
            subset=subset,
        )

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

        self.indices = self._filter_indices()
        if not self.indices:
            raise RuntimeError(f"No samples found for classes {self.classes} in subset={subset!r}")

        LOGGER.info(
            "Loaded SpeechCommands subset=%s with %d selected samples across classes=%s",
            subset,
            len(self.indices),
            self.classes,
        )

    def _filter_indices(self) -> List[int]:
        indices: List[int] = []

        # torchaudio exposes _walker in current versions. It is much faster than
        # loading every audio file only to inspect its label.
        walker = getattr(self.base, "_walker", None)
        if walker is not None:
            for idx, file_path in enumerate(walker):
                label = Path(file_path).parent.name
                if label in self.class_to_idx:
                    indices.append(idx)
            return indices

        # Fallback for older torchaudio versions.
        for idx in range(len(self.base)):
            _, _, label, *_ = self.base[idx]
            if label in self.class_to_idx:
                indices.append(idx)
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = self.indices[item]
        waveform, sample_rate, label, *_ = self.base[base_idx]

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        waveform = pad_or_truncate(waveform, self.num_samples)
        spec = self.mel(waveform)
        spec_db = self.to_db(spec)

        # Keep a stable external input domain for the classifier and attack.
        spec_db = torch.clamp(spec_db, min=DB_MIN, max=DB_MAX).to(torch.float32)
        target = torch.tensor(self.class_to_idx[label], dtype=torch.long)
        return spec_db, target


class SpeechCNN(nn.Module):
    """Small CNN for log-Mel spectrogram classification.

    Externally, this module receives spectrograms in dB scale, usually [-80, 0].
    Internally, it maps them to [0, 1] to make optimization easier.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_db: torch.Tensor) -> torch.Tensor:
        # Model input remains dB. This normalization is only internal.
        x = (x_db - DB_MIN) / (DB_MAX - DB_MIN)
        x = torch.clamp(x, 0.0, 1.0)
        x = self.features(x)
        return self.classifier(x)


def limit_dataset(dataset: Dataset, max_items: int | None, seed: int) -> Dataset:
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator).tolist()
    return Subset(dataset, perm[:max_items])


def make_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]

    train_ds = SpeechCommandsSpectrogramDataset(
        root=Path(args.data_dir),
        subset="training",
        classes=classes,
        download=not args.no_download,
    )
    test_ds = SpeechCommandsSpectrogramDataset(
        root=Path(args.data_dir),
        subset="testing",
        classes=classes,
        download=not args.no_download,
    )

    train_ds = limit_dataset(train_ds, args.max_train, args.seed)
    test_ds = limit_dataset(test_ds, args.max_test, args.seed + 1)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_items += batch_size

        LOGGER.info(
            "Epoch %d/%d - loss=%.4f - train_acc=%.2f%%",
            epoch,
            epochs,
            total_loss / max(total_items, 1),
            100.0 * total_correct / max(total_items, 1),
        )


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x).argmax(dim=1)


def fgsm_attack(
    model: nn.Module,
    x_db: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    clip_min: float = DB_MIN,
    clip_max: float = DB_MAX,
) -> torch.Tensor:
    """FGSM attack in spectrogram dB space."""
    model.eval()
    x_adv = x_db.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()

    perturbation = epsilon * x_adv.grad.detach().sign()
    x_adv = x_adv.detach() + perturbation
    return torch.clamp(x_adv, min=clip_min, max=clip_max)


def normalize_spectrogram_for_filters(spec_db: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Map a 2D dB spectrogram to uint8 [0, 255], preserving min/max."""
    spec = spec_db.astype(np.float32)
    spec_min = float(spec.min())
    spec_max = float(spec.max())

    denom = spec_max - spec_min
    if denom < 1e-8:
        return np.zeros_like(spec, dtype=np.uint8), spec_min, spec_max

    normalized = (spec - spec_min) / denom
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8), spec_min, spec_max


def denormalize_spectrogram_from_filters(
    filtered_uint8: np.ndarray,
    original_min: float,
    original_max: float,
) -> np.ndarray:
    """Map uint8 filtered spectrogram back to original dB scale."""
    x = filtered_uint8.astype(np.float32) / 255.0
    return x * (original_max - original_min) + original_min


def entropy_uint8(image: np.ndarray) -> float:
    hist = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    probs = hist / max(float(hist.sum()), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def quantize_uint8(image: np.ndarray, intervals: int) -> np.ndarray:
    """Uniform scalar quantization over uint8 values."""
    if intervals <= 1:
        return np.zeros_like(image, dtype=np.uint8)

    image_f = image.astype(np.float32)
    step = 255.0 / float(intervals - 1)
    quantized = np.round(image_f / step) * step
    return np.clip(quantized, 0, 255).astype(np.uint8)


def cross_mask(size: int = 5) -> np.ndarray:
    if size % 2 == 0 or size < 3:
        raise ValueError("cross mask size must be odd and >= 3")
    mask = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    mask[center, :] = 1.0
    mask[:, center] = 1.0
    mask /= mask.sum()
    return mask


def smooth_uint8(image: np.ndarray, size: int = 5) -> np.ndarray:
    """Small dependency-free spatial smoothing with a cross mask."""
    mask = cross_mask(size)
    pad = size // 2
    padded = np.pad(image.astype(np.float32), pad_width=pad, mode="edge")
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(size):
        for j in range(size):
            weight = mask[i, j]
            if weight == 0:
                continue
            output += weight * padded[i : i + image.shape[0], j : j + image.shape[1]]

    return np.clip(np.round(output), 0, 255).astype(np.uint8)


def deepdetector_filter_single(spec_db: torch.Tensor) -> torch.Tensor:
    """Apply the PoC DeepDetector-style filter to one [1, n_mels, time] tensor."""
    if spec_db.ndim != 3 or spec_db.size(0) != 1:
        raise ValueError(f"Expected [1, n_mels, time], got {tuple(spec_db.shape)}")

    spec_np = spec_db.squeeze(0).detach().cpu().numpy().astype(np.float32)
    spec_uint8, spec_min, spec_max = normalize_spectrogram_for_filters(spec_np)

    entropy = entropy_uint8(spec_uint8)
    if entropy < 4.0:
        filtered_uint8 = quantize_uint8(spec_uint8, intervals=2)
    elif entropy < 5.0:
        filtered_uint8 = quantize_uint8(spec_uint8, intervals=4)
    else:
        quantized = quantize_uint8(spec_uint8, intervals=6)
        smoothed = smooth_uint8(quantized, size=5)

        # Combination rule inspired by the original DeepDetector filter:
        # choose the transformed value that changes each pixel less.
        diff_quant = np.abs(quantized.astype(np.int16) - spec_uint8.astype(np.int16))
        diff_smooth = np.abs(smoothed.astype(np.int16) - spec_uint8.astype(np.int16))
        filtered_uint8 = np.where(diff_quant <= diff_smooth, quantized, smoothed).astype(np.uint8)

    filtered_db = denormalize_spectrogram_from_filters(filtered_uint8, spec_min, spec_max)
    filtered_db = np.clip(filtered_db, DB_MIN, DB_MAX).astype(np.float32)
    return torch.from_numpy(filtered_db).unsqueeze(0)


def deepdetector_filter_batch(x_db: torch.Tensor) -> torch.Tensor:
    """Apply filter sample-by-sample and return tensor on the original device."""
    device = x_db.device
    filtered = [deepdetector_filter_single(sample) for sample in x_db.detach().cpu()]
    return torch.stack(filtered, dim=0).to(device=device, dtype=x_db.dtype)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    epsilon: float,
    save_debug: bool,
    debug_dir: Path,
    class_names: Sequence[str],
) -> Metrics:
    model.eval()

    total = 0
    clean_correct = 0
    adv_correct = 0
    filtered_adv_correct = 0
    successful_attacks = 0

    tp = fp = fn = tn = 0
    debug_saved = False

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        pred_clean = predict(model, x)
        clean_filtered = deepdetector_filter_batch(x)
        pred_clean_filtered = predict(model, clean_filtered)

        x_adv = fgsm_attack(model, x, y, epsilon=epsilon)
        pred_adv = predict(model, x_adv)

        adv_filtered = deepdetector_filter_batch(x_adv)
        pred_adv_filtered = predict(model, adv_filtered)

        batch_size = x.size(0)
        total += batch_size
        clean_correct += (pred_clean == y).sum().item()
        adv_correct += (pred_adv == y).sum().item()
        filtered_adv_correct += (pred_adv_filtered == y).sum().item()

        # Detection on benign samples: C(x) != C(T(x)) is a false positive.
        benign_changed = pred_clean != pred_clean_filtered
        fp += benign_changed.sum().item()
        tn += (~benign_changed).sum().item()

        # Detection on adversarial samples: evaluate only effectual attacks.
        # A successful adversarial sample is one where C(x_adv) != y.
        successful = pred_adv != y
        successful_attacks += successful.sum().item()

        detected_adv = pred_adv != pred_adv_filtered
        tp += (successful & detected_adv).sum().item()
        fn += (successful & ~detected_adv).sum().item()

        if save_debug and not debug_saved and batch_size > 0:
            save_debug_images(
                clean=x[0].detach().cpu(),
                adversarial=x_adv[0].detach().cpu(),
                filtered=adv_filtered[0].detach().cpu(),
                true_label=class_names[int(y[0].detach().cpu())],
                pred_clean=class_names[int(pred_clean[0].detach().cpu())],
                pred_adv=class_names[int(pred_adv[0].detach().cpu())],
                pred_filtered=class_names[int(pred_adv_filtered[0].detach().cpu())],
                output_dir=debug_dir,
            )
            debug_saved = True

    recall = safe_div(tp, tp + fn)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2.0 * recall * precision, recall + precision)

    return Metrics(
        clean_accuracy=safe_div(clean_correct, total),
        adversarial_accuracy=safe_div(adv_correct, total),
        filtered_adversarial_accuracy=safe_div(filtered_adv_correct, total),
        attack_success_rate=safe_div(successful_attacks, total),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        recall=recall,
        precision=precision,
        f1=f1,
    )


def save_debug_images(
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    filtered: torch.Tensor,
    true_label: str,
    pred_clean: str,
    pred_adv: str,
    pred_filtered: str,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    clean_np = clean.squeeze(0).numpy()
    adv_np = adversarial.squeeze(0).numpy()
    filtered_np = filtered.squeeze(0).numpy()
    delta_np = adv_np - clean_np

    items = [
        ("benign.png", clean_np, f"Benign | y={true_label} | C(x)={pred_clean}", "auto"),
        ("adversarial.png", adv_np, f"Adversarial | C(x_adv)={pred_adv}", "auto"),
        ("delta.png", delta_np, "Delta: adversarial - benign", "auto"),
        ("filtered.png", filtered_np, f"Filtered adversarial | C(T(x_adv))={pred_filtered}", "auto"),
    ]

    for filename, matrix, title, scale in items:
        plt.figure(figsize=(8, 4))
        plt.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest")
        plt.title(title)
        plt.xlabel("Time frames")
        plt.ylabel("Mel bins")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()

    LOGGER.info("Saved debug spectrograms to %s", output_dir)


def save_metrics(metrics: Metrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)
    LOGGER.info("Saved metrics to %s", output_path)


def print_metrics(metrics: Metrics) -> None:
    print("\n=== Audio Adversarial PoC Results ===")
    print(f"Clean accuracy:               {100.0 * metrics.clean_accuracy:6.2f}%")
    print(f"Adversarial accuracy:         {100.0 * metrics.adversarial_accuracy:6.2f}%")
    print(f"Filtered adversarial accuracy:{100.0 * metrics.filtered_adversarial_accuracy:6.2f}%")
    print(f"Attack success rate:          {100.0 * metrics.attack_success_rate:6.2f}%")
    print("")
    print("Detection metrics using C(x) != C(T(x)):")
    print(f"TP: {metrics.tp}")
    print(f"FP: {metrics.fp}")
    print(f"FN: {metrics.fn}")
    print(f"TN: {metrics.tn}")
    print(f"Recall:    {100.0 * metrics.recall:6.2f}%")
    print(f"Precision: {100.0 * metrics.precision:6.2f}%")
    print(f"F1:        {100.0 * metrics.f1:6.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone PoC for DeepDetector-style audio adversarial detection."
    )
    parser.add_argument("--data-dir", default="data/raw/speech_commands", help="Speech Commands data directory.")
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class labels to use.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs when checkpoint is absent.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training/evaluation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epsilon", type=float, default=2.0, help="FGSM epsilon in dB units.")
    parser.add_argument("--max-train", type=int, default=3000, help="Limit training samples. Use <=0 for all.")
    parser.add_argument("--max-test", type=int, default=600, help="Limit test samples. Use <=0 for all.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/audio_poc/speech_cnn_poc.pt",
        help="Checkpoint path. If it exists, it is loaded unless --force-train is passed.",
    )
    parser.add_argument("--force-train", action="store_true", help="Train even if checkpoint exists.")
    parser.add_argument("--no-download", action="store_true", help="Disable dataset download.")
    parser.add_argument("--save-debug", action="store_true", help="Save debug PNG spectrograms.")
    parser.add_argument("--debug-dir", default="artifacts/audio/debug", help="Directory for debug PNGs.")
    parser.add_argument("--metrics-out", default="artifacts/audio_poc/metrics.json", help="JSON metrics output path.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    set_seed(args.seed)

    class_names = [item.strip() for item in args.classes.split(",") if item.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Classes: %s", class_names)

    train_loader, test_loader = make_loaders(args)
    model = SpeechCNN(num_classes=len(class_names)).to(device)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists() and not args.force_train:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=device)
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        model.load_state_dict(state_dict)
    else:
        LOGGER.info("Training SpeechCNN for %d epochs", args.epochs)
        train_model(model, train_loader, device, epochs=args.epochs, lr=args.lr)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": class_names,
                "sample_rate": SAMPLE_RATE,
                "n_mels": N_MELS,
            },
            checkpoint_path,
        )
        LOGGER.info("Saved checkpoint to %s", checkpoint_path)

    metrics = evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        epsilon=args.epsilon,
        save_debug=args.save_debug,
        debug_dir=Path(args.debug_dir),
        class_names=class_names,
    )
    print_metrics(metrics)
    save_metrics(metrics, Path(args.metrics_out))


if __name__ == "__main__":
    main()
