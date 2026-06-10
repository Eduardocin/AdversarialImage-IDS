#!/usr/bin/env python3
"""Standalone PoC: DeepDetector-style adversarial detection for audio.

This script intentionally does NOT integrate with the project runner/YAML flow.
It validates the idea quickly while keeping the evaluation closer to the
DeepDetector paper tables.

Main idea:
    - C(x) is the classifier prediction for an input.
    - T(x) is the filtered input.
    - The detector flags a sample as adversarial when C(x) != C(T(x)).

For adversarial samples, the paper-style metrics are computed over *effective*
attacks only:
    - eligible attack sample: the clean sample is correctly classified.
    - attack success: C(x_clean) == y and C(x_adv) != y.
    - attack failure: C(x_clean) == y and C(x_adv) == y.
    - TP/FN: computed over successful adversarial examples.
    - FP/TN: computed over benign clean examples.
    - RTP: true positives that are also restored to the original class after
      filtering, i.e. C(T(x_adv)) == y.

Example:
    python scripts/dev/audio_adversarial_poc.py \
        --device cuda \
        --data-dir data/raw/speech_commands \
        --epochs 20 \
        --max-train 12000 \
        --max-test 1200 \
        --epsilon 2.0 \
        --force-train \
        --checkpoint artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# WSL exposes the NVIDIA driver libraries here. Put this in LD_LIBRARY_PATH
# before importing torch/torchaudio so CUDA dlopen calls can find libcuda.so.
if Path("/usr/lib/wsl/lib").exists():
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    wsl_cuda_path = "/usr/lib/wsl/lib"
    if wsl_cuda_path not in current_ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{wsl_cuda_path}:{current_ld_path}" if current_ld_path else wsl_cuda_path
        )

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
ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"
EXTRACTED_DIR = Path("SpeechCommands") / "speech_commands_v0.02"


@dataclass
class PaperStyleMetrics:
    """DeepDetector-style metrics for the audio PoC.

    Positives are successful adversarial examples.
    Negatives are benign clean examples.

    Therefore:
        TP/FN denominator = successful adversarial examples.
        FP/TN denominator = benign clean examples.
        num_failures = attack failures on clean-correct eligible samples.
        RTP = TP samples restored to the true class after filtering.
    """

    num_clean_total: int
    num_clean_correct: int
    num_clean_misclassified: int
    num_eligible_for_attack: int
    num_failures: int
    num_successful_adversarial: int
    tp: int
    fn: int
    fp: int
    tn: int
    rtp: int
    rtp_percent: float
    recall: float
    precision: float
    f1: float
    false_positive_rate: float
    clean_accuracy: float
    adversarial_accuracy_all: float
    adversarial_accuracy_on_eligible: float
    filtered_adversarial_accuracy_all: float
    attack_success_rate_on_eligible: float
    attack_success_rate_all: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception as exc:
            LOGGER.warning("Could not seed CUDA safely: %s", exc)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def configure_backend(disable_cudnn: bool) -> None:
    if disable_cudnn:
        torch.backends.cudnn.enabled = False
        LOGGER.warning(
            "cuDNN disabled by --disable-cudnn. The script can still use CUDA, "
            "but convolutions may be slower."
        )
    else:
        LOGGER.info("cuDNN enabled: %s", torch.backends.cudnn.enabled)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cpu":
        LOGGER.info("Device forced by user: cpu")
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
        LOGGER.info("Device forced by user: cuda")
        return torch.device("cuda")

    if torch.cuda.is_available():
        LOGGER.info("CUDA is available. Using device: cuda")
        return torch.device("cuda")

    LOGGER.info("CUDA is not available. Using device: cpu")
    return torch.device("cpu")


def log_environment(device: torch.device) -> None:
    LOGGER.info("torch version: %s", torch.__version__)
    LOGGER.info("torchaudio version: %s", torchaudio.__version__)
    LOGGER.info("torch CUDA build: %s", torch.version.cuda)
    LOGGER.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
    LOGGER.info("Selected device: %s", device)

    if device.type == "cuda":
        LOGGER.info("CUDA device name: %s", torch.cuda.get_device_name(0))
        LOGGER.info("LD_LIBRARY_PATH: %s", os.environ.get("LD_LIBRARY_PATH", ""))


def pad_or_truncate(waveform: torch.Tensor, num_samples: int = NUM_SAMPLES) -> torch.Tensor:
    """Ensure every waveform has exactly num_samples samples."""
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with shape [channels, samples], got {tuple(waveform.shape)}")

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    current = waveform.size(1)
    if current > num_samples:
        return waveform[:, :num_samples]
    if current < num_samples:
        return F.pad(waveform, (0, num_samples - current))
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
        self.root.mkdir(parents=True, exist_ok=True)
        self.classes = list(classes)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.sample_rate = sample_rate
        self.num_samples = num_samples

        self._log_dataset_state(subset=subset, download=download)

        self.base = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(self.root),
            url="speech_commands_v0.02",
            folder_in_archive="SpeechCommands",
            download=download,
            subset=subset,
        )

        LOGGER.info("Speech Commands metadata ready for subset=%s. Filtering selected classes...", subset)

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

    def _log_dataset_state(self, subset: str, download: bool) -> None:
        archive_path = self.root / ARCHIVE_NAME
        extracted_path = self.root / EXTRACTED_DIR
        LOGGER.info("Preparing Speech Commands subset=%s", subset)
        LOGGER.info("Dataset root: %s", self.root.resolve())
        LOGGER.info("Download enabled: %s", download)

        if extracted_path.exists():
            LOGGER.info("Extracted dataset directory already exists: %s", extracted_path)
            return

        if archive_path.exists():
            archive_size_gb = archive_path.stat().st_size / (1024**3)
            LOGGER.info(
                "Archive already exists: %s (%.2f GB). torchaudio may now extract it silently.",
                archive_path,
                archive_size_gb,
            )
        elif download:
            LOGGER.info(
                "Archive not found. torchaudio will download it and then extract it. "
                "The extraction step can take several minutes and may not show progress."
            )
        else:
            LOGGER.warning("Archive/dataset not found and --no-download was passed.")

        LOGGER.info(
            "If execution appears paused after 'Opened tar file', it is probably extracting many wav files. "
            "On /mnt/c or OneDrive this can be very slow; prefer a Linux path such as ~/datasets/speech_commands."
        )

    def _filter_indices(self) -> List[int]:
        indices: List[int] = []
        walker = getattr(self.base, "_walker", None)

        if walker is not None:
            for idx, file_path in enumerate(walker):
                label = Path(file_path).parent.name
                if label in self.class_to_idx:
                    indices.append(idx)
            return indices

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
        spec_db = torch.clamp(spec_db, min=DB_MIN, max=DB_MAX).to(torch.float32)
        target = torch.tensor(self.class_to_idx[label], dtype=torch.long)
        return spec_db, target


class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm + SiLU block.

    SiLU is usually a little smoother than ReLU and works well for compact audio
    CNNs. Bias is disabled because BatchNorm has affine parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int | Tuple[int, int] = 1,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Small residual block for log-Mel spectrograms."""

    def __init__(self, in_channels: int, out_channels: int, stride: int | Tuple[int, int] = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut: nn.Module
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class SpeechCNN(nn.Module):
    """Compact residual CNN for log-Mel spectrogram classification.

    Compared with the first PoC CNN, this version has:
        - residual blocks, which usually train more stably;
        - strided convolutions instead of repeated max-pooling;
        - global adaptive pooling, so the classifier is independent of time size;
        - SiLU activations and AdamW-friendly regularization.
    """

    def __init__(self, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(1, 32, kernel_size=5, stride=1, padding=2),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=(2, 2)),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=(2, 2)),
            ResidualBlock(128, 128),
            ResidualBlock(128, 192, stride=(2, 2)),
            ResidualBlock(192, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x_db: torch.Tensor) -> torch.Tensor:
        # External domain is dB. Internal normalization keeps optimization stable.
        x = (x_db - DB_MIN) / (DB_MAX - DB_MIN)
        x = torch.clamp(x, 0.0, 1.0)
        x = self.features(x)
        return self.classifier(x)


def limit_dataset(dataset: Dataset, max_items: Optional[int], seed: int) -> Dataset:
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator).tolist()
    LOGGER.info("Limiting dataset from %d to %d samples", len(dataset), max_items)
    return Subset(dataset, perm[:max_items])


def make_loaders(args: argparse.Namespace, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]

    LOGGER.info("Creating training dataset. This step downloads/extracts Speech Commands if needed.")
    train_ds = SpeechCommandsSpectrogramDataset(
        root=Path(args.data_dir),
        subset="training",
        classes=classes,
        download=not args.no_download,
    )

    LOGGER.info("Creating testing dataset.")
    test_ds = SpeechCommandsSpectrogramDataset(
        root=Path(args.data_dir),
        subset="testing",
        classes=classes,
        download=not args.no_download,
    )

    train_ds = limit_dataset(train_ds, args.max_train, args.seed)
    test_ds = limit_dataset(test_ds, args.max_test, args.seed + 1)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    LOGGER.info("DataLoaders ready: train_batches=%d, test_batches=%d", len(train_loader), len(test_loader))
    return train_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    scheduler_name: str,
    log_interval: int,
) -> None:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=lr * 0.05,
        )
    else:
        scheduler = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0
        start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=device.type == "cuda")
            y = y.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_items += batch_size

            if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(train_loader):
                current_lr = optimizer.param_groups[0]["lr"]
                LOGGER.info(
                    "Epoch %d/%d - batch %d/%d - lr=%.2e - running_loss=%.4f - running_acc=%.2f%%",
                    epoch,
                    epochs,
                    batch_idx,
                    len(train_loader),
                    current_lr,
                    total_loss / max(total_items, 1),
                    100.0 * total_correct / max(total_items, 1),
                )

        if scheduler is not None:
            scheduler.step()

        LOGGER.info(
            "Epoch %d/%d done - loss=%.4f - train_acc=%.2f%% - elapsed=%.1fs",
            epoch,
            epochs,
            total_loss / max(total_items, 1),
            100.0 * total_correct / max(total_items, 1),
            time.time() - start,
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
    """FGSM attack in spectrogram dB space.

    The function restores the original model.training state after generating
    adversarial examples, so it is safe to reuse outside evaluation code.
    """
    was_training = model.training
    model.eval()

    x_adv = x_db.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()

    x_adv = x_adv.detach() + epsilon * x_adv.grad.detach().sign()
    x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)

    if was_training:
        model.train()

    return x_adv


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

        # Combination rule inspired by DeepDetector: pick the transform that
        # changes each value less relative to the original normalized input.
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


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    epsilon: float,
    save_debug: bool,
    debug_dir: Path,
    class_names: Sequence[str],
    log_interval: int,
) -> PaperStyleMetrics:
    """Evaluate with DeepDetector-style detection metrics.

    Detection rule:
        detected(x) = C(x) != C(T(x))

    Paper-style counting used here:
        - FP/TN: detector decision on benign clean samples.
        - TP/FN: detector decision on successful adversarial samples only.
        - num_failures: clean-correct samples where FGSM did not fool C.
        - RTP: TP samples whose filtered prediction returns to the true label.
    """
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
    debug_saved = False

    LOGGER.info("Starting evaluation with FGSM epsilon=%.4f over %d batches", epsilon, len(test_loader))

    for batch_idx, (x, y) in enumerate(test_loader, start=1):
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")

        pred_clean = predict(model, x)
        clean_filtered = deepdetector_filter_batch(x)
        pred_clean_filtered = predict(model, clean_filtered)

        x_adv = fgsm_attack(model, x, y, epsilon=epsilon)
        pred_adv = predict(model, x_adv)

        adv_filtered = deepdetector_filter_batch(x_adv)
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

        # Benign side of the detector.
        # False positive means a clean sample changes class after filtering.
        benign_detected_as_adv = pred_clean != pred_clean_filtered
        fp += benign_detected_as_adv.sum().item()
        tn += (~benign_detected_as_adv).sum().item()

        # Adversarial side of the detector.
        # Only clean-correct samples can produce meaningful attack successes.
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

        if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(test_loader):
            LOGGER.info(
                "Eval batch %d/%d - clean_acc=%.2f%% - adv_acc=%.2f%% - "
                "failures=%d tp=%d fn=%d fp=%d rtp=%d",
                batch_idx,
                len(test_loader),
                100.0 * safe_div(clean_correct_total, num_clean_total),
                100.0 * safe_div(adv_correct_total, num_clean_total),
                num_failures,
                tp,
                fn,
                fp,
                rtp,
            )

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
    rtp_percent = 100.0 * safe_div(rtp, tp)
    false_positive_rate = safe_div(fp, fp + tn)

    return PaperStyleMetrics(
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
        ("benign.png", clean_np, f"Benign | y={true_label} | C(x)={pred_clean}"),
        ("adversarial.png", adv_np, f"Adversarial | C(x_adv)={pred_adv}"),
        ("delta.png", delta_np, "Delta: adversarial - benign"),
        ("filtered.png", filtered_np, f"Filtered adversarial | C(T(x_adv))={pred_filtered}"),
    ]

    for filename, matrix, title in items:
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


def safe_torch_load(path: Path, device: torch.device):
    """Use weights_only=True when available without breaking older PyTorch."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception as exc:
        LOGGER.warning("weights_only=True load failed for %s; falling back. Details: %s", path, exc)
        return torch.load(path, map_location=device, weights_only=False)


def save_metrics(metrics: PaperStyleMetrics, output_path: Path, extra: Dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": asdict(metrics),
        "extra": extra,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info("Saved metrics to %s", output_path)


def print_metrics(metrics: PaperStyleMetrics) -> None:
    print("\n=== Audio Adversarial PoC Results ===")
    print("Classifier/attack summary:")
    print(f"Clean total:                       {metrics.num_clean_total}")
    print(f"Clean correctly classified:        {metrics.num_clean_correct}")
    print(f"Clean misclassified/excluded:      {metrics.num_clean_misclassified}")
    print(f"Eligible for attack:               {metrics.num_eligible_for_attack}")
    print(f"Attack failures:                   {metrics.num_failures}")
    print(f"Successful adversarial examples:   {metrics.num_successful_adversarial}")
    print(f"Clean accuracy:                    {metrics.clean_accuracy:6.2f}%")
    print(f"Adversarial accuracy, all:          {metrics.adversarial_accuracy_all:6.2f}%")
    print(f"Adversarial accuracy, eligible:     {metrics.adversarial_accuracy_on_eligible:6.2f}%")
    print(f"Filtered adversarial accuracy:      {metrics.filtered_adversarial_accuracy_all:6.2f}%")
    print(f"Attack success rate, eligible:      {metrics.attack_success_rate_on_eligible:6.2f}%")
    print("")
    print("Paper-style DeepDetector metrics:")
    print("num_failures,tp,fn,fp,tn,rtp,rtp_percent,recall,precision,f1,false_positive_rate")
    print(
        f"{metrics.num_failures},"
        f"{metrics.tp},"
        f"{metrics.fn},"
        f"{metrics.fp},"
        f"{metrics.tn},"
        f"{metrics.rtp},"
        f"{metrics.rtp_percent:.2f},"
        f"{metrics.recall:.2f},"
        f"{metrics.precision:.2f},"
        f"{metrics.f1:.2f},"
        f"{metrics.false_positive_rate:.2f}"
    )
    print("")
    print("Interpretation note:")
    print("  TP/FN are computed only over successful adversarial examples.")
    print("  FP/TN are computed over benign clean examples.")
    print("  RTP counts detected adversarial examples restored to the true class by filtering.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone PoC for DeepDetector-style audio adversarial detection."
    )
    parser.add_argument("--data-dir", default="data/raw/speech_commands", help="Speech Commands data directory.")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated class labels to use.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs when checkpoint is absent.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training/evaluation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Cross-entropy label smoothing.")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine", help="Learning-rate scheduler.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Classifier dropout.")
    parser.add_argument("--epsilon", type=float, default=2.0, help="FGSM epsilon in dB units.")
    parser.add_argument("--max-train", type=int, default=3000, help="Limit training samples. Use <=0 for all.")
    parser.add_argument("--max-test", type=int, default=600, help="Limit test samples. Use <=0 for all.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device selection.")
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="Use CUDA without cuDNN. Useful when cuDNN libraries are broken but CUDA itself works.",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/audio_poc/speech_rescnn_poc.pt",
        help="Checkpoint path. If it exists, it is loaded unless --force-train is passed.",
    )
    parser.add_argument("--force-train", action="store_true", help="Train even if checkpoint exists.")
    parser.add_argument("--no-download", action="store_true", help="Disable dataset download.")
    parser.add_argument("--save-debug", action="store_true", help="Save debug PNG spectrograms.")
    parser.add_argument("--debug-dir", default="artifacts/audio/debug", help="Directory for debug PNGs.")
    parser.add_argument("--metrics-out", default="artifacts/audio_poc/metrics.json", help="JSON metrics output path.")
    parser.add_argument("--train-log-interval", type=int, default=10, help="Training log interval in batches.")
    parser.add_argument("--eval-log-interval", type=int, default=5, help="Evaluation log interval in batches.")
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

    LOGGER.info("Classes: %s", class_names)
    LOGGER.info("Data directory: %s", Path(args.data_dir).resolve())

    train_loader, test_loader = make_loaders(args, device=device)
    model = SpeechCNN(num_classes=len(class_names), dropout=args.dropout).to(device)
    LOGGER.info("Model architecture: SpeechResCNN")

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists() and not args.force_train:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        payload = safe_torch_load(checkpoint_path, device)
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        model.load_state_dict(state_dict)
    else:
        LOGGER.info(
            "Training SpeechResCNN for %d epochs with AdamW lr=%.2e weight_decay=%.2e label_smoothing=%.3f scheduler=%s",
            args.epochs,
            args.lr,
            args.weight_decay,
            args.label_smoothing,
            args.scheduler,
        )
        train_model(
            model,
            train_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            scheduler_name=args.scheduler,
            log_interval=max(args.train_log_interval, 1),
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "architecture": "SpeechResCNN",
                "classes": class_names,
                "sample_rate": SAMPLE_RATE,
                "n_mels": N_MELS,
                "db_min": DB_MIN,
                "db_max": DB_MAX,
                "dropout": args.dropout,
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
        log_interval=max(args.eval_log_interval, 1),
    )
    print_metrics(metrics)
    save_metrics(
        metrics,
        Path(args.metrics_out),
        extra={
            "classes": class_names,
            "architecture": "SpeechResCNN",
            "epsilon_db": args.epsilon,
            "device": device.type,
            "disable_cudnn": args.disable_cudnn,
            "max_train": args.max_train,
            "max_test": args.max_test,
            "checkpoint": str(checkpoint_path),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "scheduler": args.scheduler,
            "dropout": args.dropout,
        },
    )


if __name__ == "__main__":
    main()
