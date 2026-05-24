"""Inspect GoogLeNet predictions for a local ImageNet class folder."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.article_reproduction.table_4_imagenet import (  # noqa: E402
    DEFAULT_CONFIG,
    build_model,
    load_config,
    load_subset_samples,
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--class-name", default="digital_clock")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default=None)
    return parser


def _predict_one(model: object, image: np.ndarray) -> int:
    """Predict one normalized HWC image."""
    batch = np.asarray(image, dtype=np.float32).reshape((1,) + image.shape)
    label = model.predict_label(batch)
    return int(np.asarray(label).reshape(-1)[0])


def inspect_predictions(
    model: object,
    samples: Iterable[object],
) -> list[dict[str, object]]:
    """Return one prediction row for each sample."""
    rows = []
    for sample in samples:
        clean_pred = _predict_one(model, sample.image)
        rows.append(
            {
                "image_id": sample.image_id,
                "class_name": sample.class_name,
                "configured_label": int(sample.true_label),
                "clean_pred": int(clean_pred),
                "matches_configured_label": bool(clean_pred == int(sample.true_label)),
            }
        )
    return rows


def write_prediction_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write prediction rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "class_name",
        "configured_label",
        "clean_pred",
        "matches_configured_label",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _filtered_config(config: dict, class_name: str, data_root: Optional[str]) -> dict:
    """Return a config restricted to one ImageNet class folder."""
    filtered = dict(config)
    dataset = dict(filtered.get("dataset", {}))
    class_indices = dataset.get("class_indices", {})
    if class_name not in class_indices:
        raise ValueError("Unknown class_name {0!r} in dataset.class_indices.".format(class_name))
    dataset["class_indices"] = {class_name: class_indices[class_name]}
    if data_root is not None:
        dataset["images_dir"] = data_root
    filtered["dataset"] = dataset
    return filtered


def main() -> int:
    """Run clean inference on one class folder and print prediction counts."""
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    model = build_model(config)
    filtered_config = _filtered_config(
        config=config,
        class_name=str(args.class_name),
        data_root=args.data_root,
    )
    samples = load_subset_samples(filtered_config, limit_override=args.limit)
    rows = inspect_predictions(model=model, samples=samples)
    counts = Counter(int(row["clean_pred"]) for row in rows)

    print("n_images={0}".format(len(rows)))
    print("configured_label={0}".format(filtered_config["dataset"]["class_indices"][args.class_name]))
    for label, count in counts.most_common():
        print("pred_label={0}, count={1}".format(label, count))

    if args.output:
        output_path = write_prediction_csv(Path(args.output), rows)
        print("output_csv={0}".format(output_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
