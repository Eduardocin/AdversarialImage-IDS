"""Compare scalar quantization intervals on MNIST adversarial samples."""

from __future__ import print_function

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "article_reproduction" / "mnist_table_4.yaml"

from deepdetector.evaluation.article_reproduction import (  # noqa: E402
    ARTICLE_OUTPUT_DIR,
    close_graph,
    create_restored_mnist_graph,
    ensure_output_dir,
    evaluate_filter_on_existing_adversarial,
    format_percent,
    interval_size,
    load_mnist_test_slice,
    predict_labels,
    scalar_filter_for_intervals,
    write_csv,
    write_markdown_table,
)
from deepdetector.attacks.fgsm import generate_fgsm_examples  # noqa: E402
from deepdetector.paths import MNIST_M1_CHECKPOINT_DIR  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    return parser


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve a config path relative to the project root."""
    if path_value in (None, ""):
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: Path) -> Dict[str, Any]:
    """Load the experiment YAML config."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")
    return config


def configured_intervals(config: Dict[str, Any]) -> Iterable[int]:
    """Return scalar quantization interval counts from config."""
    intervals = config.get("quantization", {}).get("intervals", [])
    if not intervals:
        raise ValueError("Config must define quantization.intervals.")
    for value in intervals:
        yield int(value)


def main() -> int:
    """Run the interval comparison and write CSV and Markdown outputs."""
    args = build_parser().parse_args()
    config = load_config(_resolve_path(args.config) or DEFAULT_CONFIG)

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    attack_config = config.get("attack", {})
    evaluation_config = config.get("evaluation", {})
    metrics_config = config.get("metrics", {})
    output_config = config.get("output", {})

    configured_output_dir = _resolve_path(args.output_dir or output_config.get("results_dir"))
    output_dir = ensure_output_dir(
        str(configured_output_dir or (PROJECT_ROOT / ARTICLE_OUTPUT_DIR))
    )
    configured_train_dir = _resolve_path(args.train_dir or model_config.get("checkpoint_dir"))
    train_dir = str(
        configured_train_dir
        or MNIST_M1_CHECKPOINT_DIR
    )
    epsilon = float(
        args.epsilon if args.epsilon is not None else attack_config.get("epsilon", 0.2)
    )
    clip_min = float(attack_config.get("clip_min", 0.0))
    clip_max = float(attack_config.get("clip_max", 1.0))
    exclude_invalid_pairs = bool(evaluation_config.get("exclude_invalid_pairs", False))
    test_start = int(dataset_config.get("start", 0))
    test_end = int(dataset_config.get("end", 4500))

    images, labels = load_mnist_test_slice(test_start, test_end)
    graph = create_restored_mnist_graph(train_dir)
    rows = []

    try:
        adv_images = generate_fgsm_examples(
            sess=graph["sess"],
            model=graph["model"],
            x_placeholder=graph["x"],
            images=images,
            eps=epsilon,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        clean_pred = predict_labels(
            graph["sess"],
            graph["x"],
            graph["predictions"],
            images,
        )
        adv_pred = predict_labels(
            graph["sess"],
            graph["x"],
            graph["predictions"],
            adv_images,
        )
        for intervals in configured_intervals(config):
            metrics = evaluate_filter_on_existing_adversarial(
                graph=graph,
                images=images,
                labels=labels,
                adv_images=adv_images,
                clean_pred=clean_pred,
                adv_pred=adv_pred,
                filter_fn=scalar_filter_for_intervals(intervals),
                exclude_invalid_pairs=exclude_invalid_pairs,
            )
            metrics.update(
                {
                    "intervals": intervals,
                    "interval_size": interval_size(intervals),
                }
            )
            rows.append(metrics)
    finally:
        close_graph(graph)

    csv_path = write_csv(
        str(
            Path(output_dir)
            / str(output_config.get("csv", "table_4_scalar_quantization_intervals.csv"))
        ),
        rows,
        metrics_config.get(
            "raw_fields",
            [
                "intervals",
                "interval_size",
                "recall_percent",
                "precision_percent",
                "f1_percent",
            ],
        ),
    )

    markdown_rows = []
    for row in rows:
        markdown_rows.append(
            [
                row["intervals"],
                row["interval_size"],
                format_percent(row["recall_percent"]),
                format_percent(row["precision_percent"]),
                format_percent(row["f1_percent"]),
            ]
        )

    md_path = write_markdown_table(
        str(
            Path(output_dir)
            / str(output_config.get("markdown", "table_4_scalar_quantization_intervals.md"))
        ),
        "Scalar Quantization Intervals",
        metrics_config.get(
            "columns",
            ["Intervals", "Interval size", "Recall", "Precision", "F1"],
        ),
        markdown_rows,
    )

    print("results_csv={0}".format(csv_path))
    print("results_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

