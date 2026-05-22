"""Compare uniform and non-uniform quantization on MNIST samples."""

from __future__ import print_function

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mnist_table_3.yaml"

from deepdetector.evaluation.article_reproduction import (  # noqa: E402
    ARTICLE_OUTPUT_DIR,
    close_graph,
    create_restored_mnist_graph,
    ensure_output_dir,
    evaluate_filter_on_images,
    format_percent,
    load_mnist_test_slice,
    nonuniform_quantization_legacy,
    nonuniform_quantization,
    scalar_filter_for_intervals,
    time_filter_application,
    write_csv,
    write_markdown_table,
)
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


def configured_filters(config: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    """Build quantization filters from the experiment config."""
    filter_configs = config.get("detection", {}).get("filters", [])
    if not filter_configs:
        raise ValueError("Config must define detection.filters.")

    for filter_config in filter_configs:
        name = str(filter_config.get("name"))
        method = str(filter_config.get("method"))
        if method == "scalar_quantization":
            intervals = int(filter_config.get("intervals", 2))
            yield name, scalar_filter_for_intervals(intervals)
        elif method == "nonuniform_quantization":
            yield name, nonuniform_quantization
        elif method == "nonuniform_quantization_legacy":
            yield name, nonuniform_quantization_legacy
        else:
            raise ValueError("Unknown quantization method: {0}".format(method))


def main() -> int:
    """Run the quantization experiment and write CSV and Markdown outputs."""
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
    exclude_invalid_pairs = bool(evaluation_config.get("exclude_invalid_pairs", True))
    test_start = int(dataset_config.get("start", 0))
    test_end = int(dataset_config.get("end", 100))

    images, labels = load_mnist_test_slice(test_start, test_end)
    graph = create_restored_mnist_graph(train_dir)
    rows = []

    try:
        for name, filter_fn in configured_filters(config):
            metrics = evaluate_filter_on_images(
                graph=graph,
                images=images,
                labels=labels,
                epsilon=epsilon,
                filter_fn=filter_fn,
                clip_min=clip_min,
                clip_max=clip_max,
                exclude_invalid_pairs=exclude_invalid_pairs,
            )
            metrics.update(
                {
                    "quantization": name,
                    "time_seconds": time_filter_application(filter_fn, images),
                }
            )
            rows.append(metrics)
    finally:
        close_graph(graph)

    csv_path = write_csv(
        str(
            Path(output_dir)
            / str(output_config.get("csv", "table_3_uniform_vs_nonuniform.csv"))
        ),
        rows,
        metrics_config.get(
            "raw_fields",
            [
                "quantization",
                "time_seconds",
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
                row["quantization"],
                "{0:.4f}".format(float(row["time_seconds"])),
                format_percent(row["recall_percent"]),
                format_percent(row["precision_percent"]),
                format_percent(row["f1_percent"]),
            ]
        )

    md_path = write_markdown_table(
        str(
            Path(output_dir)
            / str(output_config.get("markdown", "table_3_uniform_vs_nonuniform.md"))
        ),
        "Uniform vs Non-uniform Quantization",
        metrics_config.get(
            "columns",
            ["Quantização", "Time(s)", "Recall", "Precision", "F1"],
        ),
        markdown_rows,
    )

    print("results_csv={0}".format(csv_path))
    print("results_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

