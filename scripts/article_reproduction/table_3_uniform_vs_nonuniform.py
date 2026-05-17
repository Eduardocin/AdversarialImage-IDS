"""Reproduce DeepDetector Table 3 for MNIST uniform vs non-uniform quantization."""

from __future__ import print_function

import argparse
from pathlib import Path
import sys


SCRIPTS_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "_project_root.py").is_file()
)
sys.path.insert(0, str(SCRIPTS_ROOT))
from _project_root import configure_project_paths

PROJECT_ROOT = configure_project_paths(__file__)

from deepdetector.evaluation.article_reproduction import (  # noqa: E402
    ARTICLE_OUTPUT_DIR,
    close_graph,
    create_restored_mnist_graph,
    ensure_output_dir,
    evaluate_filter_on_images,
    format_percent,
    load_mnist_test_slice,
    nonuniform_quantization,
    percent_delta,
    scalar_filter_for_intervals,
    time_filter_application,
    write_csv,
    write_markdown_table,
)


ARTICLE_ROWS = {
    "Uniform": {"time_seconds": 0.004, "recall": 94.00, "precision": 100.00, "f1": 96.91},
    "Non-uniform": {"time_seconds": 137.7, "recall": 94.00, "precision": 51.65, "f1": 66.67},
}


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=str(PROJECT_ROOT / "results" / "mnist" / "clean_baseline" / "checkpoints"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / ARTICLE_OUTPUT_DIR))
    parser.add_argument("--epsilon", type=float, default=0.2)
    return parser


def main() -> int:
    """Run the Table 3 reproduction."""
    args = build_parser().parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    images, labels = load_mnist_test_slice(0, 100)
    graph = create_restored_mnist_graph(args.train_dir)
    rows = []

    try:
        filters = [
            ("Uniform", scalar_filter_for_intervals(2)),
            ("Non-uniform", nonuniform_quantization),
        ]
        for name, filter_fn in filters:
            metrics = evaluate_filter_on_images(
                graph=graph,
                images=images,
                labels=labels,
                epsilon=args.epsilon,
                filter_fn=filter_fn,
            )
            article = ARTICLE_ROWS[name]
            metrics.update(
                {
                    "quantization": name,
                    "epsilon": args.epsilon,
                    "time_seconds": time_filter_application(filter_fn, images),
                    "article_time_seconds": article["time_seconds"],
                    "article_recall_percent": article["recall"],
                    "article_precision_percent": article["precision"],
                    "article_f1_percent": article["f1"],
                }
            )
            rows.append(metrics)
    finally:
        close_graph(graph)

    csv_path = write_csv(
        str(Path(output_dir) / "table_3_uniform_vs_nonuniform.csv"),
        rows,
        [
            "quantization",
            "epsilon",
            "time_seconds",
            "F",
            "TP",
            "FN",
            "FP",
            "recall_percent",
            "precision_percent",
            "f1_percent",
        ],
    )

    markdown_rows = []
    for row in rows:
        markdown_rows.append(
            [
                row["quantization"],
                "{0:.4f}".format(float(row["article_time_seconds"])),
                "{0:.4f}".format(float(row["time_seconds"])),
                format_percent(row["article_recall_percent"]),
                format_percent(row["recall_percent"]),
                "{0:+.2f}".format(percent_delta(row["recall_percent"], row["article_recall_percent"])),
                format_percent(row["article_precision_percent"]),
                format_percent(row["precision_percent"]),
                "{0:+.2f}".format(percent_delta(row["precision_percent"], row["article_precision_percent"])),
                format_percent(row["article_f1_percent"]),
                format_percent(row["f1_percent"]),
                "{0:+.2f}".format(percent_delta(row["f1_percent"], row["article_f1_percent"])),
            ]
        )

    md_path = write_markdown_table(
        str(Path(output_dir) / "table_3_uniform_vs_nonuniform.md"),
        "Table 3 - Uniform vs Non-uniform Quantization",
        [
            "Quantization",
            "Article time",
            "Our time",
            "Article recall",
            "Our recall",
            "Delta recall",
            "Article precision",
            "Our precision",
            "Delta precision",
            "Article F1",
            "Our F1",
            "Delta F1",
        ],
        markdown_rows,
        notes=[
            "Article values are from Liang et al., Table 3.",
            "Our timing measures NumPy filter application on the 100 selected MNIST test digits.",
        ],
    )

    print("results_csv={0}".format(csv_path))
    print("comparison_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
