"""Reproduce DeepDetector Table 6 for adaptive MNIST quantization."""

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
    adaptive_quantization_filter,
    close_graph,
    create_restored_mnist_graph,
    ensure_output_dir,
    evaluate_filter_on_images,
    format_percent,
    load_mnist_test_slice,
    percent_delta,
    write_csv,
    write_markdown_table,
)


ARTICLE_TABLE_6 = {
    "Training": {"TP": 3482, "FN": 370, "FP": 146, "recall": 90.39, "precision": 95.98, "f1": 93.10},
    "Validation": {"TP": 939, "FN": 81, "FP": 48, "recall": 92.06, "precision": 95.14, "f1": 93.57},
}


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=str(PROJECT_ROOT / "results" / "mnist" / "clean_baseline" / "checkpoints"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / ARTICLE_OUTPUT_DIR))
    parser.add_argument("--epsilon", type=float, default=0.2)
    return parser


def main() -> int:
    """Run the Table 6 reproduction."""
    args = build_parser().parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    graph = create_restored_mnist_graph(args.train_dir)
    rows = []

    try:
        splits = [
            ("Training", 0, 4500),
            ("Validation", 4500, 5500),
        ]
        for split_name, start, end in splits:
            images, labels = load_mnist_test_slice(start, end)
            metrics = evaluate_filter_on_images(
                graph=graph,
                images=images,
                labels=labels,
                epsilon=args.epsilon,
                filter_fn=adaptive_quantization_filter,
            )
            article = ARTICLE_TABLE_6[split_name]
            metrics.update(
                {
                    "split": split_name,
                    "start": start,
                    "end": end,
                    "epsilon": args.epsilon,
                    "article_TP": article["TP"],
                    "article_FN": article["FN"],
                    "article_FP": article["FP"],
                    "article_recall_percent": article["recall"],
                    "article_precision_percent": article["precision"],
                    "article_f1_percent": article["f1"],
                }
            )
            rows.append(metrics)
    finally:
        close_graph(graph)

    csv_path = write_csv(
        str(Path(output_dir) / "table_6_adaptive_quantization.csv"),
        rows,
        [
            "split",
            "start",
            "end",
            "epsilon",
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
                row["split"],
                row["article_TP"],
                row["TP"],
                row["article_FN"],
                row["FN"],
                row["article_FP"],
                row["FP"],
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
        str(Path(output_dir) / "table_6_adaptive_quantization.md"),
        "Table 6 - Adaptive Quantization",
        [
            "Split",
            "Article TP",
            "Our TP",
            "Article FN",
            "Our FN",
            "Article FP",
            "Our FP",
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
            "Article values are from Liang et al., Table 6.",
            "Training and Validation name the article's MNIST test-set slices 0-4499 and 4500-5499.",
            "Adaptive quantization maps H < 4 to 2 intervals, 4 <= H < 5 to 4 intervals, and H >= 5 to 6 intervals.",
        ],
    )

    print("results_csv={0}".format(csv_path))
    print("comparison_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
