"""Compare scalar quantization intervals on MNIST adversarial samples."""

from __future__ import print_function

import argparse
from pathlib import Path


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())

from deepdetector.evaluation.article_reproduction import (  # noqa: E402
    ARTICLE_OUTPUT_DIR,
    close_graph,
    create_restored_mnist_graph,
    ensure_output_dir,
    evaluate_filter_on_existing_adversarial,
    format_percent,
    interval_size,
    load_mnist_test_slice,
    percent_delta,
    predict_labels,
    scalar_filter_for_intervals,
    write_csv,
    write_markdown_table,
)
from deepdetector.attacks.fgsm import generate_fgsm_examples  # noqa: E402


ARTICLE_TABLE_4 = {
    2: {"recall": 93.86, "precision": 99.14, "f1": 96.43},
    3: {"recall": 86.48, "precision": 99.66, "f1": 92.60},
    4: {"recall": 93.04, "precision": 99.90, "f1": 96.35},
    5: {"recall": 13.26, "precision": 99.51, "f1": 23.40},
    6: {"recall": 7.25, "precision": 99.11, "f1": 13.51},
    7: {"recall": 24.95, "precision": 99.74, "f1": 39.92},
    8: {"recall": 42.03, "precision": 99.77, "f1": 59.15},
    9: {"recall": 50.62, "precision": 100.00, "f1": 67.22},
    10: {"recall": 52.35, "precision": 100.00, "f1": 68.72},
}


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=str(PROJECT_ROOT / "results" / "mnist" / "clean_baseline" / "checkpoints"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / ARTICLE_OUTPUT_DIR))
    parser.add_argument("--epsilon", type=float, default=0.2)
    return parser


def main() -> int:
    """Run the interval comparison and write CSV and Markdown outputs."""
    args = build_parser().parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    images, labels = load_mnist_test_slice(0, 4500)
    graph = create_restored_mnist_graph(args.train_dir)
    rows = []

    try:
        adv_images = generate_fgsm_examples(
            sess=graph["sess"],
            model=graph["model"],
            x_placeholder=graph["x"],
            images=images,
            eps=args.epsilon,
            clip_min=0.0,
            clip_max=1.0,
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
        for intervals in range(2, 11):
            metrics = evaluate_filter_on_existing_adversarial(
                graph=graph,
                images=images,
                labels=labels,
                adv_images=adv_images,
                clean_pred=clean_pred,
                adv_pred=adv_pred,
                filter_fn=scalar_filter_for_intervals(intervals),
            )
            article = ARTICLE_TABLE_4[intervals]
            metrics.update(
                {
                    "intervals": intervals,
                    "interval_size": interval_size(intervals),
                    "epsilon": args.epsilon,
                    "article_recall_percent": article["recall"],
                    "article_precision_percent": article["precision"],
                    "article_f1_percent": article["f1"],
                }
            )
            rows.append(metrics)
    finally:
        close_graph(graph)

    csv_path = write_csv(
        str(Path(output_dir) / "table_4_scalar_quantization_intervals.csv"),
        rows,
        [
            "intervals",
            "interval_size",
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
                row["intervals"],
                row["interval_size"],
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
        str(Path(output_dir) / "table_4_scalar_quantization_intervals.md"),
        "Scalar Quantization Intervals",
        [
            "Intervals",
            "Interval size",
            "Reference recall",
            "Our recall",
            "Delta recall",
            "Reference precision",
            "Our precision",
            "Delta precision",
            "Reference F1",
            "Our F1",
            "Delta F1",
        ],
        markdown_rows,
        notes=[
            "Reference values use the fixed comparison targets configured in this script.",
            "This run uses MNIST test digits 0-4499 and FGSM epsilon 0.2.",
        ],
    )

    print("results_csv={0}".format(csv_path))
    print("comparison_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

