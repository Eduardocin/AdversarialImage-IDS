"""Evaluate MNIST FGSM detector metrics for fixed epsilon values."""

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
    percent_delta,
    proposed_detection_filter,
    write_csv,
    write_markdown_table,
)


ARTICLE_TABLE_10 = {
    0.1: {"F": 4026, "TP": 410, "FN": 40, "FP": 24, "RTP": 398, "RTP_percent": 97.07, "recall": 91.11, "precision": 94.47, "f1": 92.76},
    0.2: {"F": 1910, "TP": 2467, "FN": 106, "FP": 32, "RTP": 2430, "RTP_percent": 98.50, "recall": 95.88, "precision": 98.72, "f1": 97.28},
    0.3: {"F": 455, "TP": 3856, "FN": 172, "FP": 32, "RTP": 3768, "RTP_percent": 97.71, "recall": 95.73, "precision": 99.18, "f1": 97.42},
    0.4: {"F": 132, "TP": 4078, "FN": 273, "FP": 32, "RTP": 3820, "RTP_percent": 93.67, "recall": 93.73, "precision": 99.22, "f1": 96.40},
}


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=str(PROJECT_ROOT / "results" / "mnist" / "clean_baseline" / "checkpoints"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / ARTICLE_OUTPUT_DIR))
    return parser


def main() -> int:
    """Run FGSM detector evaluation and write comparison outputs."""
    args = build_parser().parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    images, labels = load_mnist_test_slice(5500, 10000)
    graph = create_restored_mnist_graph(args.train_dir)
    rows = []

    try:
        for epsilon in [0.1, 0.2, 0.3, 0.4]:
            metrics = evaluate_filter_on_images(
                graph=graph,
                images=images,
                labels=labels,
                epsilon=epsilon,
                filter_fn=proposed_detection_filter,
            )
            article = ARTICLE_TABLE_10[epsilon]
            metrics.update(
                {
                    "row": int(epsilon * 10),
                    "attack_model": "FGSM (eps={0:g})/M1".format(epsilon),
                    "dataset": "MNIST",
                    "epsilon": epsilon,
                    "article_F": article["F"],
                    "article_TP": article["TP"],
                    "article_FN": article["FN"],
                    "article_FP": article["FP"],
                    "article_RTP": article["RTP"],
                    "article_RTP_percent": article["RTP_percent"],
                    "article_recall_percent": article["recall"],
                    "article_precision_percent": article["precision"],
                    "article_f1_percent": article["f1"],
                }
            )
            rows.append(metrics)
    finally:
        close_graph(graph)

    csv_path = write_csv(
        str(Path(output_dir) / "table_10_mnist_fgsm_test.csv"),
        rows,
        [
            "epsilon",
            "F",
            "TP",
            "FN",
            "FP",
            "RTP",
            "RTP_percent",
            "recall_percent",
            "precision_percent",
            "f1_percent",
        ],
    )

    markdown_rows = []
    for row in rows:
        markdown_rows.append(
            [
                row["attack_model"],
                row["article_F"],
                row["F"],
                row["article_TP"],
                row["TP"],
                row["article_FN"],
                row["FN"],
                row["article_FP"],
                row["FP"],
                row["article_RTP"],
                row["RTP"],
                format_percent(row["article_RTP_percent"]),
                format_percent(row["RTP_percent"]),
                format_percent(row["article_recall_percent"]),
                format_percent(row["recall_percent"]),
                format_percent(row["article_precision_percent"]),
                format_percent(row["precision_percent"]),
                format_percent(row["article_f1_percent"]),
                format_percent(row["f1_percent"]),
                "{0:+.2f}".format(percent_delta(row["f1_percent"], row["article_f1_percent"])),
            ]
        )

    md_path = write_markdown_table(
        str(Path(output_dir) / "table_10_mnist_fgsm_test.md"),
        "MNIST FGSM Detector Metrics",
        [
            "Attack/Model",
            "Reference #F",
            "Our #F",
            "Reference TP",
            "Our TP",
            "Reference FN",
            "Our FN",
            "Reference FP",
            "Our FP",
            "Reference RTP",
            "Our RTP",
            "Reference RTP%",
            "Our RTP%",
            "Reference recall",
            "Our recall",
            "Reference precision",
            "Our precision",
            "Reference F1",
            "Our F1",
            "Delta F1",
        ],
        markdown_rows,
        notes=[
            "Reference values use the fixed comparison targets configured in this script.",
            "#F denotes samples whose FGSM image is still classified as the true label.",
            "This run uses MNIST test digits 5500-9999 and the final entropy-aware detection filter.",
        ],
    )

    print("results_csv={0}".format(csv_path))
    print("comparison_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
