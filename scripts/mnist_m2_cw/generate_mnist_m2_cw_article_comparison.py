"""Generate MNIST M2 CW metric comparison artifacts."""

from __future__ import print_function

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd


SCRIPTS_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "_project_root.py").is_file()
)
sys.path.insert(0, str(SCRIPTS_ROOT))
from _project_root import configure_project_paths

PROJECT_ROOT = configure_project_paths(__file__)


M2_CW_DIR = PROJECT_ROOT / "results" / "mnist" / "m2_cw"
DETECTOR_DIR = M2_CW_DIR / "detector"
OUTPUT_DIR = M2_CW_DIR / "article_comparison"


ARTICLE_ROWS = [
    {
        "row_id": "cw_l2_kappa_0p0",
        "experiment": "mnist_m2_cw",
        "attack": "CW",
        "norm": "L2",
        "kappa": 0.0,
        "article_F": 0,
        "article_TP": 984,
        "article_FN": 11,
        "article_FP": 9,
        "article_RTP": 919,
        "article_RTP_percent": 93.39,
        "article_recall": 98.89,
        "article_precision": 99.09,
        "article_f1": 98.99,
    },
    {
        "row_id": "cw_l2_kappa_0p5",
        "experiment": "mnist_m2_cw",
        "attack": "CW",
        "norm": "L2",
        "kappa": 0.5,
        "article_F": 0,
        "article_TP": 984,
        "article_FN": 11,
        "article_FP": 9,
        "article_RTP": 920,
        "article_RTP_percent": 93.50,
        "article_recall": 98.89,
        "article_precision": 99.09,
        "article_f1": 98.99,
    },
    {
        "row_id": "cw_l2_kappa_1p0",
        "experiment": "mnist_m2_cw",
        "attack": "CW",
        "norm": "L2",
        "kappa": 1.0,
        "article_F": 0,
        "article_TP": 983,
        "article_FN": 12,
        "article_FP": 9,
        "article_RTP": 913,
        "article_RTP_percent": 92.88,
        "article_recall": 98.79,
        "article_precision": 99.09,
        "article_f1": 98.94,
    },
    {
        "row_id": "cw_l2_kappa_2p0",
        "experiment": "mnist_m2_cw",
        "attack": "CW",
        "norm": "L2",
        "kappa": 2.0,
        "article_F": 0,
        "article_TP": 979,
        "article_FN": 16,
        "article_FP": 9,
        "article_RTP": 897,
        "article_RTP_percent": 91.62,
        "article_recall": 98.39,
        "article_precision": 99.09,
        "article_f1": 98.74,
    },
    {
        "row_id": "cw_l2_kappa_4p0",
        "experiment": "mnist_m2_cw",
        "attack": "CW",
        "norm": "L2",
        "kappa": 4.0,
        "article_F": 0,
        "article_TP": 959,
        "article_FN": 36,
        "article_FP": 9,
        "article_RTP": 866,
        "article_RTP_percent": 90.30,
        "article_recall": 96.38,
        "article_precision": 99.07,
        "article_f1": 97.71,
    },
    {
        "row_id": "cw_linf",
        "experiment": "mnist_m2_cw",
        "attack": "CW",
        "norm": "Linf",
        "kappa": "",
        "article_F": 0,
        "article_TP": 991,
        "article_FN": 7,
        "article_FP": 1,
        "article_RTP": 991,
        "article_RTP_percent": 100.00,
        "article_recall": 99.30,
        "article_precision": 99.90,
        "article_f1": 99.60,
    },
]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    return argparse.ArgumentParser(
        description=(
            "Generate MNIST M2 CW comparison artifacts. Percent metrics use "
            "the 0-100 scale."
        )
    )

OUTPUT_COLUMNS = [
    "row_id",
    "experiment",
    "article_F",
    "article_TP",
    "article_FN",
    "article_FP",
    "article_RTP",
    "article_RTP_percent",
    "article_recall",
    "article_precision",
    "article_f1",
    "our_F",
    "our_TP",
    "our_FN",
    "our_FP",
    "our_RTP",
    "our_RTP_percent",
    "our_recall",
    "our_precision",
    "our_f1",
    "abs_diff_recall",
    "abs_diff_precision",
    "abs_diff_f1",
    "possible_explanation",
]


def _read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV when it exists and has rows."""
    if not path.exists():
        return None
    df = pd.read_csv(str(path))
    if df.empty:
        return None
    return df


def _matching_row(ours: Optional[pd.DataFrame], article_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the matching detector row, if present."""
    if ours is None:
        return None
    subset = ours[ours["norm"].astype(str).str.lower() == str(article_row["norm"]).lower()]
    if str(article_row["norm"]).lower() == "l2":
        subset = subset[subset["kappa"].astype(float) == float(article_row["kappa"])]
    if subset.empty:
        return None
    return subset.iloc[0].to_dict()


def _get_metric(ours_row: Optional[Dict[str, Any]], name: str) -> Any:
    """Return the requested metric from our detector row with fallbacks."""
    if ours_row is None:
        return ""
    if name in ours_row:
        return ours_row.get(name, "")
    if name == "F" and "#F" in ours_row:
        return ours_row.get("#F", "")
    return ""


def build_comparison() -> pd.DataFrame:
    """Load detector CSVs and return one comparison row per attack setting."""
    l2 = _read_optional_csv(DETECTOR_DIR / "cw_l2_detector_results.csv")
    linf = _read_optional_csv(DETECTOR_DIR / "cw_linf_detector_results.csv")
    ours = pd.concat([df for df in [l2, linf] if df is not None], ignore_index=True) if any(
        df is not None for df in [l2, linf]
    ) else None

    rows = []
    for article_row in ARTICLE_ROWS:
        row = dict(article_row)
        ours_row = _matching_row(ours, article_row)
        for field in ["F", "TP", "FN", "FP", "RTP", "RTP_percent", "recall", "precision", "f1"]:
            row["our_{0}".format(field)] = _get_metric(ours_row, field)
        for field in ["recall", "precision", "f1"]:
            our_value = row["our_{0}".format(field)]
            article_value = row["article_{0}".format(field)]
            if our_value == "" or pd.isna(our_value):
                row["abs_diff_{0}".format(field)] = ""
            else:
                row["abs_diff_{0}".format(field)] = abs(float(our_value) - float(article_value))
        if ours_row is None:
            row["possible_explanation"] = "Detector results not available yet."
        elif str(article_row["norm"]).lower() == "linf" and str(ours_row.get("notes", "")).startswith("not_executed"):
            row["possible_explanation"] = "CW Linf was not executed in this legacy stack."
        else:
            row["possible_explanation"] = (
                "Differences may reflect the approximate M2 architecture, CW parameters, "
                "library versions, and filter implementation."
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, path: Path) -> Path:
    """Write a Markdown table."""
    lines = ["# MNIST M2 CW Comparison", ""]
    headers = list(df.columns)
    lines.append("| {0} |".format(" | ".join(headers)))
    lines.append("| {0} |".format(" | ".join(["---"] * len(headers))))
    for _, row in df.iterrows():
        values = ["" if pd.isna(row[column]) else str(row[column]) for column in headers]
        lines.append("| {0} |".format(" | ".join(values)))
    lines.append("")
    lines.append("Percent metrics use the 0-100 scale.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _plot_l2_metric(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Plot reference and local L2 metric values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    l2 = df[df["norm"] == "L2"].copy()
    kappas = l2["kappa"].astype(float).values
    article_values = l2["article_{0}".format(metric)].astype(float).values
    our_values = pd.to_numeric(l2["our_{0}".format(metric)], errors="coerce").values

    plt.figure(figsize=(7, 4))
    plt.plot(kappas, article_values, marker="o", label="Reference")
    if not pd.isna(our_values).all():
        plt.plot(kappas, our_values, marker="s", label="Ours")
    plt.xlabel("CW L2 kappa")
    plt.ylabel(metric.replace("_", " "))
    plt.title("MNIST M2 CW L2 {0}".format(metric.replace("_", " ")))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def main() -> int:
    """Generate comparison CSV, Markdown, and plots."""
    build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_comparison()
    df_output = df[OUTPUT_COLUMNS]
    csv_path = OUTPUT_DIR / "table_10_m2_cw_comparison.csv"
    md_path = OUTPUT_DIR / "table_10_m2_cw_comparison.md"
    df_output.to_csv(str(csv_path), index=False)
    write_markdown(df_output, md_path)
    _plot_l2_metric(df, "recall", OUTPUT_DIR / "cw_l2_recall_comparison.png")
    _plot_l2_metric(df, "precision", OUTPUT_DIR / "cw_l2_precision_comparison.png")
    _plot_l2_metric(df, "f1", OUTPUT_DIR / "cw_l2_f1_comparison.png")
    _plot_l2_metric(df, "RTP_percent", OUTPUT_DIR / "cw_l2_rtp_comparison.png")
    print("comparison_csv={0}".format(csv_path))
    print("comparison_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
