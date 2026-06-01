"""Materialize the ImageNet subset used by Table 10 Inception v3.

The generated directory is intentionally under ``data/`` and should stay out of
git. Run this before the Inception v3 CW experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepdetector.data.imagenet_subset import materialize_class_subset


DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "imagenet" / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "inceptionV3"
CLASS_ORDER = ("zebra", "panda", "cab")
CLASS_INDICES = {"zebra": 80, "panda": 169, "cab": 267}
CLASS_QUOTAS = {"zebra": 40, "panda": 40, "cab": 20}


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate managed class folders if the subset already exists.",
    )
    return parser


def main() -> int:
    """Create the deterministic Inception v3 subset."""
    args = build_parser().parse_args()
    subset = materialize_class_subset(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        class_order=CLASS_ORDER,
        class_indices=CLASS_INDICES,
        class_quotas=CLASS_QUOTAS,
        overwrite=bool(args.force),
    )
    print("Materialized Inception v3 subset: {0}".format(subset.output_dir))
    for class_name in CLASS_ORDER:
        print("{0}: {1}".format(class_name, subset.class_counts[class_name]))
    print("total: {0}".format(len(subset.rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
