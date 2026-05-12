"""Check whether the Carlini MNIST M2 weights are available and loadable."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mnist.m2 import check_m2


def main() -> int:
    result = check_m2()
    print("m2_weights_path={}".format(result.weights_path))
    print("m2_exists={}".format(result.exists))
    print("m2_file_size={}".format(result.file_size))
    print("m2_loaded={}".format(result.loaded))
    return 0 if result.loaded else 1


if __name__ == "__main__":
    raise SystemExit(main())

