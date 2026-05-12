"""Train Carlini's MNIST M2 model for the DeepDetector C&W baseline."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mnist.m2 import main


if __name__ == "__main__":
    raise SystemExit(main())
