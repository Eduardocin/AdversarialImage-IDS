"""Check legacy MNIST dependencies for the M1/M2 replication."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mnist.environment import main


if __name__ == "__main__":
    raise SystemExit(main())
