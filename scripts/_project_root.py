"""Helpers for running scripts from any subdirectory under ``scripts``."""

import logging
import os
from pathlib import Path
import sys
from typing import Union
import warnings


PathLike = Union[str, Path]


def find_project_root(start: PathLike) -> Path:
    """Find the project root by walking up from ``start``."""
    path = Path(start).resolve()
    current = path if path.is_dir() else path.parent

    for candidate in [current] + list(current.parents):
        if (candidate / "src" / "deepdetector").is_dir() and (
            candidate / "environment.yml"
        ).is_file():
            return candidate

    raise RuntimeError("Could not find project root from {0}".format(path))


def configure_project_paths(start: PathLike) -> Path:
    """Add ``src`` to ``sys.path`` and return the project root."""
    configure_legacy_ml_output()
    project_root = find_project_root(start)
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return project_root


def configure_legacy_ml_output() -> None:
    """Keep expected TF1/Keras/CleverHans deprecation noise out of script logs."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*cleverhans.utils_mnist.*")
    warnings.filterwarnings("ignore", message=".*is deprecated.*")
    warnings.filterwarnings("ignore", message=".*is deprecrated.*")
