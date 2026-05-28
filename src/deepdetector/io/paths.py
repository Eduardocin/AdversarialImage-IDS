"""Project path helpers for experiment scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def get_project_root() -> Path:
    """Return the repository root containing pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not find project root containing pyproject.toml.")


def resolve_project_path(path_value: Optional[Union[str, Path]]) -> Optional[Path]:
    """Resolve a path relative to the project root when it is not absolute."""
    if path_value in (None, ""):
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path
    return get_project_root() / path


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory if needed and return it as Path."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

