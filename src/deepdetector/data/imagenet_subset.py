"""Utilities for materializing deterministic ImageNet class subsets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence, Union


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ImageNetSubsetRow:
    """One selected ImageNet image and its class metadata."""

    path: Path
    class_name: str
    label_index: int


@dataclass(frozen=True)
class MaterializedImageNetSubset:
    """Summary of a copied ImageNet subset."""

    source_dir: Path
    output_dir: Path
    rows: tuple[ImageNetSubsetRow, ...]
    class_counts: dict[str, int]


def _image_files(class_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(class_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def select_class_subset(
    *,
    source_dir: Path,
    class_order: Sequence[str],
    class_indices: Mapping[str, int],
    class_quotas: Mapping[str, int],
) -> tuple[ImageNetSubsetRow, ...]:
    """Select deterministic per-class ImageNet rows from class folders."""
    rows: list[ImageNetSubsetRow] = []
    for class_name in class_order:
        if class_name not in class_indices:
            raise ValueError("Missing class index for {0}.".format(class_name))
        if class_name not in class_quotas:
            raise ValueError("Missing class quota for {0}.".format(class_name))

        quota = int(class_quotas[class_name])
        if quota < 0:
            raise ValueError("Class quota must be non-negative for {0}.".format(class_name))

        class_dir = source_dir / class_name
        if not class_dir.is_dir():
            raise ValueError("Missing ImageNet class directory: {0}".format(class_dir))

        files = _image_files(class_dir)
        if len(files) < quota:
            raise ValueError(
                "Class {0} has {1} images; quota requires {2}.".format(
                    class_name,
                    len(files),
                    quota,
                )
            )

        rows.extend(
            ImageNetSubsetRow(
                path=path,
                class_name=class_name,
                label_index=int(class_indices[class_name]),
            )
            for path in files[:quota]
        )
    return tuple(rows)


def _prepare_output_dir(
    output_dir: Path,
    class_order: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return

    if not output_dir.is_dir():
        raise ValueError("Output path is not a directory: {0}".format(output_dir))

    managed_paths = [output_dir / class_name for class_name in class_order]
    manifest_path = output_dir / "manifest.json"
    has_existing_subset_files = any(
        path.exists() and any(path.iterdir())
        for path in managed_paths
        if path.is_dir()
    )
    if manifest_path.exists():
        has_existing_subset_files = True

    if has_existing_subset_files and not overwrite:
        raise FileExistsError(
            "Output subset already exists at {0}; pass overwrite=True to regenerate.".format(
                output_dir
            )
        )

    if overwrite:
        for path in managed_paths:
            if path.exists():
                shutil.rmtree(str(path))
        if manifest_path.exists():
            manifest_path.unlink()


def _write_manifest(
    *,
    output_dir: Path,
    source_dir: Path,
    rows: Sequence[ImageNetSubsetRow],
    class_order: Sequence[str],
    class_quotas: Mapping[str, int],
) -> None:
    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "class_order": list(class_order),
        "class_quotas": {class_name: int(class_quotas[class_name]) for class_name in class_order},
        "total": len(rows),
        "rows": [
            {
                "class": row.class_name,
                "label_index": int(row.label_index),
                "source": str(row.path),
                "filename": row.path.name,
            }
            for row in rows
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def materialize_class_subset(
    *,
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    class_order: Sequence[str],
    class_indices: Mapping[str, int],
    class_quotas: Mapping[str, int],
    overwrite: bool = False,
) -> MaterializedImageNetSubset:
    """Copy a deterministic class-folder subset to a standalone directory."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    if not source_path.is_dir():
        raise ValueError("Source ImageNet directory does not exist: {0}".format(source_path))
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Source and output directories must be different.")

    rows = select_class_subset(
        source_dir=source_path,
        class_order=class_order,
        class_indices=class_indices,
        class_quotas=class_quotas,
    )
    _prepare_output_dir(output_path, class_order, overwrite=overwrite)

    class_counts = {class_name: 0 for class_name in class_order}
    for row in rows:
        class_output_dir = output_path / row.class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(row.path), str(class_output_dir / row.path.name))
        class_counts[row.class_name] += 1

    _write_manifest(
        output_dir=output_path,
        source_dir=source_path,
        rows=rows,
        class_order=class_order,
        class_quotas=class_quotas,
    )
    return MaterializedImageNetSubset(
        source_dir=source_path,
        output_dir=output_path,
        rows=tuple(rows),
        class_counts=class_counts,
    )
