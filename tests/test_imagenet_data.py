from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.data.imagenet import load_imagenet_images, resize_normalized_image
from deepdetector.data.imagenet_subset import (  # noqa: E402
    materialize_class_subset,
    select_class_subset,
)


def test_load_imagenet_images_skips_missing_and_corrupt_files(tmp_path) -> None:
    """Check ImageNet CSV loading with unreadable files."""
    Image = pytest.importorskip("PIL.Image")

    images_dir = tmp_path / "images"
    images_dir.mkdir()

    image = np.full((10, 12, 3), 128, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(str(images_dir / "valid.JPEG"))
    (images_dir / "corrupt.JPEG").write_text("not an image", encoding="utf-8")

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "valid.JPEG,3\nmissing.JPEG,4\ncorrupt.JPEG,5\n",
        encoding="utf-8",
    )

    rng = np.random.RandomState(0)
    images, labels = load_imagenet_images(
        csv_path=str(csv_path),
        images_dir=str(images_dir),
        n_samples=3,
        preprocess_fn=lambda array: resize_normalized_image(array, image_size=8),
        rng=rng,
    )

    assert images.shape == (1, 8, 8, 3)
    assert images.dtype == np.float32
    assert labels.shape == (1,)
    assert labels.dtype == np.int32
    assert int(labels[0]) == 3
    assert images.min() >= 0.0
    assert images.max() <= 1.0


def _write_class_files(root: Path, class_name: str, count: int) -> None:
    class_dir = root / class_name
    class_dir.mkdir(parents=True)
    for index in range(count):
        (class_dir / "{0}_{1:03d}.JPEG".format(class_name, index)).write_bytes(
            "{0}-{1}".format(class_name, index).encode("utf-8")
        )
    (class_dir / "ignore.txt").write_text("not an image", encoding="utf-8")


def test_select_class_subset_uses_order_and_quotas(tmp_path) -> None:
    """Subset selection should mirror the Inception v3 reproduction order."""
    source_dir = tmp_path / "imagenet" / "test"
    _write_class_files(source_dir, "zebra", 3)
    _write_class_files(source_dir, "panda", 3)
    _write_class_files(source_dir, "cab", 3)

    rows = select_class_subset(
        source_dir=source_dir,
        class_order=("zebra", "panda", "cab"),
        class_indices={"zebra": 80, "panda": 169, "cab": 267},
        class_quotas={"zebra": 2, "panda": 2, "cab": 1},
    )

    assert [row.class_name for row in rows] == ["zebra", "zebra", "panda", "panda", "cab"]
    assert [row.label_index for row in rows] == [80, 80, 169, 169, 267]
    assert [row.path.name for row in rows] == [
        "zebra_000.JPEG",
        "zebra_001.JPEG",
        "panda_000.JPEG",
        "panda_001.JPEG",
        "cab_000.JPEG",
    ]


def test_materialize_class_subset_copies_files_and_manifest(tmp_path) -> None:
    """The materializer should create the standalone Inception v3 dataset."""
    source_dir = tmp_path / "imagenet" / "test"
    output_dir = tmp_path / "inceptionV3"
    _write_class_files(source_dir, "zebra", 3)
    _write_class_files(source_dir, "panda", 3)
    _write_class_files(source_dir, "cab", 3)

    subset = materialize_class_subset(
        source_dir=source_dir,
        output_dir=output_dir,
        class_order=("zebra", "panda", "cab"),
        class_indices={"zebra": 80, "panda": 169, "cab": 267},
        class_quotas={"zebra": 2, "panda": 2, "cab": 1},
    )

    assert subset.class_counts == {"zebra": 2, "panda": 2, "cab": 1}
    assert sorted(path.name for path in (output_dir / "zebra").iterdir()) == [
        "zebra_000.JPEG",
        "zebra_001.JPEG",
    ]
    assert sorted(path.name for path in (output_dir / "panda").iterdir()) == [
        "panda_000.JPEG",
        "panda_001.JPEG",
    ]
    assert sorted(path.name for path in (output_dir / "cab").iterdir()) == ["cab_000.JPEG"]
    assert (output_dir / "manifest.json").is_file()


def test_materialize_class_subset_requires_force_for_existing_subset(tmp_path) -> None:
    """Existing materialized subsets should not be overwritten silently."""
    source_dir = tmp_path / "imagenet" / "test"
    output_dir = tmp_path / "inceptionV3"
    _write_class_files(source_dir, "zebra", 2)

    kwargs = {
        "source_dir": source_dir,
        "output_dir": output_dir,
        "class_order": ("zebra",),
        "class_indices": {"zebra": 80},
        "class_quotas": {"zebra": 1},
    }
    materialize_class_subset(**kwargs)

    with pytest.raises(FileExistsError, match="already exists"):
        materialize_class_subset(**kwargs)

    subset = materialize_class_subset(overwrite=True, **kwargs)

    assert subset.class_counts == {"zebra": 1}
