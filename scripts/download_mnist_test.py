from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


EXPECTED_TEST_IMAGES = 10000
EXPECTED_IMAGE_SHAPE = (28, 28)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = DATA_DIR / "images"
DATASET_PATH = DATA_DIR / "mnist_splits.npz"
SAMPLES_PATH = OUTPUT_DIR / "mnist_train_samples.png"
TRAIN_RANGE = slice(0, 4500)
VALIDATION_RANGE = slice(4500, 5500)
TEST_RANGE = slice(5500, 10000)


def validate_mnist_test(x_test, y_test):
    if x_test.shape != (EXPECTED_TEST_IMAGES, *EXPECTED_IMAGE_SHAPE):
        raise ValueError(
            "Unexpected x_test shape: "
            f"{x_test.shape}; expected {(EXPECTED_TEST_IMAGES, *EXPECTED_IMAGE_SHAPE)}"
        )

    if y_test.shape != (EXPECTED_TEST_IMAGES,):
        raise ValueError(
            f"Unexpected y_test shape: {y_test.shape}; expected {(EXPECTED_TEST_IMAGES,)}"
        )

    if not np.issubdtype(y_test.dtype, np.integer):
        raise ValueError(f"Unexpected label dtype: {y_test.dtype}; expected integer labels")

    unique_labels = np.unique(y_test)
    expected_labels = np.arange(10)
    if not np.array_equal(unique_labels, expected_labels):
        raise ValueError(
            f"Unexpected label set: {unique_labels.tolist()}; expected digits 0 through 9"
        )


def split_mnist_test(x_test, y_test):
    return {
        "train": (x_test[TRAIN_RANGE], y_test[TRAIN_RANGE], TRAIN_RANGE.start),
        "validation": (
            x_test[VALIDATION_RANGE],
            y_test[VALIDATION_RANGE],
            VALIDATION_RANGE.start,
        ),
        "test": (x_test[TEST_RANGE], y_test[TEST_RANGE], TEST_RANGE.start),
    }


def save_split_images(splits):
    for split_name, (images, labels, start_index) in splits.items():
        split_dir = IMAGES_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for offset, (image, label) in enumerate(zip(images, labels)):
            original_index = start_index + offset
            image_path = split_dir / f"mnist_{original_index:05d}_label_{int(label)}.png"
            plt.imsave(image_path, image, cmap="gray", vmin=0, vmax=255)


def save_sample_preview(x_train, y_train):
    sample_indices = [0, 1, 2]
    figure, axes = plt.subplots(1, len(sample_indices), figsize=(6, 2.4))

    for axis, index in zip(axes, sample_indices):
        axis.imshow(x_train[index], cmap="gray")
        axis.set_title(f"Label: {int(y_train[index])}")
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(SAMPLES_PATH, dpi=150)
    plt.close(figure)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (_, _), (x_test, y_test) = mnist.load_data()
    validate_mnist_test(x_test, y_test)
    splits = split_mnist_test(x_test, y_test)
    x_train, y_train, _ = splits["train"]
    x_validation, y_validation, _ = splits["validation"]
    x_test_split, y_test_split, _ = splits["test"]

    np.savez_compressed(
        DATASET_PATH,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test_split,
        y_test=y_test_split,
    )
    save_split_images(splits)
    save_sample_preview(x_train, y_train)

    train_label_counts = np.bincount(y_train, minlength=10)
    validation_label_counts = np.bincount(y_validation, minlength=10)
    test_label_counts = np.bincount(y_test_split, minlength=10)

    print("MNIST test dataset downloaded, validated, and split.")
    print(f"Train images: {len(x_train)} (indices 0-4499)")
    print(f"Validation images: {len(x_validation)} (indices 4500-5499)")
    print(f"Test images: {len(x_test_split)} (indices 5500-9999)")
    print(f"x_train shape: {x_train.shape}")
    print(f"x_train dtype: {x_train.dtype}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_train dtype: {y_train.dtype}")
    print(f"Train label distribution: {train_label_counts.tolist()}")
    print(f"Validation label distribution: {validation_label_counts.tolist()}")
    print(f"Test label distribution: {test_label_counts.tolist()}")
    print(f"Dataset saved to: {DATASET_PATH}")
    print(f"Split image folders saved to: {IMAGES_DIR}")
    print("Images are split only by original MNIST test index order.")
    print(f"Train sample preview saved to: {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
