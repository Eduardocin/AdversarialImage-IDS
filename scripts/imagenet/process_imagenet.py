import csv
import os
import random
import shutil
from pathlib import Path


random.seed(42)

# Quantidades do artigo, Table 2.
TARGET_CLASSES = {
    "train": {
        "goldfish": {
            "synset": "n01443537",
            "label": 1,
            "quantity": 648,
        },
        "pineapple": {
            "synset": "n07753275",
            "label": 953,
            "quantity": 520,
        },
        "digital_clock": {
            "synset": "n03196217",
            "label": 530,
            "quantity": 455,
        },
    },
    "validation": {
        "zebra": {
            "synset": "n02391049",
            "label": 340,
            "quantity": 503,
        },
        "jellyfish": {
            "synset": "n01910747",
            "label": 107,
            "quantity": 618,
        },
    },
    "test": {
        "panda": {
            "synset": "n02510455",
            "label": 388,
            "quantity": 501,
        },
        "cab": {
            "synset": "n02930766",
            "label": 468,
            "quantity": 485,
        },
    },
}


ORIGIN_PATH = Path("./data")

# Destino organizado para os experimentos.
DESTINATION_PATH = Path("./data/imagenet/")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".JPEG", ".JPG")


def list_images(folder: Path):
    return sorted(
        filename
        for filename in os.listdir(folder)
        if filename.lower().endswith(tuple(ext.lower() for ext in IMAGE_EXTENSIONS))
    )


def clean_destination(destination_root: Path):
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)


def process_dataset(force: bool = False):
    if force:
        clean_destination(DESTINATION_PATH)
    else:
        DESTINATION_PATH.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    split_label_rows = {split: [] for split in TARGET_CLASSES}

    for split, classes in TARGET_CLASSES.items():
        for class_name, info in classes.items():
            synset = info["synset"]
            label = info["label"]
            quantity = info["quantity"]

            origin_folder = ORIGIN_PATH / synset
            destination_folder = DESTINATION_PATH / split / class_name

            if not origin_folder.exists():
                print(f"ERROR: origin folder does not exist: {origin_folder}")
                continue

            destination_folder.mkdir(parents=True, exist_ok=True)

            images = list_images(origin_folder)

            if len(images) < quantity:
                print(
                    f"ERROR: class {class_name} ({synset}) has only "
                    f"{len(images)} images; {quantity} are required."
                )
                continue

            selected_images = random.sample(images, quantity)

            for image_name in selected_images:
                shutil.copy2(
                    origin_folder / image_name,
                    destination_folder / image_name,
                )
                split_label_rows[split].append(
                    {
                        "filename": "{0}/{1}".format(class_name, image_name),
                        "label": label,
                    }
                )

            manifest_rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "synset": synset,
                    "label": label,
                    "source_path": str(origin_folder),
                    "destination_path": str(destination_folder),
                    "n_available": len(images),
                    "n_selected": quantity,
                }
            )

            print(
                f"[{split.upper()}] copied {quantity} images "
                f"from {class_name} ({synset}), label={label}."
            )

    write_manifest(manifest_rows)
    write_split_label_csvs(split_label_rows)
    print("Selection completed successfully.")


def write_manifest(rows):
    manifest_path = DESTINATION_PATH / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
        "class_name",
        "synset",
        "label",
        "source_path",
        "destination_path",
        "n_available",
        "n_selected",
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest saved to: {manifest_path}")


def write_split_label_csvs(split_label_rows):
    for split, rows in split_label_rows.items():
        labels_path = DESTINATION_PATH / f"{split}_labels.csv"
        with open(labels_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for row in rows:
                writer.writerow([row["filename"], row["label"]])

        print(f"{split} labels saved to: {labels_path}")


if __name__ == "__main__":
    process_dataset(force=True)
