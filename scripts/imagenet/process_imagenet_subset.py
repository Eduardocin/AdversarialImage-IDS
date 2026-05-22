import os
import random
import shutil
from pathlib import Path


random.seed(42)

TARGET_CLASSES = {
    "train": {"goldfish": 648, "pineapple": 520, "clock": 455},
    "validation": {"jellyfish": 618},
    "test": {"zebra": 503, "panda": 501, "cab": 485},
}

ORIGIN_PATH = Path("./data/imagenet/raw")
DESTINATION_PATH = Path("./data/imagenet/subset")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def process_dataset():
    for split, classes in TARGET_CLASSES.items():
        for class_name, quantity in classes.items():
            origin_folder = ORIGIN_PATH / class_name
            destination_folder = DESTINATION_PATH / split / class_name

            destination_folder.mkdir(parents=True, exist_ok=True)

            images = [
                filename
                for filename in os.listdir(origin_folder)
                if filename.lower().endswith(IMAGE_EXTENSIONS)
            ]

            if len(images) < quantity:
                print(
                    "ERROR: class {0} has only {1} images; {2} are required.".format(
                        class_name, len(images), quantity
                    )
                )
                continue

            selected_images = random.sample(images, quantity)

            for image_name in selected_images:
                shutil.copy2(origin_folder / image_name, destination_folder / image_name)

            print(
                "[{0}] copied {1} images from {2}.".format(
                    split.upper(), quantity, class_name
                )
            )


if __name__ == "__main__":
    process_dataset()
    print("Selection completed successfully.")
