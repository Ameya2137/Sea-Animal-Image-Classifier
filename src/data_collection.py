import os
import shutil
import yaml
from pathlib import Path
from PIL import Image
from collections import Counter


def load_params():
    """Load configuration parameters from params.yaml."""
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def collect_dataset(source_dir, destination_dir):
    """
    Copy the image dataset from the source directory into
    the project's raw data directory.
    """

    source_path = Path(source_dir)
    destination_path = Path(destination_dir)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {source_path}"
        )

    destination_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    class_counts = Counter()
    invalid_images = []

    class_directories = [
        directory
        for directory in source_path.iterdir()
        if directory.is_dir()
    ]

    if not class_directories:
        raise ValueError(
            "No class directories found in the source dataset."
        )

    print(f"Found {len(class_directories)} classes.")

    for class_directory in sorted(class_directories):

        class_name = class_directory.name
        destination_class = destination_path / class_name
        destination_class.mkdir(parents=True, exist_ok=True)

        for image_file in class_directory.iterdir():

            if image_file.suffix.lower() not in valid_extensions:
                continue

            try:
                # Validate that the file is a readable image.
                with Image.open(image_file) as image:
                    image.verify()

                destination_file = destination_class / image_file.name

                shutil.copy2(
                    image_file,
                    destination_file
                )

                class_counts[class_name] += 1

            except Exception:
                invalid_images.append(str(image_file))

    print("\nDataset collection completed.")
    print(f"Classes: {len(class_counts)}")
    print(f"Total valid images: {sum(class_counts.values())}")

    print("\nImages per class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count}")

    if invalid_images:
        print(
            f"\nInvalid images detected: {len(invalid_images)}"
        )

        with open(
            "results/invalid_images.txt",
            "w",
            encoding="utf-8"
        ) as file:

            for image in invalid_images:
                file.write(image + "\n")

    # Save dataset statistics.
    os.makedirs("results", exist_ok=True)

    statistics = {
        "number_of_classes": len(class_counts),
        "total_images": sum(class_counts.values()),
        "images_per_class": dict(class_counts),
        "invalid_images": len(invalid_images)
    }

    with open(
        "results/dataset_statistics.yaml",
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(
            statistics,
            file,
            sort_keys=False
        )


def main():

    params = load_params()

    source_dir = params["data"]["source_dir"]
    destination_dir = params["data"]["raw_dir"]

    collect_dataset(
        source_dir,
        destination_dir
    )


if __name__ == "__main__":
    main()