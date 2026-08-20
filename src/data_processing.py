import os
import random
import yaml
import pandas as pd
from pathlib import Path


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def collect_image_paths(raw_dir):
    """Collect image paths and their corresponding class labels."""

    raw_path = Path(raw_dir)

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    records = []

    for class_directory in sorted(raw_path.iterdir()):

        if not class_directory.is_dir():
            continue

        class_name = class_directory.name

        for image_file in class_directory.iterdir():

            if image_file.suffix.lower() in valid_extensions:

                records.append({
                    "image_path": str(image_file),
                    "label": class_name
                })

    return records


def split_dataset(records, validation_size, test_size, random_seed):
    """
    Split images into training, validation and test datasets.
    Splitting is performed independently for every class.
    """

    random.seed(random_seed)

    class_records = {}

    for record in records:
        class_records.setdefault(
            record["label"],
            []
        ).append(record)

    train_records = []
    validation_records = []
    test_records = []

    for label, items in class_records.items():

        random.shuffle(items)

        total = len(items)

        test_count = int(total * test_size)
        validation_count = int(total * validation_size)

        test_items = items[:test_count]

        validation_items = items[
            test_count:
            test_count + validation_count
        ]

        train_items = items[
            test_count + validation_count:
        ]

        for item in train_items:
            item["split"] = "train"
            train_records.append(item)

        for item in validation_items:
            item["split"] = "validation"
            validation_records.append(item)

        for item in test_items:
            item["split"] = "test"
            test_records.append(item)

    return (
        train_records,
        validation_records,
        test_records
    )


def save_manifests(
    train_records,
    validation_records,
    test_records,
    processed_dir
):
    """Save dataset manifests as CSV files."""

    processed_path = Path(processed_dir)
    processed_path.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df = pd.DataFrame(train_records)
    validation_df = pd.DataFrame(validation_records)
    test_df = pd.DataFrame(test_records)

    train_df.to_csv(
        processed_path / "train.csv",
        index=False
    )

    validation_df.to_csv(
        processed_path / "validation.csv",
        index=False
    )

    test_df.to_csv(
        processed_path / "test.csv",
        index=False
    )

    print("\nDataset split completed.")

    print(f"Training images:   {len(train_df)}")
    print(f"Validation images: {len(validation_df)}")
    print(f"Test images:       {len(test_df)}")

    print("\nClass distribution:")
    print(
        train_df["label"]
        .value_counts()
        .sort_index()
    )


def main():

    params = load_params()

    raw_dir = params["data"]["raw_dir"]
    processed_dir = params["data"]["processed_dir"]

    validation_size = params["split"]["validation_size"]
    test_size = params["split"]["test_size"]
    random_seed = params["split"]["random_seed"]

    records = collect_image_paths(raw_dir)

    print(f"Total images found: {len(records)}")

    train_records, validation_records, test_records = split_dataset(
        records,
        validation_size,
        test_size,
        random_seed
    )

    save_manifests(
        train_records,
        validation_records,
        test_records,
        processed_dir
    )


if __name__ == "__main__":
    main()