"""
Automated tests for the Sea Animal Image Classification project.

These tests validate core data-processing functionality without requiring
the complete image dataset or retraining the neural network. This makes
them suitable for execution in a GitHub Actions CI environment.
"""

import tempfile
from pathlib import Path

from src.data_processing import (
    collect_image_paths,
    split_dataset,
)


def test_required_project_files():
    """Verify that essential project configuration files exist."""

    required_files = [
        "params.yaml",
        "dvc.yaml",
        "dvc.lock",
        "requirements.txt",
        "src/data_collection.py",
        "src/data_processing.py",
        "src/feature_selection.py",
        "src/model_training.py",
        "src/model_evaluation.py",
    ]

    for file_path in required_files:
        assert Path(file_path).exists(), f"Missing required file: {file_path}"


def test_collect_image_paths():
    """Check whether valid image files are detected correctly."""

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        fish_dir = root / "Fish"
        shark_dir = root / "Sharks"

        fish_dir.mkdir()
        shark_dir.mkdir()

        # Dummy files are sufficient because this function checks paths/extensions.
        (fish_dir / "fish1.jpg").touch()
        (fish_dir / "fish2.png").touch()
        (shark_dir / "shark1.jpeg").touch()

        # Invalid extension should be ignored.
        (fish_dir / "notes.txt").touch()

        records = collect_image_paths(root)

        assert len(records) == 3

        labels = [record["label"] for record in records]

        assert labels.count("Fish") == 2
        assert labels.count("Sharks") == 1


def test_dataset_split():
    """Verify train/validation/test splitting."""

    records = []

    # 20 artificial samples per class.
    for label in ["Fish", "Sharks"]:
        for index in range(20):
            records.append(
                {
                    "image_path": f"{label}_{index}.jpg",
                    "label": label,
                }
            )

    train, validation, test = split_dataset(
        records,
        validation_size=0.20,
        test_size=0.20,
        random_seed=42,
    )

    # 40 samples:
    # 60% train, 20% validation, 20% test.
    assert len(train) == 24
    assert len(validation) == 8
    assert len(test) == 8

    assert all(item["split"] == "train" for item in train)
    assert all(item["split"] == "validation" for item in validation)
    assert all(item["split"] == "test" for item in test)


def run_tests():
    """Execute all CI tests."""

    tests = [
        test_required_project_files,
        test_collect_image_paths,
        test_dataset_split,
    ]

    print("=" * 60)
    print("SEA ANIMAL IMAGE CLASSIFIER - AUTOMATED TESTS")
    print("=" * 60)

    passed = 0

    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
            passed += 1

        except Exception as error:
            print(f"[FAIL] {test.__name__}")
            print(f"       {error}")
            raise

    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print("All automated tests completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()