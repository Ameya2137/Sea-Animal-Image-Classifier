import os
import yaml
import numpy as np
import pandas as pd
import joblib

from PIL import Image
from skimage.feature import hog
from sklearn.feature_selection import VarianceThreshold


def load_params():
    """Load configuration parameters from params.yaml."""

    with open("params.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_hog_features(image_path, image_size):
    """Resize an image and extract its HOG feature vector."""

    image = Image.open(image_path).convert("L")

    image = image.resize(
        (image_size, image_size)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    features = hog(
        image_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


def build_feature_matrix(csv_path, image_size):
    """Convert all images in a manifest into HOG features."""

    dataframe = pd.read_csv(csv_path)

    features = []

    print(
        f"Extracting features from "
        f"{len(dataframe)} images..."
    )

    for index, image_path in enumerate(
        dataframe["image_path"]
    ):

        feature_vector = extract_hog_features(
            image_path,
            image_size
        )

        features.append(feature_vector)

        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1}/"
                f"{len(dataframe)} images"
            )

    return np.asarray(features), dataframe


def main():

    params = load_params()

    processed_dir = params["data"]["processed_dir"]
    image_size = params["image"]["image_size"]

    threshold = params[
        "feature_selection"
    ]["threshold"]

    max_features = params[
        "feature_selection"
    ]["max_features"]

    print("Starting feature extraction and selection.")

    # ---------------------------------------------------------
    # 1. Extract HOG features from all three datasets
    # ---------------------------------------------------------

    datasets = {}

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        print(
            f"\n--- Processing {split_name} set ---"
        )

        csv_path = os.path.join(
            processed_dir,
            f"{split_name}.csv"
        )

        features, dataframe = build_feature_matrix(
            csv_path,
            image_size
        )

        datasets[split_name] = {
            "features": features,
            "labels": dataframe["label"].to_numpy()
        }

        print(
            f"Raw HOG features: "
            f"{features.shape}"
        )

    # ---------------------------------------------------------
    # 2. FIT feature selector ONLY on training data
    # ---------------------------------------------------------

    print("\nFitting feature selector on training data...")

    selector = VarianceThreshold(
        threshold=threshold
    )

    train_features = datasets[
        "train"
    ]["features"]

    train_selected = selector.fit_transform(
        train_features
    )
    
    # Save fitted feature selector for deployment
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        selector,
        "models/feature_selector.joblib"
    )

    print(
        f"Features before selection: "
        f"{train_features.shape[1]}"
    )

    print(
        f"Features after variance selection: "
        f"{train_selected.shape[1]}"
    )

    # ---------------------------------------------------------
    # 3. Optional maximum-feature selection
    # ---------------------------------------------------------

    selected_indices = np.arange(
        train_selected.shape[1]
    )

    if train_selected.shape[1] > max_features:

        variances = np.var(
            train_selected,
            axis=0
        )

        selected_indices = np.argsort(
            variances
        )[-max_features:]

        selected_indices = np.sort(
            selected_indices
        )

        train_selected = train_selected[
            :,
            selected_indices
        ]

        print(
            f"Features after maximum-feature "
            f"selection: {train_selected.shape[1]}"
        )

    # ---------------------------------------------------------
    # 4. Transform validation and test using SAME selector
    # ---------------------------------------------------------

    output_feature_count = train_selected.shape[1]

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        features = datasets[
            split_name
        ]["features"]

        labels = datasets[
            split_name
        ]["labels"]

        if split_name == "train":

            selected_features = train_selected

        else:

            selected_features = selector.transform(
                features
            )

            selected_features = selected_features[
		:,
		selected_indices
	    ]

        output_path = os.path.join(
            processed_dir,
            f"{split_name}_features.npz"
        )

        np.savez_compressed(
            output_path,
            X=selected_features,
            y=labels
        )

        print(
            f"Saved {split_name} features: "
            f"{selected_features.shape}"
        )

    # ---------------------------------------------------------
    # 5. Final verification
    # ---------------------------------------------------------

    print("\nFeature selection completed.")

    print(
        f"Final feature count: "
        f"{output_feature_count}"
    )

    print("\nFinal feature shapes:")

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        feature_file = np.load(
            os.path.join(
                processed_dir,
                f"{split_name}_features.npz"
            )
        )

        print(
            f"{split_name}: "
            f"{feature_file['X'].shape}"
        )


if __name__ == "__main__":
    main()