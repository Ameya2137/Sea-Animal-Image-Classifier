import os
import json
import yaml
import numpy as np
import tensorflow as tf
import random

from sklearn.preprocessing import LabelEncoder


def load_params():
    """Load configuration parameters from params.yaml."""

    with open("params.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_features(processed_dir, split_name):
    """Load selected features and labels."""

    feature_file = os.path.join(
        processed_dir,
        f"{split_name}_features.npz"
    )

    data = np.load(
        feature_file,
        allow_pickle=True
    )

    return data["X"], data["y"]


def create_model(
    input_dim,
    number_of_classes,
    dense_units_1,
    dense_units_2,
    dropout_rate,
    learning_rate,
    optimizer_name
):
    """Create and compile the neural-network classifier."""

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(input_dim,)
        ),

        tf.keras.layers.Dense(
            dense_units_1,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            dropout_rate
        ),

        tf.keras.layers.Dense(
            dense_units_2,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            dropout_rate
        ),

        tf.keras.layers.Dense(
            number_of_classes,
            activation="softmax"
        )
    ])

    if optimizer_name.lower() == "adam":

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

    elif optimizer_name.lower() == "sgd":

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate
        )

    elif optimizer_name.lower() == "rmsprop":

        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate
        )

    else:

        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    params = load_params()
    random_seed = params["split"]["random_seed"]

    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    processed_dir = params[
        "data"
    ]["processed_dir"]

    batch_size = params[
        "training"
    ]["batch_size"]

    epochs = params[
        "training"
    ]["epochs"]

    learning_rate = params[
        "training"
    ]["learning_rate"]

    optimizer_name = params[
        "training"
    ]["optimizer"]

    dropout_rate = params[
        "training"
    ]["dropout_rate"]

    dense_units_1 = params[
        "model"
    ]["dense_units_1"]

    dense_units_2 = params[
        "model"
    ]["dense_units_2"]

    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------

    print("Loading selected features...")

    X_train, y_train = load_features(
        processed_dir,
        "train"
    )

    X_validation, y_validation = load_features(
        processed_dir,
        "validation"
    )

    print(
        f"Training data: {X_train.shape}"
    )

    print(
        f"Validation data: {X_validation.shape}"
    )

    # ---------------------------------------------------------
    # Encode class labels
    # ---------------------------------------------------------

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(
        y_train
    )

    y_validation_encoded = label_encoder.transform(
        y_validation
    )

    class_names = label_encoder.classes_

    print(
        f"Number of classes: {len(class_names)}"
    )

    print(
        f"Classes: {list(class_names)}"
    )

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------

    model = create_model(
        input_dim=X_train.shape[1],
        number_of_classes=len(class_names),
        dense_units_1=dense_units_1,
        dense_units_2=dense_units_2,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )

    model.summary()

    # ---------------------------------------------------------
    # Create output directories
    # ---------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------

    print("\nStarting model training...")

    history = model.fit(
        X_train,
        y_train_encoded,
        validation_data=(
            X_validation,
            y_validation_encoded
        ),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    # ---------------------------------------------------------
    # Save trained model
    # ---------------------------------------------------------

    model_path = os.path.join(
        "models",
        "sea_animal_classifier.keras"
    )

    model.save(model_path)

    print(
        f"\nModel saved to: {model_path}"
    )

    # ---------------------------------------------------------
    # Save label encoder
    # ---------------------------------------------------------

    label_mapping = {
        int(index): str(label)
        for index, label in enumerate(
            class_names
        )
    }

    with open(
        "models/class_mapping.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            label_mapping,
            file,
            indent=4
        )

    # ---------------------------------------------------------
    # Save training history
    # ---------------------------------------------------------

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    with open(
        "results/training_history.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=4
        )

    print(
        "Training history saved."
    )


if __name__ == "__main__":
    main()