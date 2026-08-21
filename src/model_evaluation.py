import os
import json
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder


def load_params():
    """Load configuration parameters from params.yaml."""

    with open(
        "params.yaml",
        "r",
        encoding="utf-8"
    ) as file:
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


def load_class_mapping():
    """Load the class mapping created during training."""

    with open(
        "models/class_mapping.json",
        "r",
        encoding="utf-8"
    ) as file:
        mapping = json.load(file)

    mapping = {
        int(key): value
        for key, value in mapping.items()
    }

    return mapping


def evaluate_model(
    model,
    X_test,
    y_test,
    class_mapping
):
    """Evaluate the trained model on the test set."""

    class_names = [
        class_mapping[index]
        for index in sorted(class_mapping)
    ]

    label_encoder = LabelEncoder()

    label_encoder.fit(class_names)

    y_test_encoded = label_encoder.transform(
        y_test
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    print("\nGenerating predictions...")

    probabilities = model.predict(
        X_test,
        verbose=1
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # ---------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test_encoded,
        y_pred
    )

    macro_precision = precision_score(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(
        f"\nTest Accuracy: {accuracy:.4f}"
    )

    print(
        f"Macro Precision: {macro_precision:.4f}"
    )

    print(
        f"Macro Recall: {macro_recall:.4f}"
    )

    print(
        f"Macro F1 Score: {macro_f1:.4f}"
    )

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    report = classification_report(
        y_test_encoded,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # ---------------------------------------------------------
    # Save classification report
    # ---------------------------------------------------------

    report_dict = classification_report(
        y_test_encoded,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    with open(
        "results/classification_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report_dict,
            file,
            indent=4
        )

    # ---------------------------------------------------------
    # Save overall metrics
    # ---------------------------------------------------------

    metrics = {
        "test_accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "number_of_test_images": int(len(y_test)),
        "number_of_classes": int(len(class_names))
    }

    with open(
        "results/evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_test_encoded,
        y_pred
    )

    figure_size = (
        max(12, len(class_names) * 0.6),
        max(10, len(class_names) * 0.6)
    )

    fig, ax = plt.subplots(
        figsize=figure_size
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=False
    )

    plt.title(
        "Sea Animal Classification - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        "results/confusion_matrix.png",
        dpi=200
    )

    plt.close()

    print(
        "\nConfusion matrix saved to:"
        " results/confusion_matrix.png"
    )

    return accuracy


def plot_training_history():
    """Generate training and validation accuracy/loss plots."""

    history_path = (
        "results/training_history.json"
    )

    with open(
        history_path,
        "r",
        encoding="utf-8"
    ) as file:
        history = json.load(file)

    # ---------------------------------------------------------
    # Accuracy plot
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/accuracy_curve.png",
        dpi=200
    )

    plt.close()

    # ---------------------------------------------------------
    # Loss plot
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/loss_curve.png",
        dpi=200
    )

    plt.close()

    print(
        "Training curves saved."
    )


def main():

    params = load_params()

    processed_dir = params[
        "data"
    ]["processed_dir"]

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model_path = (
        "models/sea_animal_classifier.keras"
    )

    print(
        f"Loading model: {model_path}"
    )

    model = tf.keras.models.load_model(
        model_path
    )

    # ---------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------

    print(
        "Loading test features..."
    )

    X_test, y_test = load_features(
        processed_dir,
        "test"
    )

    print(
        f"Test data: {X_test.shape}"
    )

    # ---------------------------------------------------------
    # Load class mapping
    # ---------------------------------------------------------

    class_mapping = load_class_mapping()

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    evaluate_model(
        model,
        X_test,
        y_test,
        class_mapping
    )

    # ---------------------------------------------------------
    # Plot training history
    # ---------------------------------------------------------

    plot_training_history()

    print(
        "\nModel evaluation completed."
    )


if __name__ == "__main__":
    main()