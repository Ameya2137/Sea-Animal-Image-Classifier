# Sea Animal Image Classifier

A modular image classification pipeline for classifying **23 categories of sea animals**, built with Python, TensorFlow, HOG feature extraction, and DVC for reproducible machine learning workflows.

The project separates the complete ML workflow into independent stages for data collection, preprocessing, feature extraction and selection, model training, and evaluation. DVC manages the dependencies between these stages and enables automated end-to-end pipeline execution.

---

## Project Overview

The classifier uses **Histogram of Oriented Gradients (HOG)** to extract visual features from sea animal images. Variance-based feature selection reduces the resulting feature space before classification using a **Multi-Layer Perceptron (MLP)** neural network.

The complete workflow is managed through a DVC pipeline:

```text
Raw Dataset
     |
     v
Data Collection
     |
     v
Data Processing
     |
     v
HOG Feature Extraction
     |
     v
Feature Selection
     |
     v
Model Training
     |
     v
Model Evaluation
```

## Dataset

The project uses the **Sea Animals Image Dataset** available on Kaggle:

https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste

The dataset contains **13,711 images across 23 classes**.

| Dataset Split | Images |
|---|---:|
| Training | 9,619 |
| Validation | 2,046 |
| Test | 2,046 |
| **Total** | **13,711** |

The 23 classes are:

```text
Clams, Corals, Crabs, Dolphin, Eel, Fish, Jelly Fish, Lobster,
Nudibranchs, Octopus, Otter, Penguin, Puffers, Sea Rays,
Sea Urchins, Seahorse, Seal, Sharks, Shrimp, Squid, Starfish,
Turtle_Tortoise, Whale
```

The raw dataset is not stored directly in this repository because of its size. DVC metadata is provided through `data/raw.dvc`.

---

## Project Structure

```text
sea_animal_classifier/
|
|-- src/
|   |-- data_collection.py
|   |-- data_processing.py
|   |-- feature_selection.py
|   |-- model_training.py
|   `-- model_evaluation.py
|
|-- data/
|   |-- raw.dvc
|   `-- processed/
|
|-- models/
|-- results/
|
|-- params.yaml
|-- dvc.yaml
|-- dvc.lock
|-- requirements.txt
|-- README.md
|-- .gitignore
`-- .dvcignore
```

---

## Pipeline Stages

### 1. Data Collection

`src/data_collection.py`

The data collection stage:

- Locates the downloaded dataset.
- Detects the class directories.
- Validates image files.
- Counts images for each class.
- Generates dataset statistics.

The validated dataset contains **13,711 images belonging to 23 classes**.

### 2. Data Processing

`src/data_processing.py`

The processing stage creates reproducible training, validation, and testing splits.

| Split | Percentage | Images |
|---|---:|---:|
| Training | ~70% | 9,619 |
| Validation | ~15% | 2,046 |
| Testing | ~15% | 2,046 |

A fixed random seed is used to make the splitting process reproducible.

### 3. Feature Extraction and Selection

`src/feature_selection.py`

Images are resized to **128 × 128 pixels** and converted to grayscale before feature extraction.

**Histogram of Oriented Gradients (HOG)** is used to represent image shape and edge information numerically.

HOG initially produces:

```text
8,100 features per image
```

Variance-based feature selection is then applied.

Importantly, the feature selector is fitted **only on the training data** and subsequently applied to the validation and test sets. This prevents information from the validation or test sets from leaking into feature selection.

Final feature dimensions:

```text
Training:   (9619, 303)
Validation: (2046, 303)
Test:       (2046, 303)
```

Therefore, the feature-selection stage reduces the representation from **8,100 to 303 features**.

### 4. Model Training

`src/model_training.py`

The selected HOG features are classified using a Multi-Layer Perceptron neural network.

Model architecture:

```text
Input
303 features
     |
     v
Dense Layer
256 neurons, ReLU
     |
     v
Dropout
     |
     v
Dense Layer
128 neurons, ReLU
     |
     v
Dropout
     |
     v
Output Layer
23 neurons, Softmax
```

The model contains:

```text
113,687 trainable parameters
```

Training parameters such as learning rate, batch size, epochs, optimizer, dropout rate, and hidden-layer dimensions are controlled through `params.yaml`.

### 5. Model Evaluation

`src/model_evaluation.py`

The trained model is evaluated against the independent test set.

The evaluation stage calculates and generates:

- Accuracy
- Precision
- Recall
- F1-score
- Per-class classification report
- Confusion matrix
- Training and validation accuracy curves
- Training and validation loss curves

The final automated DVC pipeline run achieved:

```text
Test Accuracy: 20.28%
```

The model performs above the approximately **4.35% random-choice baseline** for a balanced 23-class classification problem, but the results also demonstrate the limitations of HOG features for a complex multi-class natural-image dataset.

---

## DVC Pipeline

The complete workflow is defined in `dvc.yaml`.

The pipeline consists of five stages:

```text
data_collection
      |
      v
data_processing
      |
      v
feature_selection
      |
      v
model_training
      |
      v
model_evaluation
```

DVC tracks stage dependencies, parameters, and outputs. If nothing has changed, DVC avoids unnecessarily executing stages again.

For example:

```text
Stage 'data_collection' didn't change, skipping
Stage 'data_processing' didn't change, skipping
Stage 'feature_selection' didn't change, skipping
Stage 'model_training' didn't change, skipping
Stage 'model_evaluation' didn't change, skipping

Data and pipelines are up to date.
```

---

## Configuration

Pipeline parameters are centralized in `params.yaml`.

Configurable parameters include:

- Image size
- Number of image channels
- Validation split
- Test split
- Random seed
- Batch size
- Number of epochs
- Learning rate
- Optimizer
- Dropout rate
- Hidden-layer dimensions
- Feature-selection method
- Variance threshold
- Maximum number of features
- Rotation configuration
- Width shift
- Height shift
- Zoom
- Horizontal flipping
- Brightness configuration

This allows experiments to be configured without hard-coding values throughout the individual pipeline modules.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ameya2137/Sea-Animal-Image-Classifier.git
cd Sea-Animal-Image-Classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project was developed using **Python 3.11**.

---

## Dataset Setup

Download and extract the Sea Animals Image Dataset from Kaggle.

The extracted dataset should contain one directory for each animal class.

Example:

```text
sea_animals_dataset/
|-- Clams/
|-- Corals/
|-- Crabs/
|-- Dolphin/
|-- Eel/
|-- Fish/
|-- ...
`-- Whale/
```

Update the dataset location in `params.yaml`:

```yaml
data:
  source_dir: "C:/path/to/sea_animals_dataset"
  raw_dir: "data/raw"
  processed_dir: "data/processed"
```

The data collection stage can then prepare the dataset for the rest of the pipeline.

---

## Running the Pipeline

### Execute the complete pipeline

```bash
dvc repro
```

DVC automatically determines which stages need to execute based on changes to dependencies, parameters, source code, and previous outputs.

### Check pipeline status

```bash
dvc status
```

A fully reproduced pipeline should report:

```text
Data and pipelines are up to date.
```

### View the dependency graph

```bash
dvc dag
```

---

## Running Individual Stages

Each stage is implemented as an independent Python module and can also be executed manually.

```bash
python src/data_collection.py
python src/data_processing.py
python src/feature_selection.py
python src/model_training.py
python src/model_evaluation.py
```

For normal use, `dvc repro` is recommended because it manages stage dependencies automatically.

---

## Generated Outputs

The pipeline generates the following evaluation artifacts:

```text
results/
|-- dataset_statistics.yaml
|-- training_history.json
|-- evaluation_metrics.json
|-- classification_report.json
|-- confusion_matrix.png
|-- accuracy_curve.png
`-- loss_curve.png
```

The trained model and class mapping are generated under:

```text
models/
|-- sea_animal_classifier.keras
`-- class_mapping.json
```

Generated artifacts are managed by DVC rather than being stored directly in Git.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| TensorFlow / Keras | Neural network training |
| NumPy | Numerical processing |
| Pandas | Dataset metadata processing |
| scikit-learn | Feature selection and evaluation |
| scikit-image | HOG feature extraction |
| Matplotlib | Evaluation visualizations |
| Pillow | Image processing |
| PyYAML | Configuration management |
| DVC | ML pipeline and data versioning |
| Git | Source-code version control |

---

## Limitations

The current classifier uses handcrafted HOG features. HOG captures edges and shape information effectively but does not capture complex color, texture, and high-level semantic features as effectively as modern convolutional neural networks.

The dataset is also imbalanced, with some classes containing substantially more images than others. For example, the `Turtle_Tortoise` class contains considerably more samples than many other categories.

These factors contribute to the relatively modest classification performance.

---

## Future Improvements

Potential improvements include:

- Replacing HOG features with learned CNN features.
- Using transfer learning with architectures such as MobileNet or EfficientNet.
- Applying class weighting to address dataset imbalance.
- Introducing controlled image augmentation during training.
- Comparing handcrafted features against deep-learning representations.
- Performing hyperparameter optimization.
- Adding DVC experiment tracking for systematic model comparison.

---

## Reproducibility

The project uses:

- Modular Python pipeline stages
- Centralized YAML configuration
- Fixed random seeds
- DVC dependency tracking
- DVC pipeline automation
- Version-controlled pipeline definitions
- Environment dependencies through `requirements.txt`

Together, these components provide a reproducible workflow for experimentation and further development.