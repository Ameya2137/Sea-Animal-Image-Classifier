\# Sea Animal Image Classification using DVC



\## Project Overview



This project implements a modular image classification application with an automated Machine Learning pipeline using DVC (Data Version Control).



The system classifies sea animal images into 23 different classes using HOG feature extraction, variance-based feature selection, and a Multi-Layer Perceptron neural network.



Dataset:

Sea Animals Image Dataset - Kaggle  

https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste



\## Dataset Information



\- Total Images: 13,711

\- Number of Classes: 23

\- Training Images: 9,619

\- Validation Images: 2,046

\- Test Images: 2,046



\## Project Structure



sea\_animal\_classifier/

|

|-- src/

|   |-- data\_collection.py

|   |-- data\_processing.py

|   |-- feature\_selection.py

|   |-- model\_training.py

|   |-- model\_evaluation.py

|

|-- data/

|   |-- raw.dvc

|   |-- processed/

|

|-- models/

|-- results/

|-- params.yaml

|-- dvc.yaml

|-- dvc.lock

|-- requirements.txt

|-- README.md



\## Pipeline Stages



\### 1. Data Collection



The data collection module validates the Sea Animals dataset, detects class folders, verifies image files, and generates dataset statistics.



\### 2. Data Processing



The dataset is divided into training, validation, and testing subsets using a reproducible class-wise split.



\- Training: approximately 70%

\- Validation: approximately 15%

\- Testing: approximately 15%



\### 3. Feature Selection



Images are resized to 128x128 pixels and converted to grayscale.



Histogram of Oriented Gradients (HOG) is used to extract image features.



Initial HOG features:



8100 features per image



VarianceThreshold feature selection is then fitted only on the training data to avoid data leakage.



Final selected features:



303 features per image



\### 4. Model Training



A Multi-Layer Perceptron neural network is used for classification.



Architecture:



\- Input Layer: 303 features

\- Dense Layer: 256 neurons

\- Dropout

\- Dense Layer: 128 neurons

\- Dropout

\- Output Layer: 23 classes with Softmax activation



The model contains approximately 113,687 trainable parameters.



\### 5. Model Evaluation



The trained model is evaluated on the test dataset using:



\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Classification Report

\- Confusion Matrix

\- Training/Validation Accuracy Curve

\- Training/Validation Loss Curve



Final test accuracy obtained during the automated DVC run:



20.28%



\## Configuration



All configurable parameters are stored inside:



params.yaml



Parameters include:



\- Image size

\- Number of channels

\- Validation split

\- Test split

\- Random seed

\- Batch size

\- Number of epochs

\- Learning rate

\- Optimizer

\- Dropout rate

\- Hidden-layer sizes

\- Feature-selection threshold

\- Maximum selected features

\- Augmentation parameters



More than 10 tunable parameters are provided.



\## Installation



Create a virtual environment:



python -m venv .venv



Activate it on Windows:



.venv\\Scripts\\activate



Install dependencies:



pip install -r requirements.txt



\## Dataset Setup



Download the Sea Animals Image Dataset from Kaggle.



Extract the dataset and update the source\_dir parameter inside params.yaml if necessary.



Example:



data:

&#x20; source\_dir: "C:/Users/USERNAME/Downloads/sea\_animals\_dataset"



\## DVC



Initialize DVC:



dvc init



The raw dataset is tracked using:



dvc add data/raw



To execute the complete ML pipeline:



dvc repro



DVC automatically executes the following dependency graph:



Data Collection

\-> Data Processing

\-> Feature Selection

\-> Model Training

\-> Model Evaluation



To check the pipeline status:



dvc status



To display the pipeline graph:



dvc dag



\## Results



The pipeline generates:



\- dataset\_statistics.yaml

\- training\_history.json

\- evaluation\_metrics.json

\- classification\_report.json

\- confusion\_matrix.png

\- accuracy\_curve.png

\- loss\_curve.png



\## Technology Stack



\- Python 3.11

\- TensorFlow

\- NumPy

\- Pandas

\- scikit-learn

\- scikit-image

\- Matplotlib

\- Pillow

\- PyYAML

\- DVC

\- Git

