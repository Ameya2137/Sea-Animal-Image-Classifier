import io
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from skimage.feature import hog


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"

MODEL_PATH = PROJECT_ROOT / "models" / "sea_animal_classifier.keras"
CLASS_MAPPING_PATH = PROJECT_ROOT / "models" / "class_mapping.json"
SELECTOR_PATH = PROJECT_ROOT / "models" / "feature_selector.joblib"

IMAGE_SIZE = 128

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}


# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)
feature_selector = joblib.load(SELECTOR_PATH)

with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as file:
    class_mapping = json.load(file)


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="Sea Animal Classifier",
    description="ML-powered sea animal image classification",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=APP_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(
    directory=APP_DIR / "templates"
)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def extract_features(image: Image.Image):

    image = image.convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = (
        np.asarray(image, dtype=np.float32) / 255.0
    )

    hog_features = hog(
        image_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )

    hog_features = hog_features.reshape(1, -1)

    selected_features = feature_selector.transform(
        hog_features
    )

    return selected_features.astype(np.float32)


# ---------------------------------------------------------
# Web UI
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "number_of_classes": len(class_mapping)
        },
    )


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_selector_loaded": True,
        "number_of_classes": len(class_mapping),
    }


# ---------------------------------------------------------
# Classes
# ---------------------------------------------------------

@app.get("/classes")
def classes():

    return {
        "number_of_classes": len(class_mapping),
        "classes": class_mapping,
    }


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, WEBP or BMP image.",
        )

    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image must be smaller than 10 MB.",
            )

        image = Image.open(io.BytesIO(contents))
        image.verify()

        # Reopen after verify()
        image = Image.open(io.BytesIO(contents))

        features = extract_features(image)

        expected_features = int(model.input_shape[-1])

        if features.shape[1] != expected_features:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Feature mismatch: model expects "
                    f"{expected_features}, preprocessing produced "
                    f"{features.shape[1]}."
                ),
            )

        probabilities = model.predict(
            features,
            verbose=0,
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = class_mapping[
            str(predicted_index)
        ]

        confidence = float(
            probabilities[predicted_index]
        )

        top_indices = np.argsort(
            probabilities
        )[-3:][::-1]

        top_predictions = [
            {
                "class": class_mapping[str(int(index))],
                "confidence": round(
                    float(probabilities[index]),
                    4,
                ),
                "percentage": round(
                    float(probabilities[index]) * 100,
                    2,
                ),
            }
            for index in top_indices
        ]

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(
                confidence * 100,
                2,
            ),
            "top_3_predictions": top_predictions,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {error}",
        )