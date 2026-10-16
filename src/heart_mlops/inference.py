"""Loads the saved pipeline once and serves single-patient predictions."""

from functools import lru_cache

import joblib
import pandas as pd

from heart_mlops.config import MODEL_PATH
from heart_mlops.features import add_engineered_features

# Raw input fields the API/consumer must supply, in the order the raw
# dataset defines them (before the `hr_reserve` feature is derived).
INPUT_FIELDS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


@lru_cache(maxsize=1)
def get_pipeline():
    return joblib.load(MODEL_PATH)


def predict_risk(patient: dict) -> dict:
    """patient: dict with the 13 raw INPUT_FIELDS as keys."""
    missing = [f for f in INPUT_FIELDS if f not in patient]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    row = pd.DataFrame([{field: patient[field] for field in INPUT_FIELDS}])
    row = add_engineered_features(row)

    pipeline = get_pipeline()
    proba = pipeline.predict_proba(row)[0]
    prediction = int(proba[1] >= 0.5)
    confidence = float(proba[prediction])

    return {
        "prediction": prediction,
        "risk_label": "high_risk" if prediction == 1 else "low_risk",
        "confidence": round(confidence, 4),
        "probability_disease": round(float(proba[1]), 4),
    }
