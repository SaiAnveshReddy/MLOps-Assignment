"""
Central place for paths and constants so every script (notebook, training,
tests, API) agrees on the same values instead of re-declaring them.
"""

from pathlib import Path

# repo_root/src/heart_mlops/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "processed.cleveland.data"
PROCESSED_DATA_PATH = REPO_ROOT / "data" / "processed" / "heart_cleaned.csv"

MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "heart_risk_model.pkl"
PIPELINE_PATH = MODEL_DIR / "preprocess_pipeline.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

MLFLOW_TRACKING_DIR = REPO_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "cardiorisk-heart-disease"

# Raw Cleveland file has no header row; this is the column order documented
# in heart-disease.names (the UCI "processed.cleveland.data" field order).
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Columns that use "?" as a missing-value sentinel in the raw file.
COLUMNS_WITH_MISSING = ["ca", "thal"]

TARGET_COL = "num"
BINARY_TARGET_COL = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# Last 4 digits of my BITS ID (2025AE05969) used as the project's fixed seed -
# makes every run reproducible and is easy for me to recall/justify.
RANDOM_STATE = 5969
