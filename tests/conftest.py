"""
Shared fixture: trains a small real pipeline on the actual dataset and points
the inference module at it, so inference/API tests exercise the real
predict_risk() code path instead of mocking it away.
"""

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

import heart_mlops.inference as inference_module
from heart_mlops.data import load_clean_dataset
from heart_mlops.features import build_full_pipeline, split_features_target


@pytest.fixture()
def trained_model_path(tmp_path, monkeypatch):
    df = load_clean_dataset()
    X, y = split_features_target(df)

    pipeline = build_full_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)

    model_path = tmp_path / "test_model.pkl"
    joblib.dump(pipeline, model_path)

    monkeypatch.setattr(inference_module, "MODEL_PATH", model_path)
    inference_module.get_pipeline.cache_clear()

    yield model_path

    inference_module.get_pipeline.cache_clear()


@pytest.fixture()
def sample_patient():
    return {
        "age": 57, "sex": 1, "cp": 4, "trestbps": 140, "chol": 289,
        "fbs": 0, "restecg": 2, "thalach": 145, "exang": 1,
        "oldpeak": 2.0, "slope": 2, "ca": 1, "thal": 7,
    }
