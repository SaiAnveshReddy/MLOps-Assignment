import pandas as pd

from heart_mlops.data import load_clean_dataset
from heart_mlops.features import (
    add_engineered_features,
    build_full_pipeline,
    split_features_target,
)
from sklearn.linear_model import LogisticRegression


def test_hr_reserve_is_computed_correctly():
    df = pd.DataFrame({"age": [50], "thalach": [150]})
    out = add_engineered_features(df)
    assert out.loc[0, "hr_reserve"] == (220 - 50) - 150


def test_split_features_target_drops_target_and_adds_engineered_col():
    df = load_clean_dataset()
    X, y = split_features_target(df)
    assert "target" not in X.columns
    assert "hr_reserve" in X.columns
    assert len(X) == len(y) == 303


def test_pipeline_fits_and_predicts_on_real_data():
    df = load_clean_dataset()
    X, y = split_features_target(df)

    pipeline = build_full_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert probas.shape == (len(X), 2)
    assert set(preds) <= {0, 1}
