"""
Feature engineering and the preprocessing pipeline.

Engineered feature -- `hr_reserve`:
Cardiology commonly estimates a patient's predicted max heart rate as
(220 - age). The gap between that prediction and the achieved max heart
rate during the stress test (`thalach`) is the "heart rate reserve": a
patient who falls well short of their predicted max under exercise is
generally read as having lower cardiovascular fitness. Per the EDA
(notebooks/01_eda.ipynb, section 6), `hr_reserve` doesn't actually beat
`thalach`'s own linear correlation with the target - it's kept anyway
as a single clinically-named number instead of asking the model (and a
reader) to implicitly recombine `age` and `thalach`.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_mlops.config import BINARY_TARGET_COL, NUMERIC_FEATURES

# Already-binary columns: left as-is, no encoding needed.
BINARY_PASSTHROUGH = ["sex", "fbs", "exang"]
# True multi-class categoricals: need one-hot encoding.
MULTI_CATEGORY = ["cp", "restecg", "slope", "thal"]

ENGINEERED_NUMERIC = ["hr_reserve"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hr_reserve"] = (220 - df["age"]) - df["thalach"]
    return df


def split_features_target(df: pd.DataFrame):
    df = add_engineered_features(df)
    y = df[BINARY_TARGET_COL]
    X = df.drop(columns=[BINARY_TARGET_COL])
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_cols = NUMERIC_FEATURES + ENGINEERED_NUMERIC

    return ColumnTransformer(
        transformers=[
            ("scale_numeric", StandardScaler(), numeric_cols),
            ("passthrough_binary", "passthrough", BINARY_PASSTHROUGH),
            (
                "onehot_multicategory",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                MULTI_CATEGORY,
            ),
        ],
        remainder="drop",
    )


def build_full_pipeline(estimator) -> Pipeline:
    """Wrap an estimator with preprocessing so it can be saved/loaded as one unit."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
