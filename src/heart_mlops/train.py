"""
Trains Logistic Regression and Random Forest classifiers on the Cleveland
heart disease data, tunes each with a small grid under stratified CV, logs
every run to MLflow, and persists the best-performing pipeline to disk.

Run directly:
    python -m heart_mlops.train
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe for CI runners
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heart_mlops.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_DIR,
    MODEL_DIR,
    MODEL_PATH,
    METADATA_PATH,
    PIPELINE_PATH,
    RANDOM_STATE,
)
from heart_mlops.data import load_clean_dataset
from heart_mlops.features import build_full_pipeline, split_features_target

CV_FOLDS = 5

# Kept intentionally small: this is a 303-row dataset, an exhaustive grid
# would just overfit the CV folds without teaching us anything new.
MODEL_GRID = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "param_grid": {
            "model__C": [0.1, 1.0, 3.0],
            "model__penalty": ["l2"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators": [150, 300],
            "model__max_depth": [4, 6, None],
            "model__min_samples_leaf": [1, 3],
        },
    },
}


def _evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def _log_plots(pipeline, X_test, y_test, run_tag: str, out_dir) -> None:
    cm_path = out_dir / f"{run_tag}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {run_tag}")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    roc_path = out_dir / f"{run_tag}_roc_curve.png"
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve - {run_tag}")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    mlflow.log_artifact(str(cm_path))
    mlflow.log_artifact(str(roc_path))


def train_and_track():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR.as_posix()}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_clean_dataset()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for model_name, cfg in MODEL_GRID.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = build_full_pipeline(cfg["estimator"])

            search = GridSearchCV(
                pipeline,
                param_grid=cfg["param_grid"],
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_

            test_metrics = _evaluate(best_pipeline, X_test, y_test)
            cv_mean_auc = search.best_score_

            mlflow.log_param("model_type", model_name)
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_mean_roc_auc", cv_mean_auc)
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            _log_plots(best_pipeline, X_test, y_test, model_name, MODEL_DIR)
            mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

            results.append(
                {
                    "model_name": model_name,
                    "pipeline": best_pipeline,
                    "best_params": search.best_params_,
                    "cv_mean_roc_auc": cv_mean_auc,
                    **test_metrics,
                }
            )

            print(f"[{model_name}] cv_roc_auc={cv_mean_auc:.4f} "
                  f"test_roc_auc={test_metrics['roc_auc']:.4f} "
                  f"test_accuracy={test_metrics['accuracy']:.4f}")

    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\nSelected model: {best['model_name']} (test ROC-AUC={best['roc_auc']:.4f})")

    joblib.dump(best["pipeline"], MODEL_PATH)
    joblib.dump(best["pipeline"].named_steps["preprocess"], PIPELINE_PATH)

    feature_columns = list(X.columns)
    metadata = {
        "model_name": best["model_name"],
        "best_params": best["best_params"],
        "cv_mean_roc_auc": best["cv_mean_roc_auc"],
        "test_metrics": {k: best[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
        "feature_columns": feature_columns,
        "random_state": RANDOM_STATE,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    summary = pd.DataFrame(results)[
        ["model_name", "cv_mean_roc_auc", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    print("\nModel comparison:\n", summary.to_string(index=False))

    return best


if __name__ == "__main__":
    train_and_track()
