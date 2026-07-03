from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor


def build_model(model_type: str = "random_forest") -> Pipeline:
    """Build a fully reproducible sklearn Pipeline containing preprocessing and model logic."""
    model_type = model_type.lower().strip()
    if model_type == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=140,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError("model_type must be either 'random_forest' or 'logistic_regression'")

    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])
