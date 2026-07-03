from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_PATH, TARGET_COLUMN
from src.data import load_training_data, split_features_target
from src.evaluate import evaluate_binary_classifier, find_recall_focused_threshold
from src.model_registry import save_metadata, save_model
from src.modeling import build_model


def _optional_mlflow_start(run_name: str):
    """Start MLflow when installed; otherwise use a no-op context manager."""
    class NoOpRun:
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False

    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        return mlflow.start_run(run_name=run_name)
    except Exception:
        return NoOpRun()


def _optional_log_to_mlflow(params: dict, metrics: dict, model, artifact_path: str) -> str | None:
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        active = mlflow.active_run()
        return active.info.run_id if active else None
    except Exception:
        return None


def train(
    data_path: str | Path | None = None,
    model_path: str | Path = MODEL_PATH,
    model_type: str = "random_forest",
    test_size: float = 0.2,
) -> dict:
    df = load_training_data(data_path)
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42, stratify=y
    )

    model = build_model(model_type=model_type)
    with _optional_mlflow_start(run_name=f"{MODEL_NAME}-{model_type}"):
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        threshold = find_recall_focused_threshold(y_test, probabilities)
        metrics = evaluate_binary_classifier(y_test, probabilities, threshold=threshold).to_dict()
        params = {
            "model_type": model_type,
            "rows": int(len(df)),
            "features": int(len(x.columns)),
            "test_size": test_size,
            "target": TARGET_COLUMN,
        }
        run_id = _optional_log_to_mlflow(params, metrics, model, artifact_path="model")

    saved_model_path = save_model(model, model_path)
    metadata = {
        "model_name": MODEL_NAME,
        "model_type": model_type,
        "model_path": str(saved_model_path),
        "threshold": threshold,
        "metrics": metrics,
        "training_rows": int(len(df)),
        "features": list(x.columns),
        "mlflow_run_id": run_id,
    }
    save_metadata(metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the churn model with MLflow tracking.")
    parser.add_argument("--data-path", default="data/raw/sample_churn.csv")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--model-type", choices=["random_forest", "logistic_regression"], default="random_forest")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train(
        data_path=args.data_path,
        model_path=args.model_path,
        model_type=args.model_type,
        test_size=args.test_size,
    )
    print(json.dumps(results, indent=2))
