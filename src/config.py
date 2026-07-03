from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MONITORING_DIR = PROJECT_ROOT / "monitoring"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
MONITORING_DIR.mkdir(exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "customer_churn_classifier")
MODEL_STAGE = os.getenv("MODEL_STAGE", "local")
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_PATH", "models/churn_pipeline.joblib")
METADATA_PATH = PROJECT_ROOT / os.getenv("METADATA_PATH", "models/model_metadata.json")
PREDICTION_LOG_PATH = PROJECT_ROOT / os.getenv(
    "PREDICTION_LOG_PATH", "monitoring/prediction_logs.jsonl"
)
REFERENCE_DATA_PATH = PROJECT_ROOT / os.getenv("REFERENCE_DATA_PATH", "data/raw/sample_churn.csv")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-churn-mlops")

TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls_last_90d",
    "late_payments_last_12m",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RISK_THRESHOLDS = {
    "low": 0.35,
    "medium": 0.65,
}
