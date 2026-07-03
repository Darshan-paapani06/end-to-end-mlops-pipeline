from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


def generate_synthetic_churn_data(rows: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic telecom/SaaS churn dataset for local demos and CI tests."""
    rng = np.random.default_rng(random_state)
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[0.56, 0.24, 0.20])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], rows, p=[0.35, 0.50, 0.15])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        rows,
        p=[0.42, 0.16, 0.22, 0.20],
    )
    tenure = rng.integers(1, 73, rows)
    monthly_charges = np.round(np.clip(rng.normal(70, 25, rows), 18, 120), 2)
    support_calls = np.clip(rng.poisson(1.6, rows), 0, 8)
    late_payments = np.clip(rng.poisson(0.7, rows), 0, 6)
    total_charges = np.round(np.clip(monthly_charges * tenure + rng.normal(0, 40, rows), 20, None), 2)

    logit = (
        -2.2
        + (contract_type == "Month-to-month") * 1.15
        + (internet_service == "Fiber optic") * 0.55
        + (payment_method == "Electronic check") * 0.45
        + (tenure < 12) * 0.8
        + support_calls * 0.22
        + late_payments * 0.3
        + (monthly_charges > 85) * 0.35
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    churn = (rng.random(rows) < churn_probability).astype(int)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{10000 + i}" for i in range(rows)],
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "support_calls_last_90d": support_calls,
            "late_payments_last_12m": late_payments,
            "churn": churn,
        }
    )


def load_training_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load training data from CSV, or generate synthetic data when no path is supplied."""
    if data_path is None:
        return generate_synthetic_churn_data()

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    df = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the training dataframe into model features and target."""
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].astype(int).copy()


def validate_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and order input fields for batch prediction."""
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Prediction input is missing required columns: {sorted(missing)}")
    cleaned = df.copy()
    for col in NUMERIC_FEATURES:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        cleaned[col] = cleaned[col].astype(str).fillna("Unknown")
    return cleaned[FEATURE_COLUMNS]
