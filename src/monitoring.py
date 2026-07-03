from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import NUMERIC_FEATURES, PREDICTION_LOG_PATH, RISK_THRESHOLDS


def risk_band(probability: float) -> str:
    """Map churn probability to a business-readable risk label."""
    if probability < RISK_THRESHOLDS["low"]:
        return "Low"
    if probability < RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def retention_recommendation(probability: float) -> str:
    """Return a simple decision-support action for business users."""
    band = risk_band(probability)
    if band == "High":
        return "Prioritize retention call, discount review, and service-quality investigation."
    if band == "Medium":
        return "Monitor engagement and send targeted retention offer."
    return "Maintain standard engagement journey."


def log_prediction(payload: dict[str, Any], prediction: dict[str, Any], log_path: str | Path = PREDICTION_LOG_PATH) -> None:
    """Append an inference event to a JSONL log for lightweight monitoring."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
        "prediction": prediction,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")


def read_prediction_logs(log_path: str | Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI to estimate distribution drift between reference and current data."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(reference, quantiles))
    if len(cut_points) < 3:
        return 0.0

    reference_counts, _ = np.histogram(reference, bins=cut_points)
    current_counts, _ = np.histogram(current, bins=cut_points)

    reference_pct = np.clip(reference_counts / max(reference_counts.sum(), 1), 1e-6, None)
    current_pct = np.clip(current_counts / max(current_counts.sum(), 1), 1e-6, None)
    psi = np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))
    return round(float(psi), 4)


def drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict[str, float]:
    """Generate a compact numeric-feature drift report."""
    report = {}
    for column in NUMERIC_FEATURES:
        if column in reference_df.columns and column in current_df.columns:
            report[column] = population_stability_index(reference_df[column], current_df[column])
    return report
