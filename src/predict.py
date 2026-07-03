from __future__ import annotations

import pandas as pd

from src.config import FEATURE_COLUMNS
from src.data import validate_prediction_frame
from src.model_registry import load_model, load_metadata
from src.monitoring import retention_recommendation, risk_band


def predict_dataframe(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Generate churn predictions for a dataframe of customer records."""
    model = model or load_model()
    features = validate_prediction_frame(df)
    probabilities = model.predict_proba(features)[:, 1]
    threshold = float(load_metadata().get("threshold", 0.5))
    output = df.copy()
    output["churn_probability"] = probabilities.round(4)
    output["churn_prediction"] = (probabilities >= threshold).astype(int)
    output["risk_band"] = [risk_band(float(p)) for p in probabilities]
    output["recommended_action"] = [retention_recommendation(float(p)) for p in probabilities]
    return output


def predict_single(payload: dict, model=None) -> dict:
    """Generate a prediction for one customer payload."""
    frame = pd.DataFrame([payload])
    result = predict_dataframe(frame, model=model).iloc[0].to_dict()
    return {
        "churn_probability": float(result["churn_probability"]),
        "churn_prediction": int(result["churn_prediction"]),
        "risk_band": result["risk_band"],
        "recommended_action": result["recommended_action"],
        "features_used": FEATURE_COLUMNS,
    }
