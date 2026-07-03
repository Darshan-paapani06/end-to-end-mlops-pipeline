from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120, examples=[12])
    monthly_charges: float = Field(..., ge=0, examples=[89.90])
    total_charges: float = Field(..., ge=0, examples=[1080.50])
    contract_type: str = Field(..., examples=["Month-to-month"])
    internet_service: str = Field(..., examples=["Fiber optic"])
    payment_method: str = Field(..., examples=["Electronic check"])
    support_calls_last_90d: int = Field(..., ge=0, le=30, examples=[3])
    late_payments_last_12m: int = Field(..., ge=0, le=30, examples=[2])


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_band: str
    recommended_action: str
    features_used: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_stage: str
