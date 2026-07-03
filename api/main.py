from __future__ import annotations

import io
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from src.config import MODEL_STAGE
from src.model_registry import load_metadata, load_model
from src.monitoring import log_prediction, read_prediction_logs
from src.predict import predict_dataframe, predict_single

REQUEST_COUNT = Counter("prediction_requests_total", "Total number of prediction requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction request latency")

app = FastAPI(
    title="End-to-End MLOps Churn Prediction API",
    description="Production-style ML API with model versioning, CI/CD, Docker, MLflow, and monitoring hooks.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_model():
    return load_model()


@app.get("/", tags=["System"])
def root() -> dict:
    return {
        "message": "MLOps churn prediction API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    try:
        get_model()
        model_loaded = True
    except Exception:
        model_loaded = False
    return HealthResponse(status="ok", model_loaded=model_loaded, model_stage=MODEL_STAGE)


@app.get("/model-info", tags=["Model"])
def model_info() -> dict:
    metadata = load_metadata()
    if not metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found. Train the model first.")
    return metadata


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: CustomerFeatures) -> PredictionResponse:
    REQUEST_COUNT.inc()
    with PREDICTION_LATENCY.time():
        try:
            prediction = predict_single(payload.model_dump(), model=get_model())
            log_prediction(payload.model_dump(), prediction)
            return PredictionResponse(**prediction)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/batch-predict", tags=["Prediction"])
async def batch_predict(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".xlsx")):
        raise HTTPException(status_code=400, detail="Upload a CSV or XLSX file.")

    content = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
        predictions = predict_dataframe(df, model=get_model())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    buffer = io.StringIO()
    predictions.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.get("/monitoring/prediction-logs", tags=["Monitoring"])
def prediction_logs(limit: int = 20) -> dict:
    logs = read_prediction_logs()
    if logs.empty:
        return {"count": 0, "events": []}
    return {"count": int(len(logs)), "events": logs.tail(limit).to_dict(orient="records")}


@app.get("/metrics", tags=["Monitoring"])
def prometheus_metrics():
    return StreamingResponse(iter([generate_latest()]), media_type=CONTENT_TYPE_LATEST)
