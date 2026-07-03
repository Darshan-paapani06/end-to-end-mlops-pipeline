from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import sys
import os

# This forces Python to look in the parent directory for the 'src' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import METADATA_PATH
from src.model_registry import load_metadata, load_model
from src.monitoring import read_prediction_logs
from src.predict import predict_dataframe, predict_single

st.set_page_config(page_title="MLOps Churn Monitoring", page_icon="🚀", layout="wide")
st.title("🚀 End-to-End MLOps Pipeline Dashboard")
st.caption("Model serving, prediction simulation, model metadata, and lightweight monitoring in one place.")

try:
    model = load_model()
    metadata = load_metadata()
except Exception as exc:
    st.error(f"Model is not ready: {exc}")
    st.code("python -m src.train --data-path data/raw/sample_churn.csv")
    st.stop()

metric_cols = st.columns(4)
metrics = metadata.get("metrics", {})
metric_cols[0].metric("Model", metadata.get("model_type", "unknown"))
metric_cols[1].metric("Recall", metrics.get("recall", "n/a"))
metric_cols[2].metric("ROC AUC", metrics.get("roc_auc", "n/a"))
metric_cols[3].metric("Threshold", metadata.get("threshold", "n/a"))

st.divider()
left, right = st.columns([1, 1.2])
with left:
    st.subheader("Single-customer prediction")
    payload = {
        "tenure_months": st.slider("Tenure months", 0, 120, 8),
        "monthly_charges": st.number_input("Monthly charges", 0.0, 500.0, 92.5),
        "total_charges": st.number_input("Total charges", 0.0, 50000.0, 780.0),
        "contract_type": st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"]),
        "internet_service": st.selectbox("Internet service", ["DSL", "Fiber optic", "No"]),
        "payment_method": st.selectbox(
            "Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
        ),
        "support_calls_last_90d": st.slider("Support calls last 90 days", 0, 10, 3),
        "late_payments_last_12m": st.slider("Late payments last 12 months", 0, 10, 1),
    }
    if st.button("Predict churn risk", type="primary"):
        result = predict_single(payload, model=model)
        st.metric("Churn probability", f"{result['churn_probability']:.1%}")
        st.success(f"Risk band: {result['risk_band']}")
        st.info(result["recommended_action"])

with right:
    st.subheader("Batch prediction")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded:
        frame = pd.read_csv(uploaded)
        predictions = predict_dataframe(frame, model=model)
        st.dataframe(predictions, use_container_width=True)
        fig = px.histogram(predictions, x="risk_band", title="Predicted customer risk distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download predictions",
            predictions.to_csv(index=False),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )

st.divider()
st.subheader("Prediction monitoring logs")
logs = read_prediction_logs()
if logs.empty:
    st.warning("No prediction logs yet. Use the API or single-customer prediction to create monitoring events.")
else:
    st.dataframe(logs.tail(50), use_container_width=True)

with st.expander("Model metadata"):
    st.json(metadata)
    st.caption(f"Metadata source: {METADATA_PATH}")
