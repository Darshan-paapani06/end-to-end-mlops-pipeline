from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_endpoint():
    payload = {
        "tenure_months": 6,
        "monthly_charges": 95.0,
        "total_charges": 570.0,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "support_calls_last_90d": 4,
        "late_payments_last_12m": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_band"] in {"Low", "Medium", "High"}
