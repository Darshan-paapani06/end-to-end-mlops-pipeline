import pandas as pd

from src.monitoring import drift_report, population_stability_index, retention_recommendation, risk_band


def test_risk_band_mapping():
    assert risk_band(0.10) == "Low"
    assert risk_band(0.50) == "Medium"
    assert risk_band(0.90) == "High"


def test_recommendation_is_business_readable():
    recommendation = retention_recommendation(0.92)
    assert "retention" in recommendation.lower() or "discount" in recommendation.lower()


def test_psi_returns_non_negative_value():
    reference = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    current = pd.Series([2, 3, 4, 5, 6, 7, 8, 9])
    assert population_stability_index(reference, current) >= 0


def test_drift_report_contains_numeric_feature():
    reference = pd.DataFrame({"monthly_charges": [10, 20, 30], "tenure_months": [1, 2, 3]})
    current = pd.DataFrame({"monthly_charges": [12, 22, 35], "tenure_months": [1, 4, 6]})
    report = drift_report(reference, current)
    assert "monthly_charges" in report
