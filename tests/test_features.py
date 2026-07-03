from src.data import generate_synthetic_churn_data, split_features_target
from src.modeling import build_model


def test_synthetic_data_has_expected_columns():
    df = generate_synthetic_churn_data(rows=25)
    assert "churn" in df.columns
    assert "monthly_charges" in df.columns
    assert len(df) == 25


def test_model_pipeline_can_fit_and_predict():
    df = generate_synthetic_churn_data(rows=80)
    x, y = split_features_target(df)
    model = build_model("logistic_regression")
    model.fit(x, y)
    probabilities = model.predict_proba(x.head(5))[:, 1]
    assert len(probabilities) == 5
    assert all(0 <= p <= 1 for p in probabilities)
