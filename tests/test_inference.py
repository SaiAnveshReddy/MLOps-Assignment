import pytest

from heart_mlops.inference import predict_risk


def test_predict_risk_returns_expected_keys(trained_model_path, sample_patient):
    result = predict_risk(sample_patient)
    assert set(result.keys()) == {
        "prediction", "risk_label", "confidence", "probability_disease"
    }
    assert result["prediction"] in (0, 1)
    assert result["risk_label"] in ("low_risk", "high_risk")
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["probability_disease"] <= 1.0


def test_predict_risk_raises_on_missing_field(trained_model_path, sample_patient):
    incomplete = dict(sample_patient)
    del incomplete["age"]
    with pytest.raises(ValueError, match="age"):
        predict_risk(incomplete)


def test_risk_label_matches_prediction(trained_model_path, sample_patient):
    result = predict_risk(sample_patient)
    expected_label = "high_risk" if result["prediction"] == 1 else "low_risk"
    assert result["risk_label"] == expected_label
