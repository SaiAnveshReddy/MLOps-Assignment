from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_returns_prediction(trained_model_path, sample_patient):
    response = client.post("/predict", json=sample_patient)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["risk_label"] in ("low_risk", "high_risk")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_endpoint_rejects_missing_field(trained_model_path, sample_patient):
    bad_payload = dict(sample_patient)
    del bad_payload["chol"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # pydantic validation error


def test_predict_endpoint_rejects_out_of_range_value(trained_model_path, sample_patient):
    bad_payload = dict(sample_patient)
    bad_payload["sex"] = 5  # only 0/1 allowed
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "cardiorisk_requests_total" in response.text
