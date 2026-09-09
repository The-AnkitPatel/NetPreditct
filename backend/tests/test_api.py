import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["is_model_trained"] is True


def test_api_predict_horizons(client):
    response = client.get("/api/v1/predict/horizons")
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert "horizons" in data
    assert "15" in data["horizons"]
    assert "primary_horizon" in data


def test_api_explain(client):
    response = client.get("/api/v1/explain/latest")
    assert response.status_code == 200
    data = response.json()
    assert "top_risk_drivers" in data
    assert "operator_summary" in data


def test_api_simulation(client):
    payload = {
        "device_id": "core-router-alpha",
        "interface_id": "xe-0/0/1",
        "horizon_minutes": 15,
        "reroute_traffic_pct": 30.0,
        "ingress_rate_limit_pct": 10.0,
        "buffer_expansion_factor": 1.5,
        "enable_priority_queuing": True,
    }
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] in ["OPTIMAL", "EFFECTIVE", "MARGINAL", "STABLE"]
    assert data["risk_delta_pct"] <= 0


def test_api_live_poll(client):
    response = client.post("/api/v1/telemetry/live-poll")
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "live_hardware_adapter"
    assert "record" in data
    assert "prediction" in data
    assert data["record"]["throughput_mbps"] >= 0


def test_api_prescribe(client):
    response = client.post("/api/v1/simulate/prescribe?device_id=core-router-alpha&interface_id=xe-0/0/1&horizon_minutes=15")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "recommended_scenario" in data
    assert "simulation" in data
    assert "rationale" in data


def test_api_recalibrate(client):
    response = client.post("/api/v1/health/recalibrate")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "RECALIBRATED"
    assert data["is_calibrated"] is True


