import pytest
from backend.app.services.simulation_service import simulation_service
from backend.app.domain.simulation_schemas import SimulationScenarioRequest


def test_simulation_service_counterfactual():
    req = SimulationScenarioRequest(
        device_id="core-router-alpha",
        interface_id="xe-0/0/1",
        horizon_minutes=15,
        reroute_traffic_pct=35.0,
        ingress_rate_limit_pct=15.0,
        buffer_expansion_factor=1.5,
        enable_priority_queuing=True,
    )
    result = simulation_service.evaluate_scenario(req)

    assert result.baseline_risk >= 0.0
    assert result.counterfactual_risk <= result.baseline_risk
    assert result.risk_delta_pct <= 0.0
    assert result.counterfactual_rtt_ms <= result.baseline_rtt_ms
    assert result.verdict in ["OPTIMAL", "EFFECTIVE", "MARGINAL", "STABLE"]
