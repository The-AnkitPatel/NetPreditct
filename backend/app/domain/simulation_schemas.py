from pydantic import BaseModel, Field
from typing import Optional


class SimulationScenarioRequest(BaseModel):
    """Parameters for what-if counterfactual intervention analysis."""
    device_id: str = Field(..., example="edge-rt-01")
    interface_id: str = Field(..., example="xe-0/0/1")
    horizon_minutes: int = Field(default=15, description="Prediction horizon to simulate against")

    # Candidate Operator Actions
    reroute_traffic_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of ingress traffic to divert to redundant path"
    )
    ingress_rate_limit_pct: float = Field(
        default=0.0, ge=0.0, le=90.0,
        description="Policing / traffic shaping rate reduction %"
    )
    buffer_expansion_factor: float = Field(
        default=1.0, ge=1.0, le=3.0,
        description="QoS burst queue buffer size multiplier (1.0 = baseline)"
    )
    enable_priority_queuing: bool = Field(
        default=False,
        description="Enable weighted fair queuing for high-priority flows"
    )


class SimulationResponse(BaseModel):
    """Evaluated outcome of candidate network intervention."""
    scenario_id: str
    device_id: str
    interface_id: str
    horizon_minutes: int

    # Baseline vs Counterfactual comparison
    baseline_risk: float = Field(..., description="Predicted failure risk without intervention")
    counterfactual_risk: float = Field(..., description="Predicted failure risk under scenario")
    risk_delta_pct: float = Field(..., description="Relative risk change percentage")
    
    baseline_rtt_ms: float
    counterfactual_rtt_ms: float
    baseline_queue_pct: float
    counterfactual_queue_pct: float

    verdict: str = Field(..., description="Operator verdict: OPTIMAL, EFFECTIVE, MARGINAL, or INEFFECTIVE")
    operator_guidance: str
    counterfactual_explanation: str
