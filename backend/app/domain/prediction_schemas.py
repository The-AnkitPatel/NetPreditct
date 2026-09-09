from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"


class HorizonPrediction(BaseModel):
    """Forecast for a specific future prediction horizon (e.g. T+5m, T+15m, T+30m)."""
    horizon_minutes: int = Field(..., description="Minutes ahead into the future")
    failure_risk_score: float = Field(..., ge=0.0, le=1.0, description="Raw model probability")
    calibrated_probability: float = Field(..., ge=0.0, le=1.0, description="Isotonically calibrated risk")
    risk_level: RiskLevel
    predicted_rtt_ms: float = Field(..., description="Forecasted RTT in ms")
    predicted_packet_loss_pct: float = Field(..., description="Forecasted packet loss %")
    conformal_lower_bound_rtt: float = Field(..., description="90% Conformal confidence lower bound")
    conformal_upper_bound_rtt: float = Field(..., description="90% Conformal confidence upper bound")


class PredictionResponse(BaseModel):
    """Unified network intelligence response combining anomaly and multi-horizon prediction."""
    prediction_id: str
    timestamp: datetime
    device_id: str
    interface_id: str

    # Distinct Anomaly Detection (What is happening NOW)
    current_anomaly_score: float = Field(..., ge=0.0, le=1.0)
    is_current_anomaly: bool
    anomaly_status: str

    # Predictive Horizons (What is likely to happen NEXT)
    horizons: Dict[int, HorizonPrediction]
    primary_horizon: HorizonPrediction
    recommended_action: str
    primary_drivers: List[str]


class HistoricalIncident(BaseModel):
    """Historical incident record for prediction-vs-actual validation."""
    incident_id: str
    timestamp: datetime
    device_id: str
    interface_id: str
    failure_type: str
    predicted_risk: float
    actual_outcome: str
    lead_time_minutes: int
    was_prevented: bool
    root_cause_driver: str
