from pydantic import BaseModel, Field
from typing import List


class FeatureAttribution(BaseModel):
    """Single feature attribution from TreeSHAP."""
    feature_name: str
    feature_value: float
    shap_value: float
    contribution_pct: float
    impact_direction: str = Field(..., description="'INCREASES_RISK' or 'DECREASES_RISK'")
    diagnostic_description: str


class ShapExplanationResponse(BaseModel):
    """Explainability payload breaking down prediction drivers."""
    prediction_id: str
    horizon_minutes: int
    base_value: float = Field(..., description="Expected baseline log-odds / risk")
    predicted_risk: float
    calibrated_risk: float
    top_risk_drivers: List[FeatureAttribution]
    top_protective_factors: List[FeatureAttribution]
    all_attributions: List[FeatureAttribution]
    operator_summary: str
