import shap
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from backend.app.domain.explain_schemas import (
    FeatureAttribution,
    ShapExplanationResponse,
)
from backend.app.services.feature_extractor import FEATURE_COLUMNS


class NetworkShapExplainer:
    """
    TreeSHAP explanation engine providing fast, exact feature attributions
    and translating mathematical Shapley values into human-readable network diagnostics.
    """

    def __init__(self, lgb_model):
        self.model = lgb_model
        # Use TreeExplainer with model output 'raw' (log-odds)
        self.explainer = shap.TreeExplainer(self.model)
        exp_val = self.explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            self.expected_value = float(exp_val[-1])
        else:
            self.expected_value = float(exp_val)

    def explain_sample(
        self,
        features: pd.Series,
        prediction_id: str,
        predicted_risk: float,
        calibrated_risk: float,
        horizon_minutes: int = 15,
        top_k: int = 6,
    ) -> ShapExplanationResponse:
        """Computes exact SHAP attributions and returns structured operator explanation."""
        sample_df = pd.DataFrame([features[FEATURE_COLUMNS]])
        shap_vals = self.explainer.shap_values(sample_df)

        # In binary classification, shap_values can be [neg, pos] list, 3D array, 2D array, or 1D
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            vals = shap_vals[1][0]
        elif isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                vals = shap_vals[0, :, -1]
            elif shap_vals.ndim == 2:
                vals = shap_vals[0]
            else:
                vals = shap_vals
        else:
            vals = np.array(shap_vals).flatten()

        total_abs = np.sum(np.abs(vals)) + 1e-9
        attributions: List[FeatureAttribution] = []

        for col_name, shap_val in zip(FEATURE_COLUMNS, vals):
            feat_val = float(features[col_name])
            contrib_pct = round((abs(shap_val) / total_abs) * 100.0, 1)
            direction = "INCREASES_RISK" if shap_val > 0 else "DECREASES_RISK"
            desc = self._generate_diagnostic_description(col_name, feat_val, shap_val)

            attributions.append(
                FeatureAttribution(
                    feature_name=col_name,
                    feature_value=round(feat_val, 2),
                    shap_value=round(float(shap_val), 4),
                    contribution_pct=contrib_pct,
                    impact_direction=direction,
                    diagnostic_description=desc,
                )
            )

        # Sort by absolute SHAP impact
        attributions.sort(key=lambda x: abs(x.shap_value), reverse=True)

        top_risk = [a for a in attributions if a.impact_direction == "INCREASES_RISK"][:top_k]
        top_protective = [a for a in attributions if a.impact_direction == "DECREASES_RISK"][:top_k]

        summary = self._synthesize_operator_summary(top_risk, top_protective, calibrated_risk)

        return ShapExplanationResponse(
            prediction_id=prediction_id,
            horizon_minutes=horizon_minutes,
            base_value=round(self.expected_value, 4),
            predicted_risk=round(predicted_risk, 4),
            calibrated_risk=round(calibrated_risk, 4),
            top_risk_drivers=top_risk,
            top_protective_factors=top_protective,
            all_attributions=attributions[:12],
            operator_summary=summary,
        )

    def _generate_diagnostic_description(self, name: str, val: float, shap_v: float) -> str:
        """Translates raw feature telemetry and SHAP direction into operator terminology."""
        action = "elevating" if shap_v > 0 else "reducing"
        if "queue_slope" in name:
            return f"Queue fill rate ({val:+.2f}%/min) is {action} buffer overflow risk."
        if "queue_occupancy" in name:
            return f"Current queue depth at {val:.1f}% capacity is {action} packet drop probability."
        if "rtt_slope" in name:
            return f"RTT delay gradient ({val:+.2f}ms/min) is {action} latency instability."
        if "buffer_stress" in name:
            return f"Buffer stress index ({val:.1f}) is {action} risk of drop cliff."
        if "discards" in name:
            return f"Interface buffer discards ({val:.1f}/s) {action} hardware queue stress."
        if "bgp_flap" in name:
            return f"Routing protocol flap count ({int(val)}) {action} control plane risk."
        if "bandwidth_util" in name:
            return f"Bandwidth utilization ({val:.1f}%) {action} overall link strain."
        return f"{name} ({val:.1f}) is {action} predictive failure score."

    def _synthesize_operator_summary(
        self,
        top_risk: List[FeatureAttribution],
        top_protective: List[FeatureAttribution],
        risk: float,
    ) -> str:
        """Synthesizes high-level operator briefing."""
        if risk >= 0.70:
            primary = top_risk[0].diagnostic_description if top_risk else "Acute congestion pressure."
            secondary = top_risk[1].diagnostic_description if len(top_risk) > 1 else ""
            return f"CRITICAL PREDICTION ({risk*100:.1f}% risk): Impending failure driven primarily by: {primary} {secondary}"
        elif risk >= 0.35:
            primary = top_risk[0].diagnostic_description if top_risk else "Moderate buffer buildup."
            return f"ELEVATED RISK ({risk*100:.1f}% risk): Caution advised. Key factor: {primary}"
        else:
            prot = top_protective[0].diagnostic_description if top_protective else "Stable buffer and low jitter."
            return f"NORMAL SYSTEM HEALTH ({risk*100:.1f}% risk): Low risk of failure. Key stabilizing factor: {prot}"
