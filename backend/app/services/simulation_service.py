import uuid
import numpy as np
import pandas as pd
from backend.app.domain.simulation_schemas import (
    SimulationScenarioRequest,
    SimulationResponse,
)
from backend.app.services.sliding_window import window_manager
from backend.app.services.feature_extractor import (
    extract_single_step_features,
    FEATURE_COLUMNS,
)
from backend.app.services.prediction_service import prediction_service
from backend.app.core.config import settings


class SimulationService:
    """Evaluates counterfactual operator interventions through the calibrated ML model."""

    def evaluate_scenario(self, req: SimulationScenarioRequest) -> SimulationResponse:
        history_df = window_manager.to_dataframe(req.device_id, req.interface_id)
        if len(history_df) == 0:
            # Synthetic baseline if buffer is currently empty
            features = pd.Series(
                {
                    "bandwidth_util_pct": 82.0,
                    "rtt_ms": 58.0,
                    "queue_occupancy_pct": 78.0,
                    "packet_loss_pct": 0.8,
                    "tcp_retrans_rate": 4.5,
                    "interface_discards_sec": 6.0,
                    "rtt_slope_5m": 2.2,
                    "queue_slope_5m": 2.8,
                    "buffer_stress_index": 63.9,
                    "congestion_pressure_score": 45.0,
                }
            )
            for c in FEATURE_COLUMNS:
                if c not in features:
                    features[c] = 0.0
        else:
            features = extract_single_step_features(history_df)

        h = req.horizon_minutes

        # 1. Evaluate baseline risk
        _, base_risk = prediction_service.predictor.predict_horizon(features, h)
        base_rtt, _ = prediction_service.predictor.forecast_continuous(features)
        base_queue = float(features.get("queue_occupancy_pct", 70.0))

        # 2. Construct Counterfactual Feature Vector
        cf_features = features.copy()

        # Traffic reduction multiplier from rerouting and shaping
        traffic_reduction = (1.0 - req.reroute_traffic_pct / 100.0) * (
            1.0 - req.ingress_rate_limit_pct / 100.0
        )
        cf_features["bandwidth_util_pct"] = np.clip(
            features["bandwidth_util_pct"] * traffic_reduction, 5.0, 100.0
        )

        # Queue depth reduction: drops non-linearly with traffic reduction and buffer expansion
        buffer_divisor = req.buffer_expansion_factor
        cf_queue = (features["queue_occupancy_pct"] * (traffic_reduction ** 1.5)) / buffer_divisor
        cf_features["queue_occupancy_pct"] = np.clip(cf_queue, 2.0, 100.0)

        # Slopes dampen under intervention
        cf_features["queue_slope_5m"] = np.clip(features["queue_slope_5m"] * traffic_reduction - 0.5, -5.0, 5.0)
        cf_features["rtt_slope_5m"] = np.clip(features["rtt_slope_5m"] * traffic_reduction - 0.4, -5.0, 5.0)

        if req.enable_priority_queuing:
            cf_features["packet_loss_pct"] = max(0.0, float(features["packet_loss_pct"]) * 0.3)
            cf_features["tcp_retrans_rate"] = max(0.0, float(features["tcp_retrans_rate"]) * 0.4)

        # Recalculate interaction features
        cf_features["buffer_stress_index"] = (
            cf_features["queue_occupancy_pct"] * cf_features["bandwidth_util_pct"]
        ) / 100.0
        cf_features["congestion_pressure_score"] = np.clip(
            (cf_features["queue_occupancy_pct"] * 0.4)
            + (np.maximum(0.0, cf_features["rtt_slope_5m"]) * 5.0)
            + (cf_features["packet_loss_pct"] * 10.0),
            0.0,
            150.0,
        )

        # 3. Predict counterfactual outcome
        _, cf_risk = prediction_service.predictor.predict_horizon(cf_features, h)
        cf_rtt, _ = prediction_service.predictor.forecast_continuous(cf_features)
        cf_rtt = max(12.0, cf_rtt * traffic_reduction)

        # 4. Compute risk delta and verdict
        risk_delta = ((cf_risk - base_risk) / max(0.01, base_risk)) * 100.0

        if base_risk < settings.RISK_NORMAL_MAX:
            verdict = "STABLE"
            guidance = f"Baseline risk is already nominal ({base_risk*100:.1f}%). Pre-emptive action maintains headroom."
        elif risk_delta <= -50.0:
            verdict = "OPTIMAL"
            guidance = f"Highly effective intervention. Predicted risk plummets by {abs(risk_delta):.1f}%."
        elif risk_delta <= -20.0:
            verdict = "EFFECTIVE"
            guidance = f"Viable mitigation. Mitigates queue collapse with {abs(risk_delta):.1f}% risk reduction."
        elif risk_delta < 0.0:
            verdict = "MARGINAL"
            guidance = "Marginal benefit. Suggest increasing reroute percentage to >= 25%."
        else:
            verdict = "INEFFECTIVE"
            guidance = "Insufficient reduction. Stress remains above critical threshold."

        explanation = (
            f"Diverting {req.reroute_traffic_pct:.0f}% traffic and rate limiting {req.ingress_rate_limit_pct:.0f}% "
            f"lowers link utilization from {features['bandwidth_util_pct']:.1f}% to {cf_features['bandwidth_util_pct']:.1f}%, "
            f"reducing queue occupancy from {base_queue:.1f}% to {cf_features['queue_occupancy_pct']:.1f}%. "
            f"Forecasted failure risk drops from {base_risk*100:.1f}% to {cf_risk*100:.1f}%."
        )

        return SimulationResponse(
            scenario_id=f"SIM-{uuid.uuid4().hex[:8].upper()}",
            device_id=req.device_id,
            interface_id=req.interface_id,
            horizon_minutes=h,
            baseline_risk=round(base_risk, 4),
            counterfactual_risk=round(cf_risk, 4),
            risk_delta_pct=round(risk_delta, 1),
            baseline_rtt_ms=round(base_rtt, 1),
            counterfactual_rtt_ms=round(cf_rtt, 1),
            baseline_queue_pct=round(base_queue, 1),
            counterfactual_queue_pct=round(float(cf_features["queue_occupancy_pct"]), 1),
            verdict=verdict,
            operator_guidance=guidance,
            counterfactual_explanation=explanation,
        )


simulation_service = SimulationService()
