import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from backend.app.core.config import settings
from backend.app.domain.telemetry_schemas import RawTelemetryRecord, TelemetrySnapshot
from backend.app.domain.prediction_schemas import (
    PredictionResponse,
    HorizonPrediction,
    RiskLevel,
    HistoricalIncident,
)
from backend.app.services.sliding_window import window_manager
from backend.app.services.feature_extractor import (
    extract_single_step_features,
    FEATURE_COLUMNS,
)
from backend.app.ml.anomaly_detector import NetworkAnomalyDetector
from backend.app.ml.predictor_model import MultiHorizonPredictor
from backend.app.ml.conformal_predictor import ConformalPredictor
from backend.app.ml.shap_explainer import NetworkShapExplainer


class PredictionService:
    """Orchestration service for real-time anomaly detection and predictive inference."""

    def __init__(self):
        self.anomaly_detector = NetworkAnomalyDetector()
        self.predictor = MultiHorizonPredictor()
        self.conformal = ConformalPredictor()
        self.explainer: Optional[NetworkShapExplainer] = None
        self.cached_predictions: Dict[str, Tuple[PredictionResponse, pd.Series]] = {}
        self.historical_incidents: List[HistoricalIncident] = []
        self._init_defaults()

    def _init_defaults(self):
        """Populate initial representative historical incidents for validation."""
        self.historical_incidents = [
            HistoricalIncident(
                incident_id="INC-2026-0814",
                timestamp=datetime(2026, 9, 1, 8, 30),
                device_id="core-router-alpha",
                interface_id="xe-0/0/1",
                failure_type="Bufferbloat Queue Collapse",
                predicted_risk=0.88,
                actual_outcome="Congestion Event Averted (Rerouted)",
                lead_time_minutes=18,
                was_prevented=True,
                root_cause_driver="Queue depth slope (+3.2%/min)",
            ),
            HistoricalIncident(
                incident_id="INC-2026-0815",
                timestamp=datetime(2026, 9, 2, 14, 15),
                device_id="core-router-alpha",
                interface_id="xe-0/0/1",
                failure_type="Microburst Drop Surge",
                predicted_risk=0.92,
                actual_outcome="Traffic Throttled via Ingress Police",
                lead_time_minutes=12,
                was_prevented=True,
                root_cause_driver="Throughput burst (+2400Mbps)",
            ),
            HistoricalIncident(
                incident_id="INC-2026-0816",
                timestamp=datetime(2026, 9, 3, 22, 40),
                device_id="core-router-alpha",
                interface_id="xe-0/0/1",
                failure_type="BGP Route Flap Cascade",
                predicted_risk=0.84,
                actual_outcome="Interface Dampened",
                lead_time_minutes=25,
                was_prevented=True,
                root_cause_driver="BGP flap count (>6/min)",
            ),
        ]

    def set_explainer(self, explainer: NetworkShapExplainer):
        self.explainer = explainer

    def process_telemetry(self, record: RawTelemetryRecord) -> PredictionResponse:
        """Processes an incoming telemetry point through the full inference pipeline."""
        # 1. Update sliding window ring buffer
        window_manager.append(record)

        # 2. Extract features from sliding window
        history_df = window_manager.to_dataframe(record.device_id, record.interface_id)
        features = extract_single_step_features(history_df)

        # 3. Compute distinct present anomaly score (What is abnormal NOW)
        anom_score, is_anom, anom_status = self.anomaly_detector.score_single(history_df.iloc[-1])

        # 4. Predict multi-horizon future risk (What will fail NEXT)
        horizons_dict: Dict[int, HorizonPrediction] = {}
        pred_rtt, pred_loss = self.predictor.forecast_continuous(features)
        conf_lower_rtt, conf_upper_rtt = self.conformal.predict_interval_rtt(pred_rtt)

        for h in settings.HORIZONS_MINUTES:
            raw_risk, cal_risk = self.predictor.predict_horizon(features, h)
            # Determine operational risk tier
            if cal_risk >= settings.RISK_ELEVATED_MAX:
                level = RiskLevel.CRITICAL
            elif cal_risk >= settings.RISK_NORMAL_MAX:
                level = RiskLevel.ELEVATED
            else:
                level = RiskLevel.LOW

            horizons_dict[h] = HorizonPrediction(
                horizon_minutes=h,
                failure_risk_score=raw_risk,
                calibrated_probability=cal_risk,
                risk_level=level,
                predicted_rtt_ms=pred_rtt,
                predicted_packet_loss_pct=pred_loss,
                conformal_lower_bound_rtt=conf_lower_rtt,
                conformal_upper_bound_rtt=conf_upper_rtt,
            )

        primary_h = horizons_dict[settings.PRIMARY_HORIZON_MINUTES]
        prediction_id = f"PRED-{uuid.uuid4().hex[:8].upper()}"

        # 5. Recommendation guidance
        if primary_h.risk_level == RiskLevel.CRITICAL:
            rec_action = "IMMEDIATE MITIGATION: Reroute 25-40% ingress traffic to secondary path."
        elif primary_h.risk_level == RiskLevel.ELEVATED:
            rec_action = "MONITOR CLOSELY: Pre-stage egress traffic policing and observe queue slope."
        else:
            rec_action = "NORMAL OPERATIONS: Telemetry within calibrated baseline bounds."

        # Top raw drivers
        drivers = []
        if features.get("queue_occupancy_pct", 0) > 65:
            drivers.append(f"Queue Occupancy ({features['queue_occupancy_pct']:.1f}%)")
        if features.get("rtt_slope_5m", 0) > 1.0:
            drivers.append(f"RTT Delay Gradient (+{features['rtt_slope_5m']:.2f}ms/min)")
        if features.get("packet_loss_pct", 0) > 0.2:
            drivers.append(f"Packet Loss ({features['packet_loss_pct']:.2f}%)")
        if not drivers:
            drivers = ["Nominal link utilization", "Stable hardware queue depths"]

        response = PredictionResponse(
            prediction_id=prediction_id,
            timestamp=record.timestamp,
            device_id=record.device_id,
            interface_id=record.interface_id,
            current_anomaly_score=anom_score,
            is_current_anomaly=is_anom,
            anomaly_status=anom_status,
            horizons=horizons_dict,
            primary_horizon=primary_h,
            recommended_action=rec_action,
            primary_drivers=drivers,
        )

        # Cache for SHAP inspection
        self.cached_predictions[prediction_id] = (response, features)
        return response


# Global singleton service
prediction_service = PredictionService()
