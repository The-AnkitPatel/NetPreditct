import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from backend.app.core.config import settings


class NetworkAnomalyDetector:
    """
    Unsupervised real-time anomaly detection engine.
    Decoupled from future prediction: strictly answers 'What is abnormal right NOW?'
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.scaler = RobustScaler()
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = [
            "bandwidth_util_pct",
            "rtt_ms",
            "rtt_jitter_ms",
            "queue_occupancy_pct",
            "packet_loss_pct",
            "tcp_retrans_rate",
            "interface_discards_sec",
            "crc_errors_sec",
            "cpu_util_pct",
        ]
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "NetworkAnomalyDetector":
        """Fits Isolation Forest and RobustScaler on baseline telemetry."""
        X = df[self.feature_names].fillna(0.0)
        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)
        self.is_fitted = True
        return self

    def score_single(self, telemetry_row: pd.Series) -> Tuple[float, bool, str]:
        """
        Computes anomaly score in range [0.0, 1.0].
        Scores > ANOMALY_THRESHOLD (0.65) indicate anomalous present behavior.
        """
        if not self.is_fitted:
            # Fallback heuristic if not yet fit
            q = float(telemetry_row.get("queue_occupancy_pct", 0.0))
            loss = float(telemetry_row.get("packet_loss_pct", 0.0))
            score = min(1.0, (q / 100.0) * 0.6 + min(1.0, loss / 2.0) * 0.4)
            is_anom = score >= settings.ANOMALY_THRESHOLD
            return round(score, 3), is_anom, "Heuristic evaluation (unfitted)"

        sample = pd.DataFrame([telemetry_row[self.feature_names].fillna(0.0)])
        sample_scaled = self.scaler.transform(sample)

        # Isolation Forest decision_function returns negative for anomalies, positive for normal
        raw_score = self.iso_forest.decision_function(sample_scaled)[0]
        # Map to [0, 1] anomaly score where 1.0 is extremely anomalous
        # Typical decision_function range is approx [-0.3, 0.3]
        norm_score = float(np.clip(0.5 - (raw_score * 1.8), 0.0, 1.0))
        is_anom = norm_score >= settings.ANOMALY_THRESHOLD

        if is_anom:
            top_culprit = self._identify_top_anomaly_driver(sample.iloc[0])
            status_desc = f"Anomalous telemetry detected. Primary driver: {top_culprit}"
        else:
            status_desc = "Normal baseline operational behavior"

        return round(norm_score, 3), is_anom, status_desc

    def _identify_top_anomaly_driver(self, row: pd.Series) -> str:
        """Finds which raw metric is furthest into tail distribution."""
        if row["queue_occupancy_pct"] > 75.0:
            return f"Severe queue depth spike ({row['queue_occupancy_pct']:.1f}%)"
        if row["packet_loss_pct"] > 1.0:
            return f"Elevated packet loss ratio ({row['packet_loss_pct']:.2f}%)"
        if row["rtt_ms"] > 50.0:
            return f"High active probe RTT ({row['rtt_ms']:.1f}ms)"
        if row["interface_discards_sec"] > 5.0:
            return f"Hardware interface buffer discards ({row['interface_discards_sec']:.1f}/s)"
        if row["crc_errors_sec"] > 5.0:
            return f"CRC physical frame errors ({row['crc_errors_sec']:.1f}/s)"
        return "Multivariate telemetry deviation"
