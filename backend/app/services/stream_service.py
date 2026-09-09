import threading
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from backend.app.services.dataset_loader import ensure_dataset_exists
from backend.app.domain.telemetry_schemas import RawTelemetryRecord, TelemetrySnapshot
from backend.app.domain.prediction_schemas import PredictionResponse
from backend.app.services.prediction_service import prediction_service


class TelemetryStreamService:
    """Manages telemetry replay timeline and real-time streaming state."""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.current_idx: int = 0
        self.is_playing: bool = True
        self.playback_speed: float = 1.0
        self._lock = threading.Lock()
        self._initialize()

    def _initialize(self):
        self.df = ensure_dataset_exists()
        # Seed the first 45 steps so sliding window has immediate history
        seed_count = min(45, len(self.df))
        for i in range(seed_count):
            rec = self._row_to_record(self.df.iloc[i])
            prediction_service.process_telemetry(rec)
        self.current_idx = seed_count

    def _row_to_record(self, row: pd.Series) -> RawTelemetryRecord:
        return RawTelemetryRecord(
            timestamp=pd.to_datetime(row["timestamp"]),
            device_id=str(row["device_id"]),
            interface_id=str(row["interface_id"]),
            bandwidth_util_pct=float(row["bandwidth_util_pct"]),
            throughput_mbps=float(row["throughput_mbps"]),
            rtt_ms=float(row["rtt_ms"]),
            rtt_jitter_ms=float(row["rtt_jitter_ms"]),
            queue_occupancy_pct=float(row["queue_occupancy_pct"]),
            packet_loss_pct=float(row["packet_loss_pct"]),
            tcp_retrans_rate=float(row["tcp_retrans_rate"]),
            interface_discards_sec=float(row["interface_discards_sec"]),
            crc_errors_sec=float(row["crc_errors_sec"]),
            cpu_util_pct=float(row["cpu_util_pct"]),
            memory_util_pct=float(row["memory_util_pct"]),
            bgp_flap_count=int(row["bgp_flap_count"]),
        )

    def advance_step(self) -> Tuple[RawTelemetryRecord, PredictionResponse]:
        """Steps forward by 1 telemetry interval and runs the full predictive pipeline."""
        with self._lock:
            if self.df is None or len(self.df) == 0:
                self.df = ensure_dataset_exists()

            if self.current_idx >= len(self.df):
                self.current_idx = 0  # Loop replay

            row = self.df.iloc[self.current_idx]
            self.current_idx += 1
            record = self._row_to_record(row)

        pred_resp = prediction_service.process_telemetry(record)
        return record, pred_resp

    def jump_to_incident(self) -> Tuple[RawTelemetryRecord, PredictionResponse]:
        """Jumps playhead directly to a severe congestion/failure window (e.g. index 495)."""
        with self._lock:
            self.current_idx = 495
        return self.advance_step()

    def reset_stream(self) -> Tuple[RawTelemetryRecord, PredictionResponse]:
        """Resets stream back to step 45."""
        with self._lock:
            self.current_idx = 45
        return self.advance_step()

    def get_progress(self) -> Dict[str, Any]:
        """Returns replay status and timeline position."""
        total = len(self.df) if self.df is not None else 0
        return {
            "current_step": self.current_idx,
            "total_steps": total,
            "is_playing": self.is_playing,
            "playback_speed": self.playback_speed,
        }


stream_service = TelemetryStreamService()
