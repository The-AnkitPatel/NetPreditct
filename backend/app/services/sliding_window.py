import threading
from collections import deque
from typing import List, Optional, Dict
import pandas as pd
from backend.app.domain.telemetry_schemas import RawTelemetryRecord
from backend.app.core.config import settings


class SlidingWindowManager:
    """Thread-safe circular ring buffer for sliding window telemetry caching."""

    def __init__(self, capacity: int = settings.SLIDING_WINDOW_CAPACITY):
        self.capacity = capacity
        self._buffers: Dict[str, deque] = {}  # Key: "device_id:interface_id"
        self._lock = threading.Lock()

    def _get_key(self, device_id: str, interface_id: str) -> str:
        return f"{device_id}:{interface_id}"

    def append(self, record: RawTelemetryRecord) -> None:
        """Appends a new raw telemetry sample to the appropriate window."""
        key = self._get_key(record.device_id, record.interface_id)
        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = deque(maxlen=self.capacity)
            self._buffers[key].append(record)

    def append_batch(self, records: List[RawTelemetryRecord]) -> None:
        """Atomically appends a batch of telemetry records."""
        with self._lock:
            for rec in records:
                key = self._get_key(rec.device_id, rec.interface_id)
                if key not in self._buffers:
                    self._buffers[key] = deque(maxlen=self.capacity)
                self._buffers[key].append(rec)

    def get_latest(self, device_id: str, interface_id: str) -> Optional[RawTelemetryRecord]:
        """Returns the most recent sample for the interface."""
        key = self._get_key(device_id, interface_id)
        with self._lock:
            buf = self._buffers.get(key)
            return buf[-1] if buf else None

    def get_window_records(
        self, device_id: str, interface_id: str, count: Optional[int] = None
    ) -> List[RawTelemetryRecord]:
        """Retrieves historical records up to count or full buffer."""
        key = self._get_key(device_id, interface_id)
        with self._lock:
            buf = self._buffers.get(key)
            if not buf:
                return []
            records = list(buf)
            if count is not None:
                records = records[-count:]
            return records

    def to_dataframe(
        self, device_id: str, interface_id: str, count: Optional[int] = None
    ) -> pd.DataFrame:
        """Converts buffered window records to a clean Pandas DataFrame for feature extraction."""
        records = self.get_window_records(device_id, interface_id, count)
        if not records:
            return pd.DataFrame()
        data = [r.model_dump() for r in records]
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def count(self, device_id: str, interface_id: str) -> int:
        """Current number of buffered samples."""
        key = self._get_key(device_id, interface_id)
        with self._lock:
            buf = self._buffers.get(key)
            return len(buf) if buf else 0

    def clear(self) -> None:
        """Clears all sliding windows."""
        with self._lock:
            self._buffers.clear()


# Global singleton instance
window_manager = SlidingWindowManager()
