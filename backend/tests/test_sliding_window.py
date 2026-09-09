import pytest
from datetime import datetime
from backend.app.domain.telemetry_schemas import RawTelemetryRecord
from backend.app.services.sliding_window import SlidingWindowManager


def make_record(i: int, q: float = 20.0) -> RawTelemetryRecord:
    return RawTelemetryRecord(
        timestamp=datetime.utcnow(),
        device_id="test-router",
        interface_id="xe-0/0/1",
        bandwidth_util_pct=50.0,
        throughput_mbps=5000.0,
        rtt_ms=15.0,
        rtt_jitter_ms=1.2,
        queue_occupancy_pct=q,
        packet_loss_pct=0.0,
        tcp_retrans_rate=0.1,
        interface_discards_sec=0.0,
        crc_errors_sec=0.0,
        cpu_util_pct=30.0,
        memory_util_pct=40.0,
        bgp_flap_count=0,
    )


def test_sliding_window_capacity_and_fifo():
    wm = SlidingWindowManager(capacity=5)
    for i in range(10):
        wm.append(make_record(i, q=float(i * 10)))

    # Capacity should be capped at 5
    assert wm.count("test-router", "xe-0/0/1") == 5
    latest = wm.get_latest("test-router", "xe-0/0/1")
    assert latest is not None
    assert latest.queue_occupancy_pct == 90.0

    df = wm.to_dataframe("test-router", "xe-0/0/1")
    assert len(df) == 5
    assert list(df["queue_occupancy_pct"]) == [50.0, 60.0, 70.0, 80.0, 90.0]
