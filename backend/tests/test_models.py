import pytest
import numpy as np
import pandas as pd
from backend.app.services.prediction_service import prediction_service
from backend.app.domain.telemetry_schemas import RawTelemetryRecord


def test_prediction_service_pipeline():
    # Submit nominal telemetry record
    nominal = RawTelemetryRecord(
        device_id="test-rt",
        interface_id="xe-0/0/1",
        bandwidth_util_pct=35.0,
        throughput_mbps=3500.0,
        rtt_ms=14.5,
        rtt_jitter_ms=1.1,
        queue_occupancy_pct=18.0,
        packet_loss_pct=0.0,
        tcp_retrans_rate=0.02,
        interface_discards_sec=0.0,
        crc_errors_sec=0.0,
        cpu_util_pct=25.0,
        memory_util_pct=35.0,
        bgp_flap_count=0,
    )
    resp_nom = prediction_service.process_telemetry(nominal)
    assert not resp_nom.is_current_anomaly
    assert resp_nom.primary_horizon.calibrated_probability < 0.50

    # Submit severe congestion spike record
    spike = RawTelemetryRecord(
        device_id="test-rt",
        interface_id="xe-0/0/1",
        bandwidth_util_pct=96.0,
        throughput_mbps=9600.0,
        rtt_ms=85.0,
        rtt_jitter_ms=12.5,
        queue_occupancy_pct=92.0,
        packet_loss_pct=3.5,
        tcp_retrans_rate=15.0,
        interface_discards_sec=42.0,
        crc_errors_sec=8.0,
        cpu_util_pct=94.0,
        memory_util_pct=88.0,
        bgp_flap_count=3,
    )
    resp_spike = prediction_service.process_telemetry(spike)
    assert resp_spike.primary_horizon.failure_risk_score > resp_nom.primary_horizon.failure_risk_score
