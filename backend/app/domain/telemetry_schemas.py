from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class RawTelemetryRecord(BaseModel):
    """Raw network telemetry sample from a network node or interface."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    device_id: str = Field(..., example="edge-rt-01")
    interface_id: str = Field(..., example="xe-0/0/1")

    # Traffic and throughput
    bandwidth_util_pct: float = Field(..., ge=0.0, le=100.0, description="Link bandwidth utilization %")
    throughput_mbps: float = Field(..., ge=0.0, description="Measured line-rate throughput in Mbps")

    # Latency and Jitter
    rtt_ms: float = Field(..., ge=0.0, description="Active probe Round-Trip Time in ms")
    rtt_jitter_ms: float = Field(..., ge=0.0, description="RTT variance/jitter in ms")

    # Queueing & Drops (Impending Congestion Signs)
    queue_occupancy_pct: float = Field(..., ge=0.0, le=100.0, description="Hardware queue buffer depth %")
    packet_loss_pct: float = Field(..., ge=0.0, le=100.0, description="Measured packet drop ratio %")
    tcp_retrans_rate: float = Field(..., ge=0.0, description="TCP retransmissions per second")

    # Interface Health & Hardware Counters
    interface_discards_sec: float = Field(..., ge=0.0, description="Interface buffer discards per second")
    crc_errors_sec: float = Field(..., ge=0.0, description="CRC frame alignment error rate")

    # Node Resource Pressure
    cpu_util_pct: float = Field(..., ge=0.0, le=100.0, description="Control/Data plane CPU %")
    memory_util_pct: float = Field(..., ge=0.0, le=100.0, description="Packet buffer RAM %")
    bgp_flap_count: int = Field(default=0, ge=0, description="BGP/OSPF route flap count")


class TelemetryIngestBatch(BaseModel):
    """Batch payload for high-throughput telemetry ingestion."""
    source: str = Field(default="live_collector")
    records: List[RawTelemetryRecord]


class TelemetrySnapshot(BaseModel):
    """Aggregated current telemetry view for UI dashboards."""
    timestamp: datetime
    device_id: str
    interface_id: str
    bandwidth_util_pct: float
    throughput_mbps: float
    rtt_ms: float
    rtt_jitter_ms: float
    queue_occupancy_pct: float
    packet_loss_pct: float
    tcp_retrans_rate: float
    interface_discards_sec: float
    cpu_util_pct: float
    memory_util_pct: float
    bgp_flap_count: int
    is_anomaly: bool
    anomaly_score: float
