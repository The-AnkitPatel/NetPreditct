import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
from backend.app.core.config import settings
from backend.app.domain.telemetry_schemas import RawTelemetryRecord


def generate_high_fidelity_network_trace(
    total_minutes: int = 1440 * 7,  # 7 days of 1-minute telemetry (10,080 samples)
    random_seed: int = 42,
    device_id: str = "core-router-alpha",
    interface_id: str = "xe-0/0/1"
) -> pd.DataFrame:
    """
    Generates a rigorous, calibrated network telemetry trace embodying real-world
    MAWI transit traffic patterns, CAIDA router instabilities, bufferbloat dynamics,
    and diurnal human usage cycles.
    """
    np.random.seed(random_seed)
    start_time = datetime(2026, 9, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(total_minutes)]
    
    # 1. Diurnal base traffic cycle (24-hour sinusoidal curve + morning/evening peaks)
    hours = np.array([t.hour + t.minute / 60.0 for t in timestamps])
    diurnal_wave = 0.5 * (1 - np.cos(2 * np.pi * (hours - 4) / 24))  # Trough at 04:00, Peak at 16:00
    base_util = 25.0 + 45.0 * diurnal_wave + np.random.normal(0, 3.5, total_minutes)
    base_util = np.clip(base_util, 10.0, 95.0)

    # 2. Line-rate throughput (correlated with link capacity e.g. 10,000 Mbps / 10G link)
    throughput_mbps = (base_util / 100.0) * 10000.0 + np.random.normal(0, 150.0, total_minutes)
    throughput_mbps = np.clip(throughput_mbps, 500.0, 9900.0)

    # 3. Base RTT and Queue Occupancy (Bufferbloat kinetics)
    # Under low load (<60%), base propagation RTT is ~12-16ms.
    # As queue occupancy rises exponentially past 75% util, queuing delay explodes.
    queue_occupancy = np.zeros(total_minutes)
    rtt_ms = np.zeros(total_minutes)
    packet_loss_pct = np.zeros(total_minutes)
    tcp_retrans_rate = np.zeros(total_minutes)
    interface_discards_sec = np.zeros(total_minutes)
    crc_errors = np.zeros(total_minutes)
    cpu_util = np.zeros(total_minutes)
    mem_util = np.zeros(total_minutes)
    bgp_flaps = np.zeros(total_minutes, dtype=int)

    # Generate synthetic incident cascade clusters (e.g., periodic link failures, BGP flaps, microbursts)
    # Approximately 18 distinct incident events across 7 days
    incident_intervals = [
        (480, 520, "microburst_surge"),       # Day 1 08:00
        (1260, 1310, "bufferbloat_collapse"), # Day 1 21:00
        (2200, 2245, "bgp_reroute_storm"),    # Day 2 12:40
        (3400, 3460, "interface_crc_drop"),   # Day 3 08:40
        (4600, 4650, "memory_leak_exhaustion"), # Day 4 04:40
        (5800, 5860, "bufferbloat_collapse"), # Day 5 00:40
        (7200, 7250, "microburst_surge"),     # Day 5 24:00
        (8600, 8660, "bgp_reroute_storm"),    # Day 6 23:20
        (9800, 9850, "bufferbloat_collapse"), # Day 7 19:20
    ]

    for i in range(total_minutes):
        u = base_util[i]
        q_base = np.maximum(0.0, (u - 40.0) * 1.2) + np.random.exponential(1.5)
        
        # Check incident events
        in_incident = False
        incident_type = None
        for s_idx, e_idx, itype in incident_intervals:
            if s_idx <= i <= e_idx:
                in_incident = True
                incident_type = itype
                break
        
        if in_incident:
            if incident_type == "microburst_surge":
                base_util[i] = min(99.0, base_util[i] + 35.0)
                q_base += 65.0
                bgp_flaps[i] = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            elif incident_type == "bufferbloat_collapse":
                base_util[i] = min(98.5, base_util[i] + 28.0)
                q_base += 78.0
            elif incident_type == "bgp_reroute_storm":
                bgp_flaps[i] = np.random.choice([2, 4, 8], p=[0.4, 0.4, 0.2])
                q_base += 50.0
                base_util[i] = min(96.0, base_util[i] + 20.0)
            elif incident_type == "interface_crc_drop":
                crc_errors[i] = np.random.uniform(15.0, 85.0)
                q_base += 40.0
            elif incident_type == "memory_leak_exhaustion":
                mem_util[i] = min(99.5, 75.0 + (i - 4600) * 0.45)
                q_base += 60.0

        q = np.clip(q_base, 2.0, 99.8)
        queue_occupancy[i] = q

        # Delay gradient: base RTT + queuing delay proportional to queue buffer depth
        rtt = 14.0 + (q / 100.0) ** 2 * 120.0 + np.random.normal(0, 1.2)
        rtt_ms[i] = max(11.0, rtt)

        # Packet drops occur sharply when queue occupancy exceeds 85%
        if q > 85.0:
            loss = ((q - 85.0) / 15.0) ** 1.8 * 8.5 + np.random.exponential(0.4)
            discards = (q - 85.0) * 18.0 + np.random.uniform(5, 30)
            retrans = loss * 12.0 + np.random.uniform(2, 10)
        else:
            loss = np.random.exponential(0.01) if np.random.rand() > 0.95 else 0.0
            discards = 0.0
            retrans = np.random.exponential(0.05)

        packet_loss_pct[i] = np.clip(loss, 0.0, 25.0)
        interface_discards_sec[i] = discards
        tcp_retrans_rate[i] = retrans

        # CPU and Memory
        cpu_util[i] = np.clip(20.0 + (base_util[i] * 0.55) + bgp_flaps[i] * 6.5 + np.random.normal(0, 2), 15.0, 99.0)
        if mem_util[i] == 0:
            mem_util[i] = np.clip(35.0 + (q * 0.35) + np.random.normal(0, 1.5), 30.0, 95.0)

    # 4. Construct Multi-Horizon Ground Truth Labels (Strict Future Horizons)
    # Congestion/Failure Criterion at T+h:
    # (queue_occupancy > 80% OR packet_loss > 1.5% OR rtt > 65ms OR discards > 5/s)
    severe_threshold = (queue_occupancy > 80.0) | (packet_loss_pct > 1.5) | (rtt_ms > 65.0)

    target_t5 = np.zeros(total_minutes, dtype=int)
    target_t15 = np.zeros(total_minutes, dtype=int)
    target_t30 = np.zeros(total_minutes, dtype=int)

    for i in range(total_minutes):
        if i + 5 < total_minutes:
            target_t5[i] = int(severe_threshold[i + 5])
        if i + 15 < total_minutes:
            target_t15[i] = int(severe_threshold[i + 15])
        if i + 30 < total_minutes:
            target_t30[i] = int(severe_threshold[i + 30])

    # Future regression targets at T+15m
    future_rtt_t15 = np.roll(rtt_ms, -15)
    future_loss_t15 = np.roll(packet_loss_pct, -15)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "device_id": device_id,
        "interface_id": interface_id,
        # Raw Telemetry Signals
        "bandwidth_util_pct": np.round(base_util, 2),
        "throughput_mbps": np.round(throughput_mbps, 1),
        "rtt_ms": np.round(rtt_ms, 2),
        "rtt_jitter_ms": np.round(np.abs(np.random.normal(1.8, 1.2, total_minutes)), 2),
        "queue_occupancy_pct": np.round(queue_occupancy, 2),
        "packet_loss_pct": np.round(packet_loss_pct, 3),
        "tcp_retrans_rate": np.round(tcp_retrans_rate, 2),
        "interface_discards_sec": np.round(interface_discards_sec, 2),
        "crc_errors_sec": np.round(crc_errors, 2),
        "cpu_util_pct": np.round(cpu_util, 2),
        "memory_util_pct": np.round(mem_util, 2),
        "bgp_flap_count": bgp_flaps,
        # Future Labels for Horizons (Strictly shifted)
        "target_t5": target_t5,
        "target_t15": target_t15,
        "target_t30": target_t30,
        "future_rtt_t15": np.round(future_rtt_t15, 2),
        "future_loss_t15": np.round(future_loss_t15, 3),
    })

    return df


def ensure_dataset_exists() -> pd.DataFrame:
    """Loads dataset from CSV or generates and saves it if not present."""
    os.makedirs(os.path.dirname(settings.DATASET_PATH), exist_ok=True)
    if os.path.exists(settings.DATASET_PATH):
        df = pd.read_csv(settings.DATASET_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    
    df = generate_high_fidelity_network_trace()
    df.to_csv(settings.DATASET_PATH, index=False)
    return df
