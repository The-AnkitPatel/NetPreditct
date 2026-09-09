import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backend.app.services.feature_extractor import (
    extract_features_from_dataframe,
    FEATURE_COLUMNS,
)


def test_feature_extractor_no_future_leakage():
    # Construct synthetic 20-sample dataframe
    start = datetime(2026, 9, 1, 10, 0)
    records = []
    for i in range(20):
        records.append({
            "timestamp": start + timedelta(minutes=i),
            "bandwidth_util_pct": 30.0 + i * 2,
            "throughput_mbps": 3000.0 + i * 200,
            "rtt_ms": 15.0 + (i ** 1.3),
            "rtt_jitter_ms": 1.5,
            "queue_occupancy_pct": 20.0 + i * 3,
            "packet_loss_pct": 0.05 if i > 15 else 0.0,
            "tcp_retrans_rate": 0.2,
            "interface_discards_sec": 1.0 if i > 15 else 0.0,
            "crc_errors_sec": 0.0,
            "cpu_util_pct": 25.0 + i,
            "memory_util_pct": 35.0,
            "bgp_flap_count": 0,
        })
    df = pd.DataFrame(records)

    feat_df = extract_features_from_dataframe(df)

    # Check all feature columns exist
    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns

    # Verify that modifying row 19 does NOT affect feature row 10 (no future leakage)
    df_altered = df.copy()
    df_altered.loc[19, "queue_occupancy_pct"] = 999.0
    feat_df_altered = extract_features_from_dataframe(df_altered)

    # Row 10 must be identical
    pd.testing.assert_series_equal(feat_df.iloc[10], feat_df_altered.iloc[10])
