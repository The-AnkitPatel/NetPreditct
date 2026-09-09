import numpy as np
import pandas as pd
from typing import List, Dict, Any


FEATURE_COLUMNS: List[str] = [
    # Instantaneous signals
    "bandwidth_util_pct",
    "throughput_mbps",
    "rtt_ms",
    "rtt_jitter_ms",
    "queue_occupancy_pct",
    "packet_loss_pct",
    "tcp_retrans_rate",
    "interface_discards_sec",
    "crc_errors_sec",
    "cpu_util_pct",
    "memory_util_pct",
    "bgp_flap_count",
    # Rolling Statistics (5-min window)
    "rtt_mean_5m",
    "rtt_std_5m",
    "queue_mean_5m",
    "queue_max_5m",
    "loss_max_5m",
    # Rolling Statistics (15-min window)
    "util_mean_15m",
    "rtt_mean_15m",
    "queue_mean_15m",
    "queue_std_15m",
    "discards_sum_15m",
    # Trend Gradients (Rate of change over 5 mins)
    "rtt_slope_5m",
    "queue_slope_5m",
    "util_slope_5m",
    # EWMA Smoothing
    "rtt_ewma",
    "queue_ewma",
    # Physical Domain Interaction Features
    "buffer_stress_index",
    "congestion_pressure_score",
]


def extract_features_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strictly backward-looking temporal and statistical features.
    Guarantees ZERO future leakage: only observations up to current step are used.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    feat_df = df.copy()

    # 1. Rolling 5-minute aggregations (min_periods=1 ensures no missing values at start)
    feat_df["rtt_mean_5m"] = feat_df["rtt_ms"].rolling(window=5, min_periods=1).mean()
    feat_df["rtt_std_5m"] = feat_df["rtt_ms"].rolling(window=5, min_periods=1).std().fillna(0.0)
    feat_df["queue_mean_5m"] = feat_df["queue_occupancy_pct"].rolling(window=5, min_periods=1).mean()
    feat_df["queue_max_5m"] = feat_df["queue_occupancy_pct"].rolling(window=5, min_periods=1).max()
    feat_df["loss_max_5m"] = feat_df["packet_loss_pct"].rolling(window=5, min_periods=1).max()

    # 2. Rolling 15-minute aggregations
    feat_df["util_mean_15m"] = feat_df["bandwidth_util_pct"].rolling(window=15, min_periods=1).mean()
    feat_df["rtt_mean_15m"] = feat_df["rtt_ms"].rolling(window=15, min_periods=1).mean()
    feat_df["queue_mean_15m"] = feat_df["queue_occupancy_pct"].rolling(window=15, min_periods=1).mean()
    feat_df["queue_std_15m"] = feat_df["queue_occupancy_pct"].rolling(window=15, min_periods=1).std().fillna(0.0)
    feat_df["discards_sum_15m"] = feat_df["interface_discards_sec"].rolling(window=15, min_periods=1).sum()

    # 3. Trend Gradients (Differences over 5 time steps)
    rtt_diff = feat_df["rtt_ms"].diff(periods=5).fillna(0.0)
    feat_df["rtt_slope_5m"] = rtt_diff / 5.0

    queue_diff = feat_df["queue_occupancy_pct"].diff(periods=5).fillna(0.0)
    feat_df["queue_slope_5m"] = queue_diff / 5.0

    util_diff = feat_df["bandwidth_util_pct"].diff(periods=5).fillna(0.0)
    feat_df["util_slope_5m"] = util_diff / 5.0

    # 4. Exponentially Weighted Moving Averages (alpha=0.2)
    feat_df["rtt_ewma"] = feat_df["rtt_ms"].ewm(alpha=0.2, adjust=False).mean()
    feat_df["queue_ewma"] = feat_df["queue_occupancy_pct"].ewm(alpha=0.2, adjust=False).mean()

    # 5. Domain Interaction Terms
    # Buffer stress index: joint pressure of queue depth and link load
    feat_df["buffer_stress_index"] = (
        (feat_df["queue_occupancy_pct"] * feat_df["bandwidth_util_pct"]) / 100.0
    )

    # Congestion pressure score: combines delay gradient with queue occupancy and loss
    feat_df["congestion_pressure_score"] = np.clip(
        (feat_df["queue_occupancy_pct"] * 0.4)
        + (np.maximum(0.0, feat_df["rtt_slope_5m"]) * 5.0)
        + (feat_df["packet_loss_pct"] * 10.0)
        + (feat_df["interface_discards_sec"] * 0.8),
        0.0,
        150.0,
    )

    return feat_df[FEATURE_COLUMNS]


def extract_single_step_features(history_df: pd.DataFrame) -> pd.Series:
    """Computes features for the latest step using buffered sliding window history."""
    if len(history_df) == 0:
        return pd.Series(0.0, index=FEATURE_COLUMNS)
    all_features = extract_features_from_dataframe(history_df)
    return all_features.iloc[-1]
