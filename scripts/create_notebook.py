import json
import os

os.makedirs("notebooks", exist_ok=True)

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

def add_md(source):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# Title and Overview
add_md("""# NetPredict: Network Congestion & Telemetry ML Pipeline
This notebook implements the foundational data preprocessing, data cleaning, and feature engineering steps for the NetPredict network congestion prediction system.

### Pipeline Outline:
1. **Cell 1: Environment Setup & Library Imports**
2. **Cell 2: Telemetry Dataset Ingestion & Initial Inspection**
3. **Cell 3: Data Cleaning & Domain Boundary Preprocessing**
4. **Cell 4: Physics-Grounded Temporal Feature Engineering (Zero Data Leakage)**""")

# CELL 1
add_md("""---
## Cell 1: Importing Required Libraries and Configuring Environment
We import core scientific computing libraries (`pandas`, `numpy`), visualization tools (`matplotlib`, `seaborn`), and machine learning preprocessing utilities (`scikit-learn`).""")

add_code("""# ==============================================================================
# CELL 1: Importing Required Libraries and Configuring Environment
# ==============================================================================
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn utilities for preprocessing and time-series cross-validation
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit

# Configure display options and suppress minor warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

print("=" * 60)
print("✓ Libraries successfully imported!")
print(f"  • Pandas Version: {pd.__version__}")
print(f"  • NumPy Version:  {np.__version__}")
print("=" * 60)""")

# CELL 2
add_md("""---
## Cell 2: Loading the Network Telemetry Dataset
We load the multi-variate telemetry dataset (`telemetry_trace.csv`).
The dataset captures 7 continuous days of 1-minute sampled router telemetry (10,080 rows), recording metrics such as bandwidth utilization, queue occupancy (bufferbloat), round-trip time (RTT), packet loss, and CPU load.""")

add_code("""# ==============================================================================
# CELL 2: Loading the Telemetry Dataset (Works Everywhere: Colab, Kaggle, Local)
# ==============================================================================
# Public raw GitHub URL for the dataset (works instantly in Google Colab & Kaggle)
GITHUB_RAW_URL = "https://raw.githubusercontent.com/The-AnkitPatel/NetPreditct/main/backend/data/telemetry_trace.csv"

# Potential local paths if running on your machine
local_paths = [
    os.path.join("..", "backend", "data", "telemetry_trace.csv"),
    os.path.join("backend", "data", "telemetry_trace.csv"),
    "telemetry_trace.csv"
]

# Check for local file; if not found (e.g. in Google Colab), stream from public GitHub
dataset_source = None
for p in local_paths:
    if os.path.exists(p):
        dataset_source = p
        break

if dataset_source is None:
    dataset_source = GITHUB_RAW_URL
    print("Loading dataset directly from public GitHub repository (Google Colab Mode)...")
else:
    print(f"Loading dataset from local filesystem: {dataset_source}")

# Load CSV into a pandas DataFrame
df_raw = pd.read_csv(dataset_source)

print("=" * 60)
print("✓ Dataset successfully loaded!")
print(f"  • Data Source:                        {dataset_source}")
print(f"  • Total Telemetry Records (Rows):     {df_raw.shape[0]:,}")
print(f"  • Total Raw Telemetry Signals (Cols): {df_raw.shape[1]}")
print("=" * 60)

# Display first 5 samples
print("\\nFirst 5 Records:")
display(df_raw.head())""")

# CELL 3
add_md("""---
## Cell 3: Data Cleaning & Preprocessing
Network sensor telemetry in production frequently encounters missing samples, out-of-order packets, duplicates, or corrupted counter spikes.
This cell performs a 4-stage data cleaning protocol:
1. **Timestamp Conversion & Chronological Sorting**: Guarantees strict temporal order.
2. **Duplicate Detection & Removal**: Eliminates any redundant time intervals.
3. **Missing Value Audit & Imputation**: Verifies completeness and applies forward-fill for continuous signals.
4. **Physical Boundary Sanity Checks**: Clamps percentages to `[0.0, 100.0]%` and ensures non-negative latency/throughput.""")

add_code("""# ==============================================================================
# CELL 3: Data Preprocessing and Cleaning Pipeline
# ==============================================================================
df_clean = df_raw.copy()

print("--- 1. Timestamp Parsing & Chronological Sorting ---")
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
df_clean = df_clean.sort_values(by='timestamp').reset_index(drop=True)
print(f"  • Telemetry Start Time: {df_clean['timestamp'].min()}")
print(f"  • Telemetry End Time:   {df_clean['timestamp'].max()}")

print("\\n--- 2. Duplicate Detection ---")
duplicate_count = df_clean.duplicated(subset=['timestamp']).sum()
if duplicate_count > 0:
    print(f"  • Found {duplicate_count} duplicate timestamp records. Dropping duplicates...")
    df_clean = df_clean.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
else:
    print("  • ✓ Zero duplicate timestamps found.")

print("\\n--- 3. Missing Value Audit & Imputation ---")
null_counts = df_clean.isnull().sum()
total_nulls = null_counts.sum()
print(f"  • Total missing / NaN values: {total_nulls}")
if total_nulls > 0:
    print("  • Applying forward-fill (ffill) followed by backward-fill (bfill)...")
    df_clean = df_clean.ffill().bfill()
else:
    print("  • ✓ Clean dataset: No missing values.")

print("\\n--- 4. Physical Boundary Sanity Checks & Range Clamping ---")
# Percentages cannot physically exceed 100% or drop below 0%
percentage_fields = [
    'bandwidth_util_pct', 'queue_occupancy_pct', 
    'packet_loss_pct', 'cpu_util_pct', 'memory_util_pct'
]
for col in percentage_fields:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].clip(lower=0.0, upper=100.0)

# Hardware latencies, discards, and throughput must be strictly non-negative
df_clean['rtt_ms'] = df_clean['rtt_ms'].clip(lower=1.0)
df_clean['throughput_mbps'] = df_clean['throughput_mbps'].clip(lower=0.0)
df_clean['tcp_retrans_rate'] = df_clean['tcp_retrans_rate'].clip(lower=0.0)
df_clean['interface_discards_sec'] = df_clean['interface_discards_sec'].clip(lower=0.0)
df_clean['crc_errors_sec'] = df_clean['crc_errors_sec'].clip(lower=0.0)

print("  • ✓ All telemetry signals verified within valid physical network ranges.")
print(f"  • Cleaned Dataset Dimensions: {df_clean.shape}")

# Display statistical verification
display(df_clean[percentage_fields + ['rtt_ms', 'throughput_mbps']].describe().T[['min', 'mean', 'max']])""")

# CELL 4
add_md("""---
## Cell 4: Feature Engineering (Causal Temporal Dynamics)
Raw instantaneous signals alone cannot predict future congestion because **bufferbloat is a dynamic build-up process**.
We engineer four categories of features with **strict zero future leakage** (only observations up to time $t$ are used):
1. **Rolling Window Aggregations (5m & 15m)**: Rolling averages, standard deviations, and maximums to measure sustained load and microbursts.
2. **Delay Gradients & Trend Slopes ($\\\\Delta RTT / \\\\Delta t$)**: The rate of latency increase is the physical leading indicator of impending buffer overflow.
3. **Exponentially Weighted Moving Averages (EWMA)**: Filters high-frequency noise while prioritizing recent signals.
4. **Domain Interaction Features**:
   - `buffer_stress_index`: Joint pressure of queue occupancy and link bandwidth utilization.
   - `congestion_pressure_score`: Composite early warning index combining delay slope, queue depth, and drop rate.""")

add_code("""# ==============================================================================
# CELL 4: Physics-Grounded Temporal Feature Engineering Pipeline
# ==============================================================================
df_features = df_clean.copy()

print("--- 1. Rolling Window Aggregations (Past 5m & 15m) ---")
# Rolling 5-minute window features (captures short-term microbursts)
df_features['rtt_mean_5m'] = df_features['rtt_ms'].rolling(window=5, min_periods=1).mean()
df_features['rtt_std_5m'] = df_features['rtt_ms'].rolling(window=5, min_periods=1).std().fillna(0.0)
df_features['queue_mean_5m'] = df_features['queue_occupancy_pct'].rolling(window=5, min_periods=1).mean()
df_features['queue_max_5m'] = df_features['queue_occupancy_pct'].rolling(window=5, min_periods=1).max()
df_features['loss_max_5m'] = df_features['packet_loss_pct'].rolling(window=5, min_periods=1).max()

# Rolling 15-minute window features (captures sustained medium-term queue pressure)
df_features['util_mean_15m'] = df_features['bandwidth_util_pct'].rolling(window=15, min_periods=1).mean()
df_features['rtt_mean_15m'] = df_features['rtt_ms'].rolling(window=15, min_periods=1).mean()
df_features['queue_mean_15m'] = df_features['queue_occupancy_pct'].rolling(window=15, min_periods=1).mean()
df_features['queue_std_15m'] = df_features['queue_occupancy_pct'].rolling(window=15, min_periods=1).std().fillna(0.0)
df_features['discards_sum_15m'] = df_features['interface_discards_sec'].rolling(window=15, min_periods=1).sum()

print("--- 2. Rate-of-Change & Delay Gradients (Slope = ΔValue / Δt) ---")
# Delay gradient: rate at which RTT is climbing over past 5 intervals
df_features['rtt_slope_5m'] = df_features['rtt_ms'].diff(periods=5).fillna(0.0) / 5.0
df_features['queue_slope_5m'] = df_features['queue_occupancy_pct'].diff(periods=5).fillna(0.0) / 5.0
df_features['util_slope_5m'] = df_features['bandwidth_util_pct'].diff(periods=5).fillna(0.0) / 5.0

print("--- 3. Exponentially Weighted Moving Averages (EWMA) ---")
# Smooth out telemetry jitter while reacting quickly to acute trends
df_features['rtt_ewma'] = df_features['rtt_ms'].ewm(alpha=0.2, adjust=False).mean()
df_features['queue_ewma'] = df_features['queue_occupancy_pct'].ewm(alpha=0.2, adjust=False).mean()

print("--- 4. Domain Interaction Features ---")
# Buffer Stress Index: Joint multiplicative interaction of queue depth and link utilization
df_features['buffer_stress_index'] = (
    (df_features['queue_occupancy_pct'] * df_features['bandwidth_util_pct']) / 100.0
)

# Congestion Pressure Score: Combines delay gradient with queue occupancy and discards
df_features['congestion_pressure_score'] = np.clip(
    (df_features['queue_occupancy_pct'] * 0.4)
    + (np.maximum(0.0, df_features['rtt_slope_5m']) * 5.0)
    + (df_features['packet_loss_pct'] * 10.0)
    + (df_features['interface_discards_sec'] * 0.8),
    0.0,
    150.0
)

print("=" * 60)
print("✓ Feature Engineering Complete (Zero Temporal Data Leakage)")
print(f"  • Total Original Columns:   {df_clean.shape[1]}")
print(f"  • Total Engineered Columns: {df_features.shape[1]}")
print(f"  • New Features Added:       {df_features.shape[1] - df_clean.shape[1]}")
print("=" * 60)

# Display sample of newly engineered feature values
sample_cols = [
    'timestamp', 'rtt_ms', 'rtt_slope_5m', 
    'queue_occupancy_pct', 'queue_slope_5m', 
    'buffer_stress_index', 'congestion_pressure_score'
]
display(df_features[sample_cols].head())""")

# Save notebook
target_notebook = os.path.join("notebooks", "NetPredict_ML_Pipeline.ipynb")
with open(target_notebook, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Notebook generated successfully at: {target_notebook}")
