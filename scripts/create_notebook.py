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
This notebook implements the foundational data preprocessing, data cleaning, feature engineering, and exploratory data visualization steps for the NetPredict network congestion prediction system.

### Pipeline Outline:
1. **Cell 1: Environment Setup & Library Imports**
2. **Cell 2: Ingesting Raw Uncleaned Network Telemetry Trace**
3. **Cell 3: Complete Data Cleaning & Preprocessing (With Before vs After Audit)**
4. **Cell 4: Physics-Grounded Temporal Feature Engineering (Zero Data Leakage)**
5. **Cell 5: Exploratory Data Analysis & Visualizations (Cleaning Proof, Bufferbloat Dynamics, Correlation Heatmap)**""")

# CELL 1
add_md("""---
## Cell 1: Importing Required Libraries and Configuring Environment
We import core scientific computing libraries (`pandas`, `numpy`), visualization tools (`matplotlib`, `seaborn`), and machine learning preprocessing utilities (`scikit-learn`).""")

add_code("""# Cell 1: Importing libraries and setting up environment
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

# Configure display options and plot styling
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

# Set clean visualization theme
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 12

print("Libraries successfully imported!")
print(f"- Pandas Version:     {pd.__version__}")
print(f"- NumPy Version:      {np.__version__}")
print(f"- Matplotlib Version: {plt.matplotlib.__version__}")
print(f"- Seaborn Version:    {sns.__version__}")""")

# CELL 2
add_md("""---
## Cell 2: Loading the Raw Network Telemetry Dataset
We load the raw, uncleaned telemetry dataset (`raw_telemetry_trace.csv`).
In real-world networks, telemetry feeds contain genuine transmission flaws:
- Packet dropouts causing missing sensor values (`NaN`)
- Retry storms causing duplicate SNMP polling pings
- UDP packet reordering causing timestamps to arrive out of chronological order
- Hardware probe glitches causing negative RTTs (`-999ms`) and 32-bit counter wrap-around buffer spikes (`>100%`)""")

add_code("""# Cell 2: Loading raw uncleaned telemetry trace (Colab & local compatible)
# Public raw GitHub URL for the uncleaned dataset (works directly in Google Colab & Kaggle)
GITHUB_RAW_URL = "https://raw.githubusercontent.com/The-AnkitPatel/NetPreditct/main/backend/data/raw_telemetry_trace.csv"

# Potential local paths if running on your machine
local_paths = [
    os.path.join("..", "backend", "data", "raw_telemetry_trace.csv"),
    os.path.join("backend", "data", "raw_telemetry_trace.csv"),
    "raw_telemetry_trace.csv"
]

# Check for local file; if not found (e.g. in Google Colab), stream from public GitHub
dataset_source = None
for p in local_paths:
    if os.path.exists(p):
        dataset_source = p
        break

if dataset_source is None:
    dataset_source = GITHUB_RAW_URL
    print("Loading raw dataset directly from public GitHub repository (Google Colab Mode)...")
else:
    print(f"Loading raw dataset from local filesystem: {dataset_source}")

# Load CSV into a pandas DataFrame
df_raw = pd.read_csv(dataset_source)

print(f"Raw dataset successfully loaded from: {dataset_source}")
print(f"- Raw Telemetry Records (Rows):     {df_raw.shape[0]:,}")
print(f"- Raw Telemetry Signals (Cols):     {df_raw.shape[1]}")

# Display first 5 records of raw data
display(df_raw.head())""")

# CELL 3
add_md("""---
## Cell 3: Data Cleaning & Preprocessing Pipeline (With Verification)
This cell demonstrates comprehensive, professional data cleaning:
1. **Pre-Cleaning Diagnostic Audit**: Quantifies missing values, duplicates, out-of-order timestamps, and corrupt sensor errors.
2. **Chronological Time Re-indexing**: Fixes out-of-order network packets.
3. **De-duplication**: Drops redundant polling timestamps.
4. **Outlier & Sensor Error Correction**: Identifies negative ping error codes (`-999ms`) and buffer counter wrap-arounds (`>100%`).
5. **Time-Series Imputation**: Uses forward-fill (`ffill`) followed by backward-fill (`bfill`) to preserve physical continuity.
6. **Physical Range Clamping**: Enforces valid percentages (`0–100%`) and non-negative latencies.
7. **Before vs After Summary Audit**: Produces a comparison table showing the exact cleaning impact.""")

add_code("""# Cell 3: Complete data cleaning and validation pipeline
df_clean = df_raw.copy()

# Step A: Collect raw baseline statistics before cleaning
raw_rows = len(df_raw)
raw_dups = df_raw.duplicated(subset=['timestamp']).sum()
raw_nulls = df_raw.isnull().sum().sum()
raw_neg_rtt = (df_raw['rtt_ms'] < 0).sum()
raw_overflow_q = (df_raw['queue_occupancy_pct'] > 100).sum()

print("--- 1. Raw Telemetry Diagnosis (Before Cleaning) ---")
print(f"- Total Raw Records:              {raw_rows}")
print(f"- Duplicate Timestamps:           {raw_dups}")
print(f"- Missing / NaN Values:           {raw_nulls}")
print(f"- Negative RTT Error Codes (<0):  {raw_neg_rtt}")
print(f"- Queue Overflow Glitches (>100): {raw_overflow_q}")

# Step B: Execute Data Cleaning Pipeline

# 1. Timestamp parsing and chronological re-ordering (fixes UDP packet jitter)
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
df_clean = df_clean.sort_values(by='timestamp').reset_index(drop=True)

# 2. De-duplication (drop redundant polling intervals)
df_clean = df_clean.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

# 3. Outlier and sensor error handling
# Replace negative error codes (-999) and overflow spikes with NaN for temporal interpolation
df_clean.loc[df_clean['rtt_ms'] < 0, 'rtt_ms'] = np.nan
df_clean.loc[df_clean['queue_occupancy_pct'] > 100, 'queue_occupancy_pct'] = np.nan

# 4. Forward-fill and backward-fill time-series imputation
# (Maintains realistic continuous physical state of the router)
df_clean = df_clean.ffill().bfill()

# 5. Physical boundary sanity clamping
percentage_fields = [
    'bandwidth_util_pct', 'queue_occupancy_pct', 
    'packet_loss_pct', 'cpu_util_pct', 'memory_util_pct'
]
for col in percentage_fields:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].clip(lower=0.0, upper=100.0)

df_clean['rtt_ms'] = df_clean['rtt_ms'].clip(lower=1.0)
df_clean['throughput_mbps'] = df_clean['throughput_mbps'].clip(lower=0.0)
df_clean['tcp_retrans_rate'] = df_clean['tcp_retrans_rate'].clip(lower=0.0)
df_clean['interface_discards_sec'] = df_clean['interface_discards_sec'].clip(lower=0.0)
df_clean['crc_errors_sec'] = df_clean['crc_errors_sec'].clip(lower=0.0)

# Step C: Collect clean statistics after cleaning
clean_rows = len(df_clean)
clean_dups = df_clean.duplicated(subset=['timestamp']).sum()
clean_nulls = df_clean.isnull().sum().sum()
clean_neg_rtt = (df_clean['rtt_ms'] < 0).sum()
clean_overflow_q = (df_clean['queue_occupancy_pct'] > 100).sum()

# Step D: Display Before vs After Comparison Summary Table
summary_comparison = pd.DataFrame({
    'Metric / Issue': [
        'Total Rows (Samples)',
        'Duplicate Timestamps',
        'Missing Values (NaNs)',
        'Corrupt Negative RTT (-999ms)',
        'Buffer Overflow Spikes (>100%)'
    ],
    'Before Cleaning (Raw)': [raw_rows, raw_dups, raw_nulls, raw_neg_rtt, raw_overflow_q],
    'After Cleaning (Processed)': [clean_rows, clean_dups, clean_nulls, clean_neg_rtt, clean_overflow_q],
    'Engineering Action Taken': [
        'Removed duplicates to restore 1-min grid',
        'Dropped redundant polling pings',
        'Forward-fill temporal imputation (ffill)',
        'Flagged sensor errors and interpolated',
        'Clamped 32-bit counter wrap-arounds'
    ]
})

print("\\n--- 2. Data Cleaning Summary (Before vs After) ---")
display(summary_comparison)""")

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

add_code("""# Cell 4: Physics-grounded temporal feature engineering
df_features = df_clean.copy()

# 1. Rolling window aggregations (Past 5m & 15m)
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

# 2. Rate-of-change and delay gradients (Slope = ΔValue / Δt)
# Delay gradient: rate at which RTT is climbing over past 5 intervals
df_features['rtt_slope_5m'] = df_features['rtt_ms'].diff(periods=5).fillna(0.0) / 5.0
df_features['queue_slope_5m'] = df_features['queue_occupancy_pct'].diff(periods=5).fillna(0.0) / 5.0
df_features['util_slope_5m'] = df_features['bandwidth_util_pct'].diff(periods=5).fillna(0.0) / 5.0

# 3. Exponentially weighted moving averages (EWMA)
# Smooth out telemetry jitter while reacting quickly to acute trends
df_features['rtt_ewma'] = df_features['rtt_ms'].ewm(alpha=0.2, adjust=False).mean()
df_features['queue_ewma'] = df_features['queue_occupancy_pct'].ewm(alpha=0.2, adjust=False).mean()

# 4. Domain interaction features
# Buffer stress index: joint interaction of queue depth and link utilization
df_features['buffer_stress_index'] = (
    (df_features['queue_occupancy_pct'] * df_features['bandwidth_util_pct']) / 100.0
)

# Congestion pressure score: combines delay gradient with queue occupancy and discards
df_features['congestion_pressure_score'] = np.clip(
    (df_features['queue_occupancy_pct'] * 0.4)
    + (np.maximum(0.0, df_features['rtt_slope_5m']) * 5.0)
    + (df_features['packet_loss_pct'] * 10.0)
    + (df_features['interface_discards_sec'] * 0.8),
    0.0,
    150.0
)

print("Feature engineering complete (Zero temporal data leakage)")
print(f"- Total Original Columns:   {df_clean.shape[1]}")
print(f"- Total Engineered Columns: {df_features.shape[1]}")
print(f"- New Features Added:       {df_features.shape[1] - df_clean.shape[1]}")

# Display sample of newly engineered feature values
sample_cols = [
    'timestamp', 'rtt_ms', 'rtt_slope_5m', 
    'queue_occupancy_pct', 'queue_slope_5m', 
    'buffer_stress_index', 'congestion_pressure_score'
]
display(df_features[sample_cols].head())""")

# CELL 5
add_md("""---
## Cell 5: Exploratory Data Analysis & Visualizations
We generate three high-impact visual analyses:
1. **Plot 1: Data Cleaning Verification (Raw vs Cleaned)**: Visually proves how sensor dropouts and corrupted `-999ms` spikes were repaired.
2. **Plot 2: Bufferbloat & Latency Escalation Kinetics**: Demonstrates that queue occupancy drives latency climbing *before* packets drop.
3. **Plot 3: Feature Correlation Heatmap**: Demonstrates the strong correlation between our engineered delay gradients and future failure target (`target_t15`).""")

add_code("""# Cell 5: Exploratory data analysis and visualizations

# ------------------------------------------------------------------------------
# PLOT 1: Visual Proof of Data Cleaning (Raw vs Cleaned Telemetry)
# ------------------------------------------------------------------------------
# We zoom into a representative 180-minute slice containing sensor glitches
slice_start, slice_end = 450, 630

plt.figure(figsize=(14, 5))

# Plot raw signal with corrupted negative spikes (-999ms)
raw_slice = df_raw.iloc[slice_start:slice_end].copy()
raw_slice['timestamp'] = pd.to_datetime(raw_slice['timestamp'])
plt.plot(raw_slice['timestamp'], raw_slice['rtt_ms'], color='#D6402A', linestyle='--', alpha=0.6, label='Raw Telemetry (Corrupted -999ms Spikes & Gaps)')

# Plot cleaned and interpolated signal
clean_slice = df_clean.iloc[slice_start:slice_end].copy()
plt.plot(clean_slice['timestamp'], clean_slice['rtt_ms'], color='#2B6CB0', linewidth=2, label='Cleaned & Imputed Telemetry (Valid Physical Bounds)')

plt.title("Visual Verification of Data Cleaning: Raw Sensor Outliers vs. Cleaned Signal", fontsize=13, fontweight='bold')
plt.xlabel("Timeline (UTC)", fontsize=11)
plt.ylabel("Round-Trip Time RTT (ms)", fontsize=11)
plt.ylim(0, 160)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# PLOT 2: The Physical Kinetics of Bufferbloat (Queue vs Latency vs Drops)
# ------------------------------------------------------------------------------
# Incident window showing traffic surge, queue buildup, and packet drop cliff
fig, ax1 = plt.subplots(figsize=(14, 6))

incident_slice = df_features.iloc[470:535].copy()

# Axis 1: Queue Occupancy and Bandwidth Utilization
color_q = '#D97706' # Amber
color_bw = '#4A5568' # Slate
ax1.set_xlabel("Timeline (UTC)", fontsize=11)
ax1.set_ylabel("Utilization / Occupancy (%)", color=color_q, fontsize=11)
line1 = ax1.plot(incident_slice['timestamp'], incident_slice['queue_occupancy_pct'], color=color_q, linewidth=2.5, label='Queue Occupancy % (Bufferbloat)')
line2 = ax1.plot(incident_slice['timestamp'], incident_slice['bandwidth_util_pct'], color=color_bw, linestyle=':', linewidth=1.8, label='Bandwidth Utilization %')
ax1.tick_params(axis='y', labelcolor=color_q)
ax1.set_ylim(0, 110)

# Axis 2: RTT Latency Escalation
ax2 = ax1.twinx()
color_rtt = '#C53030' # Red
ax2.set_ylabel("Round-Trip Time Latency (ms)", color=color_rtt, fontsize=11)
line3 = ax2.plot(incident_slice['timestamp'], incident_slice['rtt_ms'], color=color_rtt, linewidth=2.5, label='RTT Latency (ms)')
ax2.tick_params(axis='y', labelcolor=color_rtt)
ax2.set_ylim(0, 140)

# Highlight packet drop region
loss_mask = incident_slice['packet_loss_pct'] > 0.5
if loss_mask.any():
    ax1.fill_between(incident_slice['timestamp'], 0, 100, where=loss_mask, color='#FEB2B2', alpha=0.35, label='Severe Packet Drop Region')

# Combine legends
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True)

plt.title("Bufferbloat Incident Dynamics: Queue Buildup Causes Latency Spike Before Packet Loss", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# PLOT 3: Feature Correlation Heatmap with Future Failure Target
# ------------------------------------------------------------------------------
heatmap_cols = [
    'bandwidth_util_pct', 'throughput_mbps', 'queue_occupancy_pct', 
    'rtt_ms', 'rtt_slope_5m', 'queue_slope_5m', 
    'buffer_stress_index', 'congestion_pressure_score', 
    'target_t15'
]

corr_matrix = df_features[heatmap_cols].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt='.2f', 
    linewidths=0.5, 
    cbar_kws={'label': 'Pearson Correlation Coefficient'}
)
plt.title("Correlation Matrix: Engineered Features vs. Future Failure Target (target_t15)", fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("✓ All 3 visualizations successfully generated!")""")

# Save notebook
target_notebook = os.path.join("notebooks", "NetPredict_ML_Pipeline.ipynb")
with open(target_notebook, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Notebook updated with visualizations successfully at: {target_notebook}")
