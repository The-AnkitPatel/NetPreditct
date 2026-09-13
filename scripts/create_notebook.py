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
add_md("""# NetPredict: Network Telemetry & ML Preprocessing Pipeline
This notebook implements the data loading, data cleaning, and feature engineering pipeline for NetPredict.""")

# CELL 1
add_md("""## Cell 1: Environment Setup and Library Imports
Importing data manipulation, visualization, and preprocessing libraries.""")

add_code("""# Cell 1: Environment Setup and Library Imports
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

print("Libraries successfully imported")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")""")

# CELL 2
add_md("""## Cell 2: Loading the Network Telemetry Dataset
Loads the telemetry trace directly from GitHub or local storage.""")

add_code("""# Cell 2: Loading the Telemetry Dataset
GITHUB_RAW_URL = "https://raw.githubusercontent.com/The-AnkitPatel/NetPreditct/main/backend/data/telemetry_trace.csv"

local_paths = [
    os.path.join("..", "backend", "data", "telemetry_trace.csv"),
    os.path.join("backend", "data", "telemetry_trace.csv"),
    "telemetry_trace.csv"
]

dataset_source = None
for p in local_paths:
    if os.path.exists(p):
        dataset_source = p
        break

if dataset_source is None:
    dataset_source = GITHUB_RAW_URL
    print("Loading dataset from public GitHub repository")
else:
    print(f"Loading dataset from local path: {dataset_source}")

df_raw = pd.read_csv(dataset_source)

print(f"Dataset successfully loaded from: {dataset_source}")
print(f"Total rows: {df_raw.shape[0]}")
print(f"Total columns: {df_raw.shape[1]}")

display(df_raw.head())""")

# CELL 3
add_md("""## Cell 3: Data Cleaning and Preprocessing
Handles timestamp parsing, duplicate removal, missing value audits, and physical range boundary clamping.""")

add_code("""# Cell 3: Data Cleaning and Preprocessing
df_clean = df_raw.copy()

# 1. Timestamp parsing and chronological sorting
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
df_clean = df_clean.sort_values(by='timestamp').reset_index(drop=True)
print(f"Start time: {df_clean['timestamp'].min()}")
print(f"End time: {df_clean['timestamp'].max()}")

# 2. Duplicate detection
duplicate_count = df_clean.duplicated(subset=['timestamp']).sum()
if duplicate_count > 0:
    df_clean = df_clean.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
print(f"Duplicate timestamps found: {duplicate_count}")

# 3. Missing value audit and imputation
null_counts = df_clean.isnull().sum().sum()
if null_counts > 0:
    df_clean = df_clean.ffill().bfill()
print(f"Total missing values: {null_counts}")

# 4. Physical boundary sanity checks
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

print("Physical boundaries verified")
print(f"Cleaned dataset shape: {df_clean.shape}")

display(df_clean[percentage_fields + ['rtt_ms', 'throughput_mbps']].describe().T[['min', 'mean', 'max']])""")

# CELL 4
add_md("""## Cell 4: Temporal Feature Engineering
Constructs rolling window metrics, delay slopes, exponential moving averages, and buffer stress indicators without data leakage.""")

add_code("""# Cell 4: Temporal Feature Engineering
df_features = df_clean.copy()

# 1. Rolling window statistics
df_features['rtt_mean_5m'] = df_features['rtt_ms'].rolling(window=5, min_periods=1).mean()
df_features['rtt_std_5m'] = df_features['rtt_ms'].rolling(window=5, min_periods=1).std().fillna(0.0)
df_features['queue_mean_5m'] = df_features['queue_occupancy_pct'].rolling(window=5, min_periods=1).mean()
df_features['queue_max_5m'] = df_features['queue_occupancy_pct'].rolling(window=5, min_periods=1).max()
df_features['loss_max_5m'] = df_features['packet_loss_pct'].rolling(window=5, min_periods=1).max()

df_features['util_mean_15m'] = df_features['bandwidth_util_pct'].rolling(window=15, min_periods=1).mean()
df_features['rtt_mean_15m'] = df_features['rtt_ms'].rolling(window=15, min_periods=1).mean()
df_features['queue_mean_15m'] = df_features['queue_occupancy_pct'].rolling(window=15, min_periods=1).mean()
df_features['queue_std_15m'] = df_features['queue_occupancy_pct'].rolling(window=15, min_periods=1).std().fillna(0.0)
df_features['discards_sum_15m'] = df_features['interface_discards_sec'].rolling(window=15, min_periods=1).sum()

# 2. Rate of change and delay gradients
df_features['rtt_slope_5m'] = df_features['rtt_ms'].diff(periods=5).fillna(0.0) / 5.0
df_features['queue_slope_5m'] = df_features['queue_occupancy_pct'].diff(periods=5).fillna(0.0) / 5.0
df_features['util_slope_5m'] = df_features['bandwidth_util_pct'].diff(periods=5).fillna(0.0) / 5.0

# 3. Exponentially weighted moving averages
df_features['rtt_ewma'] = df_features['rtt_ms'].ewm(alpha=0.2, adjust=False).mean()
df_features['queue_ewma'] = df_features['queue_occupancy_pct'].ewm(alpha=0.2, adjust=False).mean()

# 4. Domain interaction features
df_features['buffer_stress_index'] = (
    (df_features['queue_occupancy_pct'] * df_features['bandwidth_util_pct']) / 100.0
)

df_features['congestion_pressure_score'] = np.clip(
    (df_features['queue_occupancy_pct'] * 0.4)
    + (np.maximum(0.0, df_features['rtt_slope_5m']) * 5.0)
    + (df_features['packet_loss_pct'] * 10.0)
    + (df_features['interface_discards_sec'] * 0.8),
    0.0,
    150.0
)

print("Feature engineering complete")
print(f"Original columns: {df_clean.shape[1]}")
print(f"Total columns after feature engineering: {df_features.shape[1]}")
print(f"New features created: {df_features.shape[1] - df_clean.shape[1]}")

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
