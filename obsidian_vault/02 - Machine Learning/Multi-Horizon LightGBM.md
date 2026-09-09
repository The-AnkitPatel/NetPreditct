---
tags: [ml, lightgbm, forecasting]
created: 2026-09-09
---
# Multi-Horizon LightGBM

We forecast failures across three distinct horizons: T+5m, T+15m, and T+30m.

## Why LightGBM?
Unlike LSTMs, Gradient Boosted Decision Trees (GBDTs) handle tabular network features (categorical router IDs, missing SNMP data) natively and train orders of magnitude faster.

> [!tip] Objective Function
> We use binary logloss for failure prediction.

## Hyperparameters
- `learning_rate`: 0.05
- `num_leaves`: 31
- `feature_fraction`: 0.8 (reduces overfitting on correlated metrics)

Requires careful [[Temporal Leakage Prevention]].
