---
tags: [ml, conformal, uncertainty]
created: 2026-09-09
---
# Split Conformal Prediction

Deterministic predictions ("RTT will be 150ms") are useless during network storms. We need prediction intervals.

## Distribution-Free Guarantees
Split Conformal Prediction gives us a rigorous way to say: "We are 90% confident the RTT will be between 120ms and 180ms."

> [!info] Methodology
> 1. Train model on $D_{train}$.
> 2. Compute non-conformity scores (e.g., absolute error) on a calibration set $D_{cal}$.
> 3. Find the 90th percentile of these scores, $\hat{q}$.
> 4. For a new point $X_{test}$, the interval is $[\hat{y} - \hat{q}, \hat{y} + \hat{q}]$.

This pairs perfectly with our [[Multi-Horizon LightGBM]] to provide bounded uncertainty for [[Counterfactual What-If Simulation]].
