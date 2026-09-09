---
tags: [network, anomaly, isolation-forest]
created: 2026-09-09
---
# Decoupling Anomaly Detection vs Future Prediction

Operators often confuse detecting anomalies with predicting failures. NetPredict explicitly separates them.

## Isolation Forest: "What is abnormal now?"
We use Isolation Forests to detect immediate point-anomalies (e.g., sudden microbursts). It works by isolating observations in a random tree structure.

## GBDT: "What will fail next?"
Our [[Multi-Horizon LightGBM]] is forward-looking. A state might not be anomalous *now*, but it might consistently lead to bufferbloat in 15 minutes.

> [!tip] Synergistic Pipeline
> The anomaly score from the Isolation Forest is actually fed as a feature into the LightGBM model!
