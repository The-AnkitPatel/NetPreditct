---
tags: [network, simulation, counterfactual]
created: 2026-09-09
---
# Counterfactual What-If Simulation

Once [[TreeSHAP Explainability]] identifies the root cause, operators need to know how to fix it. 

## Operator Decision Support
The what-if engine allows operators to simulate changes:
- *What if I route 20% of traffic via the backup link?*
- *What if I expand the QoS buffer size by 50MB?*

> [!tip] Simulation Math
> We adjust the feature vector $X$ to $X'$ based on the operator's proposed action, and re-run the [[Multi-Horizon LightGBM]] inference to see if the predicted failure probability drops below the threshold.
