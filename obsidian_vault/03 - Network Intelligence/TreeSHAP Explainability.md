---
tags: [network, explainability, shap]
created: 2026-09-09
---
# TreeSHAP Explainability

Network operators won't trust a black box. We use TreeSHAP (SHapley Additive exPlanations) to explain every alert.

## Additive Shapley Values
SHAP decomposes the model's prediction into the sum of contributions from each feature.

$$ f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i $$

> [!info] Human-Readable Root Causes
> Instead of "Model output 0.9", we present: "90% failure probability driven by: 1. `eth0_queue_depth` (+0.4) 2. `rtt_derivative` (+0.2)."

This allows operators to take immediate, targeted action in the [[Counterfactual What-If Simulation]] engine.
