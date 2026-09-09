---
tags: [ml, calibration, brier-score]
created: 2026-09-09
---
# Probability Calibration & Brier Score

Tree-based models like LightGBM often push predicted probabilities away from 0 and 1, leading to uncalibrated predictions.

## The Problem with Raw Probabilities
If the model outputs 0.8, it should mean there is an 80% chance of failure. Often, raw tree outputs are just ranking scores.

> [!tip] Isotonic Regression
> We apply Isotonic Regression post-training to calibrate the outputs. 

## Reliability Curves
We use reliability diagrams and the Brier Score (Mean Squared Error of probabilities) to measure calibration quality. 

$$ Brier = \frac{1}{N} \sum_{t=1}^{N} (f_t - o_t)^2 $$
Where $f_t$ is the forecast probability and $o_t$ is the actual outcome.
