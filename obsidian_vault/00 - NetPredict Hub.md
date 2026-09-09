---
aliases: [Hub, Index, Home]
tags: [index, netpredict, overview]
created: 2026-09-09
---
# 🌐 NetPredict Hub

Welcome to the NetPredict Obsidian Vault! This is the Map of Content (MOC) for our AI-driven network telemetry platform.

> [!info] Project Mission
> To predict network failures (bufferbloat, link saturation, packet loss) *before* they occur using multi-horizon LightGBM on a sliding window state machine.

## 🏗️ [[01 - Architecture]]
- [[System Architecture]]: High-level design (FastAPI, React 19, ML Service).
- [[Data Plane & Ingestion Pipeline]]: High-throughput telemetry ingestion.
- [[Sliding Window State Machine]]: O(1) temporal slicing buffer.

## 🧠 [[02 - Machine Learning]]
- [[Multi-Horizon LightGBM]]: T+5m, T+15m, and T+30m forecasting.
- [[Temporal Leakage Prevention]]: Strict walk-forward validation and embargoing.
- [[Probability Calibration & Brier Score]]: Making probabilities trustworthy.
- [[Split Conformal Prediction]]: 90% coverage intervals for RTT/loss.

## 🔬 [[03 - Network Intelligence]]
- [[Decoupling Anomaly Detection vs Future Prediction]]: Isolation Forests vs GBDTs.
- [[Bufferbloat & Delay Gradient Kinetics]]: Physics of queuing delay.
- [[TreeSHAP Explainability]]: Human-readable operator root causes.
- [[Counterfactual What-If Simulation]]: Testing QoS diversion scenarios.

## 🚀 [[04 - Operations & Runbooks]]
- [[Operator Guide - Using NetPredict]]: Step-by-step user manual.
- [[API Reference]]: Endpoint schemas.
- [[Incident Response Guide]]: Triage protocols.
- [[Deployment & Verification]]: Running the stack.

## 📓 [[05 - Student Engineering Diary]]
- [[Dev Diary - Building NetPredict]]: Notes from the trenches.
