---
tags: [operations, runbook, incident]
created: 2026-09-09
---
# Incident Response Guide

Standard operating procedures for network anomalies detected by NetPredict.

> [!caution] Escalation Matrix
> If confidence > 95% and T+5m horizon is triggered, automatically enact QoS rate-limiting before paging on-call.

## Triage Protocols

1. **Bufferbloat Detected**
   - Check [[Bufferbloat & Delay Gradient Kinetics]] metrics on the dashboard.
   - Action: Implement AQM (CoDel or RED) on the affected interface.
2. **Microbursts**
   - Identified by [[Decoupling Anomaly Detection vs Future Prediction]] (Isolation Forest).
   - Action: Investigate downstream storage/compute jobs causing bursts.
3. **BGP Flap Cascades**
   - Action: Apply route dampening; verify neighbor states.
