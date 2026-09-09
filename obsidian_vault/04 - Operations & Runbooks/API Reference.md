---
tags: [operations, api, docs]
created: 2026-09-09
---
# API Reference

NetPredict exposes a REST API via FastAPI.

## Endpoints

### `POST /api/v1/telemetry/ingest`
Ingests batch telemetry data.
- **Payload**:
  ```json
  {
    "device_id": "core-router-01",
    "timestamp": "2026-09-09T10:00:00Z",
    "metrics": {"rtt": 45.2, "queue_depth": 1024}
  }
  ```

### `GET /api/v1/predict/{device_id}`
Retrieves multi-horizon predictions.
- **Response**: Returns T+5m, T+15m probabilities and [[Split Conformal Prediction]] intervals.

See [[Deployment & Verification]] for how to spin up the API locally.
