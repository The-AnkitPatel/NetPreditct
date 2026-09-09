---
aliases: [Architecture, High Level Design, HLD]
tags: [architecture, fastapi, react]
created: 2026-09-09
---
# System Architecture

NetPredict follows a strict hexagonal/clean architecture, separating the core domain (network telemetry & prediction) from infrastructure (databases, APIs).

> [!tip] Tech Stack
> - **Frontend**: React 19 (concurrent rendering for real-time dashboards)
> - **Backend**: FastAPI (async I/O for high-throughput metrics)
> - **ML Layer**: Python / LightGBM

## Architecture Diagram
```mermaid
graph TD
    A[Telemetry Sources INT/SNMP] -->|UDP/Kafka| B[Ingestion Pipeline]
    B --> C[Sliding Window State Machine]
    C --> D[ML Inference Service]
    D -->|Predictions| E[FastAPI Backend]
    E -->|WebSockets| F[React 19 Dashboard]
    E -->|REST| G[Counterfactual Engine]
```

## Core Components
1. **[[Data Plane & Ingestion Pipeline]]**: Captures incoming streams.
2. **[[Sliding Window State Machine]]**: Buffers state efficiently.
3. **[[Multi-Horizon LightGBM]]**: Runs inference based on state.
