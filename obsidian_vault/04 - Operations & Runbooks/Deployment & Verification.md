---
tags: [operations, deployment, testing]
created: 2026-09-09
---
# Deployment & Verification

How to deploy the [[System Architecture]] locally for testing.

## Running the Backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Running the Frontend
```bash
cd frontend
npm install
npm run dev
```

> [!tip] Benchmarks
> Run `pytest tests/benchmarks/` to verify the [[Data Plane & Ingestion Pipeline]] is maintaining > 10,000 events/sec throughput.
