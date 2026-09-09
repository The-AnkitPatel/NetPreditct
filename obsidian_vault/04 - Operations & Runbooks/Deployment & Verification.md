---
tags: [operations, deployment, testing]
created: 2026-09-09
---
# Deployment & Verification

How to deploy and verify the [[System Architecture]] across local environments and live serverless edge infrastructure.

## 1. Zero-Cost Live Serverless Edge Deployment (Vercel)

NetPredict is architected for **zero operational cost ($0/month)**. It deploys across Vercel's global Edge CDN without requiring dedicated virtual machines or keeping a workstation powered on.

* **Live Interactive Cockpit**: [https://netpredict-cockpit.vercel.app](https://netpredict-cockpit.vercel.app)
  * Framework: React 19 + TypeScript + Vite
  * Edge Architecture: In-browser physical queue kinetics and simulation engine (`offlineEngine.ts`) with zero cold start.
  * Mobile Experience: Responsive slide-in navigation drawer, 44px+ touch targets, and iOS safe area padding.
* **Architectural Documentation Portal**: [https://netpredict-docs.vercel.app/docs](https://netpredict-docs.vercel.app/docs)
  * Framework: Next.js 15 + Fumadocs MDX

## 2. Local Environment Execution

### Backend Predictive Engine (FastAPI + LightGBM)
```bash
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Interactive Cockpit (React 19 + Vite)
```bash
cd frontend
npm install
npm run dev # Launches at http://localhost:5173
```

### Documentation Handbook (Fumadocs)
```bash
cd docs-portal
npm install
npm run dev # Launches at http://localhost:3000/docs
```

## 3. Automated Verification & Test Suite

Run the full end-to-end test suite to verify zero temporal leakage, conformal prediction coverage, and counterfactual simulation:
```bash
# Run 11/11 pytest unit & integration tests
.venv\Scripts\pytest -v

# Verify frontend production build
cd frontend && npm run build

# Verify documentation production build
cd docs-portal && npm run build
```

> [!tip] Dual Telemetry Data Plane
> NetPredict is **not a mock or toy gimmick**. Operators can toggle between live host adapter telemetry via socket/eBPF probes (`POLL HOST NIC`) and 1,440-minute real-world calibrated MAWI/CAIDA queue traces. Run `pytest backend/tests/test_api.py -k live_poll` to verify live socket telemetry extraction.
