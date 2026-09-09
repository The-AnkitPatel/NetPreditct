import type {
  RawTelemetryRecord,
  PredictionResponse,
  ShapExplanationResponse,
  SimulationScenarioRequest,
  SimulationResponse,
  HistoricalIncident,
  ModelHealthData,
} from '../types/telemetry';
import { browserEngine } from './offlineEngine';

const BASE_URL = '/api/v1';
let backendConnected = false;

export const isBackendOnline = () => backendConnected;

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    throw new Error(`API Error [${resp.status}]: ${await resp.text()}`);
  }
  backendConnected = true;
  return resp.json();
}

export const api = {
  // Telemetry Endpoints
  async getLatestTelemetry(): Promise<{
    telemetry: RawTelemetryRecord;
    anomaly: { is_anomaly: boolean; anomaly_score: number; status: string };
  }> {
    try {
      return await fetchJson(`${BASE_URL}/telemetry/latest`);
    } catch {
      const pred = browserEngine.getPrediction();
      return {
        telemetry: browserEngine.getHistory(1)[0],
        anomaly: {
          is_anomaly: pred.is_current_anomaly,
          anomaly_score: pred.current_anomaly_score,
          status: pred.anomaly_status,
        },
      };
    }
  },

  async getTelemetryHistory(limit: number = 60): Promise<RawTelemetryRecord[]> {
    try {
      return await fetchJson(`${BASE_URL}/telemetry/history?limit=${limit}`);
    } catch {
      return browserEngine.getHistory(limit);
    }
  },

  async stepStream(): Promise<{
    record: RawTelemetryRecord;
    prediction: PredictionResponse;
    progress: any;
  }> {
    try {
      return await fetchJson(`${BASE_URL}/telemetry/stream/step`, { method: 'POST' });
    } catch {
      return browserEngine.stepStream();
    }
  },

  async jumpToIncident(): Promise<{
    record: RawTelemetryRecord;
    prediction: PredictionResponse;
    progress: any;
  }> {
    try {
      return await fetchJson(`${BASE_URL}/telemetry/stream/jump-incident`, { method: 'POST' });
    } catch {
      return browserEngine.jumpToIncident();
    }
  },

  async resetStream(): Promise<{
    record: RawTelemetryRecord;
    prediction: PredictionResponse;
    progress: any;
  }> {
    try {
      return await fetchJson(`${BASE_URL}/telemetry/stream/reset`, { method: 'POST' });
    } catch {
      return browserEngine.resetStream();
    }
  },

  async pollLiveInterface(): Promise<{
    source: string;
    record: RawTelemetryRecord;
    prediction: PredictionResponse;
  }> {
    try {
      return await fetchJson(`${BASE_URL}/telemetry/live-poll`, { method: 'POST' });
    } catch {
      return {
        source: 'browser_hardware_simulation',
        ...browserEngine.stepStream(),
      };
    }
  },

  // Prediction & Explainability
  async getMultiHorizonPredictions(): Promise<PredictionResponse> {
    try {
      return await fetchJson(`${BASE_URL}/predict/horizons`);
    } catch {
      return browserEngine.getPrediction();
    }
  },

  async getHistoricalIncidents(): Promise<HistoricalIncident[]> {
    try {
      return await fetchJson(`${BASE_URL}/predict/incidents`);
    } catch {
      return browserEngine.getIncidents();
    }
  },

  async getLatestExplanation(): Promise<ShapExplanationResponse> {
    try {
      return await fetchJson(`${BASE_URL}/explain/latest`);
    } catch {
      return browserEngine.getExplanation();
    }
  },

  async getExplanationById(predictionId: string): Promise<ShapExplanationResponse> {
    try {
      return await fetchJson(`${BASE_URL}/explain/${predictionId}`);
    } catch {
      return browserEngine.getExplanation();
    }
  },

  // What-If Simulation
  async simulateScenario(req: SimulationScenarioRequest): Promise<SimulationResponse> {
    try {
      return await fetchJson(`${BASE_URL}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      });
    } catch {
      return browserEngine.simulate(req);
    }
  },

  async prescribeOptimalScenario(): Promise<{
    status: string;
    recommended_scenario: SimulationScenarioRequest;
    simulation: SimulationResponse;
    rationale: string;
  }> {
    try {
      return await fetchJson(`${BASE_URL}/simulate/prescribe`, { method: 'POST' });
    } catch {
      return browserEngine.prescribe();
    }
  },

  // Model Health & Drift Monitoring
  async getModelHealth(): Promise<ModelHealthData> {
    try {
      return await fetchJson(`${BASE_URL}/health/model`);
    } catch {
      return browserEngine.getModelHealth();
    }
  },
};

