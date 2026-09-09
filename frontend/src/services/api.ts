import type {
  RawTelemetryRecord,
  PredictionResponse,
  ShapExplanationResponse,
  SimulationScenarioRequest,
  SimulationResponse,
  HistoricalIncident,
  ModelHealthData,
} from '../types/telemetry';

const BASE_URL = '/api/v1';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    throw new Error(`API Error [${resp.status}]: ${await resp.text()}`);
  }
  return resp.json();
}

export const api = {
  // Telemetry Endpoints
  async getLatestTelemetry(): Promise<{
    telemetry: RawTelemetryRecord;
    anomaly: { is_anomaly: boolean; anomaly_score: number; status: string };
  }> {
    return fetchJson(`${BASE_URL}/telemetry/latest`);
  },

  async getTelemetryHistory(limit: number = 60): Promise<RawTelemetryRecord[]> {
    return fetchJson(`${BASE_URL}/telemetry/history?limit=${limit}`);
  },

  async stepStream(): Promise<{
    record: RawTelemetryRecord;
    prediction: PredictionResponse;
    progress: any;
  }> {
    return fetchJson(`${BASE_URL}/telemetry/stream/step`, { method: 'POST' });
  },

  async jumpToIncident(): Promise<{
    record: RawTelemetryRecord;
    prediction: PredictionResponse;
    progress: any;
  }> {
    return fetchJson(`${BASE_URL}/telemetry/stream/jump-incident`, { method: 'POST' });
  },

  async resetStream(): Promise<{
    record: RawTelemetryRecord;
    prediction: PredictionResponse;
    progress: any;
  }> {
    return fetchJson(`${BASE_URL}/telemetry/stream/reset`, { method: 'POST' });
  },

  async pollLiveInterface(): Promise<{
    source: string;
    record: RawTelemetryRecord;
    prediction: PredictionResponse;
  }> {
    return fetchJson(`${BASE_URL}/telemetry/live-poll`, { method: 'POST' });
  },

  // Prediction & Explainability
  async getMultiHorizonPredictions(): Promise<PredictionResponse> {
    return fetchJson(`${BASE_URL}/predict/horizons`);
  },

  async getHistoricalIncidents(): Promise<HistoricalIncident[]> {
    return fetchJson(`${BASE_URL}/predict/incidents`);
  },

  async getLatestExplanation(): Promise<ShapExplanationResponse> {
    return fetchJson(`${BASE_URL}/explain/latest`);
  },

  async getExplanationById(predictionId: string): Promise<ShapExplanationResponse> {
    return fetchJson(`${BASE_URL}/explain/${predictionId}`);
  },

  // What-If Simulation
  async simulateScenario(req: SimulationScenarioRequest): Promise<SimulationResponse> {
    return fetchJson(`${BASE_URL}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  },

  // Model Health & Drift Monitoring
  async getModelHealth(): Promise<ModelHealthData> {
    return fetchJson(`${BASE_URL}/health/model`);
  },
};
