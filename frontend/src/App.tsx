import React, { useState, useEffect, useRef } from 'react';
import { api } from './services/api';
import type {
  RawTelemetryRecord,
  PredictionResponse,
  ShapExplanationResponse,
  HistoricalIncident,
} from './types/telemetry';

import { CustomCursor } from './components/CustomCursor';
import { Topbar } from './components/Topbar';
import { Sidebar, type SectionId } from './components/Sidebar';
import { TelemetryGrid } from './components/TelemetryGrid';

import { PredictiveHorizonPanel } from './components/PredictiveHorizonPanel';
import { AnomalyVsPredictionCard } from './components/AnomalyVsPredictionCard';
import { ShapWaterfallChart } from './components/ShapWaterfallChart';
import { WhatIfScenarioLab } from './components/WhatIfScenarioLab';
import { ModelHealthPanel } from './components/ModelHealthPanel';
import { IncidentHistoryTable } from './components/IncidentHistoryTable';

export const App: React.FC = () => {
  const [currentTelemetry, setCurrentTelemetry] = useState<RawTelemetryRecord | null>(null);
  const [telemetryHistory, setTelemetryHistory] = useState<RawTelemetryRecord[]>([]);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] = useState<ShapExplanationResponse | null>(null);
  const [incidents, setIncidents] = useState<HistoricalIncident[]>([]);
  const [activeSection, setActiveSection] = useState<SectionId>('telemetry');
  const [isPlaying, setIsPlaying] = useState<boolean>(true);

  const timerRef = useRef<number | null>(null);

  // Fetch initial telemetry snapshot & predictions
  const fetchAllData = async () => {
    try {
      const hist = await api.getTelemetryHistory(35);
      setTelemetryHistory(hist);
      if (hist.length > 0) {
        setCurrentTelemetry(hist[hist.length - 1]);
      }

      const pred = await api.getMultiHorizonPredictions();
      setPrediction(pred);

      const expl = await api.getLatestExplanation();
      setExplanation(expl);

      const incs = await api.getHistoricalIncidents();
      setIncidents(incs);
    } catch (err) {
      console.error('Failed to hydrate dashboard:', err);
    }
  };

  useEffect(() => {
    fetchAllData();
  }, []);

  // Streaming step handler
  const handleStep = async () => {
    try {
      const res = await api.stepStream();
      setCurrentTelemetry(res.record);
      setTelemetryHistory((prev) => [...prev.slice(-34), res.record]);
      setPrediction(res.prediction);

      const expl = await api.getLatestExplanation();
      setExplanation(expl);
    } catch (err) {
      console.error('Stream step error:', err);
    }
  };

  const handleJumpIncident = async () => {
    try {
      const res = await api.jumpToIncident();
      setCurrentTelemetry(res.record);
      setTelemetryHistory((prev) => [...prev.slice(-34), res.record]);
      setPrediction(res.prediction);

      const expl = await api.getLatestExplanation();
      setExplanation(expl);
    } catch (err) {
      console.error('Incident jump error:', err);
    }
  };

  const handleResetStream = async () => {
    try {
      const res = await api.resetStream();
      setCurrentTelemetry(res.record);
      setTelemetryHistory((prev) => [...prev.slice(-34), res.record]);
      setPrediction(res.prediction);

      const expl = await api.getLatestExplanation();
      setExplanation(expl);
    } catch (err) {
      console.error('Stream reset error:', err);
    }
  };

  const handlePollLiveNic = async () => {
    try {
      const res = await api.pollLiveInterface();
      setCurrentTelemetry(res.record);
      setTelemetryHistory((prev) => [...prev.slice(-34), res.record]);
      setPrediction(res.prediction);

      const expl = await api.getLatestExplanation();
      setExplanation(expl);
    } catch (err) {
      console.error('Live NIC polling error:', err);
    }
  };

  // Auto-play interval loop
  useEffect(() => {
    if (isPlaying) {
      timerRef.current = window.setInterval(() => {
        handleStep();
      }, 3500);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying]);

  // Scroll spy to highlight active section in sidebar
  useEffect(() => {
    const handleScroll = () => {
      const sectionIds: SectionId[] = [
        'telemetry',
        'horizons',
        'anomaly_vs_pred',
        'shap',
        'whatif',
        'health',
        'incidents',
      ];
      const scrollPos = window.scrollY + 140;
      for (const id of sectionIds) {
        const el = document.getElementById(id);
        if (el) {
          const top = el.offsetTop;
          const height = el.offsetHeight;
          if (scrollPos >= top && scrollPos < top + height) {
            setActiveSection(id);
            break;
          }
        }
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="layout-root">
      <CustomCursor />
      <Sidebar
        activeSection={activeSection}
        onSelectSection={setActiveSection}
        isAnomaly={prediction?.is_current_anomaly || false}
      />

      <div className="layout-main">
        <Topbar
          isPlaying={isPlaying}
          onTogglePlay={() => setIsPlaying(!isPlaying)}
          onStep={handleStep}
          onJumpIncident={handleJumpIncident}
          onReset={handleResetStream}
          onPollLiveNic={handlePollLiveNic}
          currentTimestamp={currentTelemetry?.timestamp ? new Date(currentTelemetry.timestamp).toUTCString() : ''}
          isAnomaly={prediction?.is_current_anomaly || false}
        />

        <main className="app-shell">
          <TelemetryGrid current={currentTelemetry} history={telemetryHistory} />
          <PredictiveHorizonPanel prediction={prediction} />
          <AnomalyVsPredictionCard prediction={prediction} />
          <ShapWaterfallChart explanation={explanation} />
          <WhatIfScenarioLab />
          <ModelHealthPanel />
          <IncidentHistoryTable incidents={incidents} />

          <footer style={{ marginTop: '40px', borderTop: '2px solid var(--ink)', paddingTop: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--grey)' }}>
            <div>
              <span style={{ fontWeight: 800, color: 'var(--ink)' }}>NETPREDICT PLATFORM</span> — Autonomous Network Telemetry & Degradation Forecasting
            </div>
            <div className="font-mono">
              Calibrated LightGBM v4.7.0 • TreeSHAP Exact Engine • Scikit-Learn 1.9.0
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
};


export default App;
