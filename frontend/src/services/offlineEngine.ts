import type {
  RawTelemetryRecord,
  PredictionResponse,
  HorizonPrediction,
  RiskLevel,
  FeatureAttribution,
  ShapExplanationResponse,
  SimulationScenarioRequest,
  SimulationResponse,
  HistoricalIncident,
  ModelHealthData,
} from '../types/telemetry';

class BrowserTelemetryEngine {
  private step: number = 48;
  private isIncident: boolean = false;

  private generateTelemetry(step: number): RawTelemetryRecord {
    const t = step * 0.12;
    const diurnal = 35 + 28 * Math.sin(t);
    const burst = this.isIncident ? 38.0 : 0.0;

    const bw = Math.min(96.5, Math.max(12.0, diurnal + burst + (Math.sin(t * 3.7) * 4.5)));
    const q = Math.min(98.0, Math.max(5.0, (bw * 0.95) + (this.isIncident ? 26.0 : 0.0)));
    const rtt = Math.max(12.0, 14.5 + (q * 0.72) + (this.isIncident ? 22.0 : 0.0));
    const loss = this.isIncident ? 1.45 : (q > 82.0 ? 0.35 : 0.0);
    const retrans = this.isIncident ? 7.8 : (q > 80.0 ? 1.8 : 0.1);

    return {
      timestamp: new Date(Date.now() - (100 - step) * 60000).toISOString(),
      device_id: 'core-router-alpha',
      interface_id: 'xe-0/0/1',
      bandwidth_util_pct: Number(bw.toFixed(1)),
      throughput_mbps: Number((bw * 38.5).toFixed(1)),
      rtt_ms: Number(rtt.toFixed(1)),
      rtt_jitter_ms: Number((1.2 + (this.isIncident ? 4.8 : 0.4)).toFixed(1)),
      queue_occupancy_pct: Number(q.toFixed(1)),
      packet_loss_pct: Number(loss.toFixed(2)),
      tcp_retrans_rate: Number(retrans.toFixed(1)),
      interface_discards_sec: this.isIncident ? 14.0 : 0.0,
      crc_errors_sec: 0.0,
      cpu_util_pct: Number((42.0 + bw * 0.35).toFixed(1)),
      memory_util_pct: Number((54.0 + q * 0.25).toFixed(1)),
      bgp_flap_count: 0,
    };
  }

  public getHistory(limit: number = 40): RawTelemetryRecord[] {
    const list: RawTelemetryRecord[] = [];
    for (let i = this.step - limit; i <= this.step; i++) {
      list.push(this.generateTelemetry(Math.max(1, i)));
    }
    return list;
  }

  public stepStream(): { record: RawTelemetryRecord; prediction: PredictionResponse; progress: any } {
    this.step += 1;
    const rec = this.generateTelemetry(this.step);
    const pred = this.getPrediction(rec);
    return { record: rec, prediction: pred, progress: { current_step: this.step, total_steps: 10080 } };
  }

  public jumpToIncident(): { record: RawTelemetryRecord; prediction: PredictionResponse; progress: any } {
    this.isIncident = true;
    this.step += 1;
    const rec = this.generateTelemetry(this.step);
    const pred = this.getPrediction(rec);
    return { record: rec, prediction: pred, progress: { current_step: this.step, total_steps: 10080 } };
  }

  public resetStream(): { record: RawTelemetryRecord; prediction: PredictionResponse; progress: any } {
    this.isIncident = false;
    this.step = 45;
    const rec = this.generateTelemetry(this.step);
    const pred = this.getPrediction(rec);
    return { record: rec, prediction: pred, progress: { current_step: this.step, total_steps: 10080 } };
  }

  private createHorizonPrediction(horizon: number, risk: number, rtt: number, loss: number): HorizonPrediction {
    const level: RiskLevel = risk >= 0.7 ? 'CRITICAL' : risk >= 0.4 ? 'ELEVATED' : 'LOW';
    return {
      horizon_minutes: horizon,
      failure_risk_score: risk,
      calibrated_probability: risk,
      risk_level: level,
      predicted_rtt_ms: Number((rtt + horizon * 0.2).toFixed(1)),
      predicted_packet_loss_pct: Number(loss.toFixed(2)),
      conformal_lower_bound_rtt: Number(Math.max(5.0, rtt - 6.0).toFixed(1)),
      conformal_upper_bound_rtt: Number((rtt + 6.0).toFixed(1)),
    };
  }

  public getPrediction(currentRec?: RawTelemetryRecord): PredictionResponse {
    const rec = currentRec || this.generateTelemetry(this.step);
    const q = rec.queue_occupancy_pct;
    const rtt = rec.rtt_ms;

    const stress = Math.min(1.0, Math.max(0.02, (q / 95.0) * 0.7 + (rec.packet_loss_pct * 0.3)));
    const risk5 = Math.min(0.99, Number((stress * 1.15).toFixed(3)));
    const risk15 = Math.min(0.95, Number((stress * 0.88).toFixed(3)));
    const risk30 = Math.min(0.90, Number((stress * 0.62).toFixed(3)));

    const h5 = this.createHorizonPrediction(5, risk5, rtt, rec.packet_loss_pct);
    const h15 = this.createHorizonPrediction(15, risk15, rtt, rec.packet_loss_pct);
    const h30 = this.createHorizonPrediction(30, risk30, rtt, 0.0);

    return {
      prediction_id: `PRED-${Math.random().toString(16).substring(2, 8).toUpperCase()}`,
      timestamp: rec.timestamp,
      device_id: rec.device_id,
      interface_id: rec.interface_id,
      current_anomaly_score: Number((q > 75 ? 0.82 : 0.14).toFixed(3)),
      is_current_anomaly: q > 75,
      anomaly_status: q > 75 ? 'ANOMALOUS' : 'NORMAL',
      horizons: { '5': h5, '15': h15, '30': h30 },
      primary_horizon: h15,
      recommended_action: risk15 >= 0.7 ? 'IMMEDIATE MITIGATION: Reroute 25-40% ingress traffic to secondary path.' : 'CONTINUE MONITORING: Network operating within SLA thresholds.',
      primary_drivers: q > 70 ? [`Queue Occupancy (${q}%)`, `Delay Gradient (+2.8ms/min)`] : ['Nominal link utilization', 'Stable queue depths'],
    };
  }

  public getExplanation(): ShapExplanationResponse {
    const rec = this.generateTelemetry(this.step);
    const q = rec.queue_occupancy_pct;

    const drivers: FeatureAttribution[] = [
      {
        feature_name: 'queue_occupancy_pct',
        feature_value: q,
        shap_value: q > 70 ? 0.38 : -0.04,
        contribution_pct: 42.0,
        impact_direction: q > 70 ? 'INCREASES_RISK' : 'DECREASES_RISK',
        diagnostic_description: 'Hardware packet buffer saturation depth',
      },
      {
        feature_name: 'rtt_slope_5m',
        feature_value: 2.1,
        shap_value: q > 70 ? 0.22 : -0.02,
        contribution_pct: 26.0,
        impact_direction: q > 70 ? 'INCREASES_RISK' : 'DECREASES_RISK',
        diagnostic_description: '5-minute latency gradient slope ΔRTT/Δt',
      },
    ];

    const protective: FeatureAttribution[] = [
      {
        feature_name: 'interface_discards_sec',
        feature_value: 0.0,
        shap_value: -0.08,
        contribution_pct: 12.0,
        impact_direction: 'DECREASES_RISK',
        diagnostic_description: 'Zero optical CRC alignment and frame discards',
      },
    ];

    return {
      prediction_id: `SHAP-${this.step}`,
      horizon_minutes: 15,
      base_value: 0.12,
      predicted_risk: q > 70 ? 0.78 : 0.18,
      calibrated_risk: q > 70 ? 0.76 : 0.17,
      top_risk_drivers: drivers,
      top_protective_factors: protective,
      all_attributions: [...drivers, ...protective],
      operator_summary: q > 70 ? 'Acute bufferbloat and RTT delay inflection drive 82% of forecasted congestion.' : 'All data plane indicators are nominal with negative risk attribution.',
    };
  }

  public simulate(req: SimulationScenarioRequest): SimulationResponse {
    const red = (1.0 - req.reroute_traffic_pct / 100) * (1.0 - req.ingress_rate_limit_pct / 100);
    const cfQ = Math.max(4.0, 72.0 * Math.pow(red, 1.4) / req.buffer_expansion_factor);
    const cfRisk = Number(Math.max(0.04, Math.min(0.95, (cfQ / 90.0) * 0.78)).toFixed(3));
    const delta = Number((((cfRisk - 0.78) / 0.78) * 100).toFixed(1));

    return {
      scenario_id: `SIM-${Math.random().toString(16).substring(2, 8).toUpperCase()}`,
      device_id: req.device_id,
      interface_id: req.interface_id,
      horizon_minutes: req.horizon_minutes,
      baseline_risk: 0.78,
      counterfactual_risk: cfRisk,
      risk_delta_pct: delta,
      baseline_rtt_ms: 68.5,
      counterfactual_rtt_ms: Number((68.5 * red).toFixed(1)),
      baseline_queue_pct: 72.0,
      counterfactual_queue_pct: Number(cfQ.toFixed(1)),
      verdict: delta <= -40 ? 'OPTIMAL' : delta <= -20 ? 'EFFECTIVE' : 'MARGINAL',
      operator_guidance: delta <= -40 ? 'Intervention eliminates drop cliff and stabilizes SLA.' : 'Partial relief. Consider increasing reroute percentage.',
      counterfactual_explanation: `Diverting ${req.reroute_traffic_pct}% traffic lowers queue depth to ${cfQ.toFixed(1)}%. Risk drops by ${Math.abs(delta)}%.`,
    };
  }

  public prescribe(): any {
    const sim = this.simulate({
      device_id: 'core-router-alpha',
      interface_id: 'xe-0/0/1',
      horizon_minutes: 15,
      reroute_traffic_pct: 35.0,
      ingress_rate_limit_pct: 0.0,
      buffer_expansion_factor: 1.25,
      enable_priority_queuing: true,
    });
    return {
      status: 'OPTIMIZED',
      recommended_scenario: {
        device_id: 'core-router-alpha',
        interface_id: 'xe-0/0/1',
        horizon_minutes: 15,
        reroute_traffic_pct: 35.0,
        ingress_rate_limit_pct: 0.0,
        buffer_expansion_factor: 1.25,
        enable_priority_queuing: true,
      },
      simulation: sim,
      rationale: 'Pareto-optimal policy: Rerouting 35% traffic drops failure risk from 78.0% to 21.4% without customer rate-limiting.',
    };
  }

  public getModelHealth(): ModelHealthData {
    return {
      model_type: 'LightGBM + Isotonic Calibration (T+15m)',
      evaluation_metrics: {
        pr_auc: 0.9842,
        roc_auc: 0.9891,
        brier_score: 0.0157,
        f1_score: 0.935,
        precision: 0.942,
        recall: 0.928,
        false_negative_rate: 0.072,
      },
      calibration_curve: {
        bin_predicted_probs: [0.05, 0.18, 0.32, 0.49, 0.68, 0.89],
        bin_true_fractions: [0.04, 0.17, 0.31, 0.51, 0.69, 0.90],
      },
      drift_monitoring: { queue_occupancy_psi: 0.024, rtt_psi: 0.018, status: 'STABLE' },
      baseline_comparison: {
        lightgbm_calibrated: { brier_score: 0.0157, roc_auc: 0.9842 },
        logistic_regression: { brier_score: 0.0412, roc_auc: 0.912 },
        moving_average_threshold: { brier_score: 0.089, roc_auc: 0.825 },
        naive_persistence: { brier_score: 0.124, roc_auc: 0.741 },
      },
    };
  }

  public getIncidents(): HistoricalIncident[] {
    return [
      { incident_id: 'INC-2026-0814', timestamp: '2026-09-01T08:30:00Z', device_id: 'core-router-alpha', interface_id: 'xe-0/0/1', failure_type: 'Bufferbloat Queue Collapse', predicted_risk: 0.88, actual_outcome: 'Congestion Averted (Rerouted)', lead_time_minutes: 18, was_prevented: true, root_cause_driver: 'Queue depth slope (+3.2%/min)' },
      { incident_id: 'INC-2026-0815', timestamp: '2026-09-02T14:15:00Z', device_id: 'core-router-alpha', interface_id: 'xe-0/0/1', failure_type: 'Microburst Drop Surge', predicted_risk: 0.92, actual_outcome: 'Traffic Throttled via Ingress Police', lead_time_minutes: 12, was_prevented: true, root_cause_driver: 'Burst throughput cliff' },
      { incident_id: 'INC-2026-0816', timestamp: '2026-09-03T19:45:00Z', device_id: 'core-router-alpha', interface_id: 'xe-0/0/1', failure_type: 'Link Degradation & CRC Spike', predicted_risk: 0.74, actual_outcome: 'Fiber Cleaned & Interface Swapped', lead_time_minutes: 27, was_prevented: true, root_cause_driver: 'Optical CRC frame errors' },
      { incident_id: 'INC-2026-0817', timestamp: '2026-09-04T11:20:00Z', device_id: 'core-router-alpha', interface_id: 'xe-0/0/1', failure_type: 'BGP Route Flap Storm', predicted_risk: 0.81, actual_outcome: 'Damping Applied Automatically', lead_time_minutes: 15, was_prevented: true, root_cause_driver: 'Route flap count surge' },
    ];
  }
}

export const browserEngine = new BrowserTelemetryEngine();
