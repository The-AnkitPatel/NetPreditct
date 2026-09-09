from fastapi import APIRouter
from typing import Dict, Any
import numpy as np
import pandas as pd
from backend.app.services.dataset_loader import ensure_dataset_exists
from backend.app.services.feature_extractor import (
    extract_features_from_dataframe,
    FEATURE_COLUMNS,
)
from backend.app.services.prediction_service import prediction_service
from backend.app.ml.evaluator import (
    evaluate_binary_predictions,
    compute_calibration_data,
    compute_population_stability_index,
)
from backend.app.ml.baseline_models import (
    PersistenceBaseline,
    MovingAverageThresholdBaseline,
    LogisticRegressionBaseline,
)

router = APIRouter(prefix="/health", tags=["Health & Monitoring"])


@router.get("", response_model=Dict[str, Any])
def get_system_health():
    """System liveness check and model deployment status."""
    return {
        "status": "HEALTHY",
        "system": "NetPredict",
        "version": "1.0.0",
        "is_model_trained": prediction_service.predictor.is_trained,
        "is_explainer_ready": prediction_service.explainer is not None,
        "is_conformal_calibrated": prediction_service.conformal.is_calibrated,
    }


@router.get("/model", response_model=Dict[str, Any])
def get_model_health_and_drift():
    """
    Computes rigorous ML validation metrics, calibration curve,
    baseline comparisons, and PSI feature drift on test split.
    """
    df = ensure_dataset_exists()
    feat_df = extract_features_from_dataframe(df)

    # Use hold-out test split (last 20% of timeline)
    n = len(df)
    test_idx = int(n * 0.80)
    X_test = feat_df.iloc[test_idx:]
    y_test = df["target_t15"].iloc[test_idx:].values

    # 1. Calibrated LightGBM Evaluation
    lgb_probs = []
    for i in range(len(X_test)):
        _, p = prediction_service.predictor.predict_horizon(X_test.iloc[i], 15)
        lgb_probs.append(p)
    lgb_probs = np.array(lgb_probs)

    metrics_lgb = evaluate_binary_predictions(y_test, lgb_probs)
    calib_curve = compute_calibration_data(y_test, lgb_probs)

    # 2. Baseline Model Comparisons
    train_idx = int(n * 0.65)
    X_tr = feat_df.iloc[:train_idx]
    y_tr = df["target_t15"].iloc[:train_idx].values

    # Logistic Regression
    lr_base = LogisticRegressionBaseline().fit(X_tr, y_tr)
    lr_probs = lr_base.predict_proba(X_test)[:, 1]
    metrics_lr = evaluate_binary_predictions(y_test, lr_probs)

    # Moving Average Baseline
    ma_base = MovingAverageThresholdBaseline().fit(X_tr, y_tr)
    ma_probs = ma_base.predict_proba(X_test)[:, 1]
    metrics_ma = evaluate_binary_predictions(y_test, ma_probs)

    # Persistence Baseline
    pers_base = PersistenceBaseline().fit(X_tr, y_tr)
    pers_probs = pers_base.predict_proba(X_test)[:, 1]
    metrics_pers = evaluate_binary_predictions(y_test, pers_probs)

    # 3. Population Stability Index (PSI) Feature Drift Detection
    # Compare training distribution to test distribution for Queue Occupancy
    psi_queue = compute_population_stability_index(
        X_tr["queue_occupancy_pct"].values, X_test["queue_occupancy_pct"].values
    )
    psi_rtt = compute_population_stability_index(
        X_tr["rtt_ms"].values, X_test["rtt_ms"].values
    )

    drift_status = "STABLE"
    if max(psi_queue, psi_rtt) >= 0.25:
        drift_status = "SIGNIFICANT_DRIFT_ALERT"
    elif max(psi_queue, psi_rtt) >= 0.10:
        drift_status = "MODERATE_SHIFT"

    return {
        "model_type": "LightGBM + Isotonic Calibration (T+15m)",
        "evaluation_metrics": metrics_lgb,
        "calibration_curve": calib_curve,
        "drift_monitoring": {
            "queue_occupancy_psi": psi_queue,
            "rtt_psi": psi_rtt,
            "status": drift_status,
        },
        "baseline_comparison": {
            "lightgbm_calibrated": metrics_lgb,
            "logistic_regression": metrics_lr,
            "moving_average_threshold": metrics_ma,
            "naive_persistence": metrics_pers,
        },
    }
