import os
import numpy as np
import pandas as pd
from backend.app.core.config import settings
from backend.app.core.logging_config import logger
from backend.app.services.dataset_loader import ensure_dataset_exists
from backend.app.services.feature_extractor import extract_features_from_dataframe, FEATURE_COLUMNS
from backend.app.services.prediction_service import prediction_service
from backend.app.ml.shap_explainer import NetworkShapExplainer


def initialize_and_train_system() -> None:
    """
    Bootstrap sequence: ensures dataset exists, extracts temporal features,
    trains and calibrates multi-horizon LightGBM classifiers, fits Isolation Forest,
    initializes TreeSHAP explainer, and sets up conformal prediction bounds.
    """
    logger.info("Initializing NetPredict ML Models and Dataset...")
    df = ensure_dataset_exists()
    logger.info(f"Loaded telemetry dataset with {len(df)} samples.")

    # 1. Check if pre-trained models already exist on disk
    loaded = prediction_service.predictor.load(settings.MODEL_ARTIFACTS_DIR)
    if not loaded:
        logger.info("No saved models found. Extracting features and training from trace data...")
        feat_df = extract_features_from_dataframe(df)

        # Chronological split: 65% Train, 15% Calibration, 20% Test
        n = len(df)
        train_idx = int(n * 0.65)
        calib_idx = int(n * 0.80)

        X_train = feat_df.iloc[:train_idx]
        X_calib = feat_df.iloc[train_idx:calib_idx]
        X_test = feat_df.iloc[calib_idx:]

        targets_train = {
            5: df["target_t5"].iloc[:train_idx].values,
            15: df["target_t15"].iloc[:train_idx].values,
            30: df["target_t30"].iloc[:train_idx].values,
        }
        targets_calib = {
            5: df["target_t5"].iloc[train_idx:calib_idx].values,
            15: df["target_t15"].iloc[train_idx:calib_idx].values,
            30: df["target_t30"].iloc[train_idx:calib_idx].values,
        }

        # Train Multi-Horizon LightGBM and Calibrate
        prediction_service.predictor.train(
            X_train=X_train,
            targets_train=targets_train,
            X_calib=X_calib,
            targets_calib=targets_calib,
            future_rtt_train=df["future_rtt_t15"].iloc[:train_idx].values,
            future_loss_train=df["future_loss_t15"].iloc[:train_idx].values,
        )
        prediction_service.predictor.save(settings.MODEL_ARTIFACTS_DIR)
        logger.info("Multi-horizon LightGBM models trained and serialized.")

        # Fit Anomaly Detector on clean baseline training fold
        prediction_service.anomaly_detector.fit(df.iloc[:train_idx])

        # Calibrate Conformal Predictor on calibration fold
        calib_pred_rtts = []
        calib_pred_losses = []
        for i in range(len(X_calib)):
            r, l = prediction_service.predictor.forecast_continuous(X_calib.iloc[i])
            calib_pred_rtts.append(r)
            calib_pred_losses.append(l)

        prediction_service.conformal.calibrate(
            y_true_rtt=df["future_rtt_t15"].iloc[train_idx:calib_idx].values,
            y_pred_rtt=np.array(calib_pred_rtts),
            y_true_loss=df["future_loss_t15"].iloc[train_idx:calib_idx].values,
            y_pred_loss=np.array(calib_pred_losses),
        )
    else:
        logger.info("Successfully loaded pre-trained models from disk.")
        # Fit anomaly detector if not yet fit
        prediction_service.anomaly_detector.fit(df.iloc[:2000])

    # Initialize TreeSHAP on primary horizon classifier (T+15m)
    primary_clf = prediction_service.predictor.classifiers.get(settings.PRIMARY_HORIZON_MINUTES)
    if primary_clf:
        explainer = NetworkShapExplainer(primary_clf)
        prediction_service.set_explainer(explainer)
        logger.info("TreeSHAP explainer successfully initialized.")


if __name__ == "__main__":
    initialize_and_train_system()
