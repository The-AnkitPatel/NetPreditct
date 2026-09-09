import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from backend.app.core.config import settings
from backend.app.services.feature_extractor import FEATURE_COLUMNS


class MultiHorizonPredictor:
    """
    Multi-horizon LightGBM predictive system anticipating network congestion
    and degradation at T+5m, T+15m, and T+30m ahead of time.
    Calibrated with Isotonic Regression for valid probabilistic interpretation.
    """

    def __init__(self, horizons: Optional[List[int]] = None):
        self.horizons = horizons or settings.HORIZONS_MINUTES
        self.classifiers: Dict[int, lgb.LGBMClassifier] = {}
        self.calibrators: Dict[int, IsotonicRegression] = {}
        self.rtt_regressor: Optional[lgb.LGBMRegressor] = None
        self.loss_regressor: Optional[lgb.LGBMRegressor] = None
        self.is_trained = False

    def train(
        self,
        X_train: pd.DataFrame,
        targets_train: Dict[int, np.ndarray],
        X_calib: pd.DataFrame,
        targets_calib: Dict[int, np.ndarray],
        future_rtt_train: np.ndarray,
        future_loss_train: np.ndarray,
    ) -> "MultiHorizonPredictor":
        """Trains LightGBM models and calibrates probabilities on hold-out calibration fold."""
        X_tr = X_train[FEATURE_COLUMNS]
        X_cal = X_calib[FEATURE_COLUMNS]

        for h in self.horizons:
            y_tr = targets_train[h]
            y_cal = targets_calib[h]

            clf = lgb.LGBMClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.06,
                num_leaves=24,
                class_weight="balanced",
                random_state=settings.RANDOM_SEED,
                verbose=-1,
            )
            clf.fit(X_tr, y_tr)
            self.classifiers[h] = clf

            # Isotonic Calibration on holdout validation fold
            val_probs = clf.predict_proba(X_cal)[:, 1]
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibrator.fit(val_probs, y_cal)
            self.calibrators[h] = calibrator

        # Continuous forecast regressors for T+15m
        self.rtt_regressor = lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.07, random_state=settings.RANDOM_SEED, verbose=-1
        )
        self.rtt_regressor.fit(X_tr, future_rtt_train)

        self.loss_regressor = lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.07, random_state=settings.RANDOM_SEED, verbose=-1
        )
        self.loss_regressor.fit(X_tr, future_loss_train)

        self.is_trained = True
        return self

    def predict_horizon(self, features: pd.Series, horizon_minutes: int) -> Tuple[float, float]:
        """
        Returns (raw_risk_score, calibrated_probability) for the chosen horizon.
        """
        if not self.is_trained:
            # Fallback heuristic
            q = float(features.get("queue_occupancy_pct", 0.0)) / 100.0
            loss = float(features.get("packet_loss_pct", 0.0)) / 2.0
            r = min(0.99, max(0.01, q * 0.7 + loss * 0.3))
            return round(r, 3), round(r, 3)

        sample = pd.DataFrame([features[FEATURE_COLUMNS]])
        raw_prob = float(self.classifiers[horizon_minutes].predict_proba(sample)[0, 1])
        cal_prob = float(self.calibrators[horizon_minutes].predict([raw_prob])[0])

        return round(raw_prob, 4), round(cal_prob, 4)

    def forecast_continuous(self, features: pd.Series) -> Tuple[float, float]:
        """Returns (predicted_rtt_ms, predicted_packet_loss_pct) at T+15m."""
        if not self.is_trained or self.rtt_regressor is None:
            rtt = float(features.get("rtt_ms", 15.0))
            loss = float(features.get("packet_loss_pct", 0.0))
            return round(rtt, 1), round(loss, 2)

        sample = pd.DataFrame([features[FEATURE_COLUMNS]])
        pred_rtt = float(self.rtt_regressor.predict(sample)[0])
        pred_loss = float(self.loss_regressor.predict(sample)[0])

        return round(max(5.0, pred_rtt), 1), round(max(0.0, pred_loss), 2)

    def save(self, directory: str = settings.MODEL_ARTIFACTS_DIR) -> None:
        """Serializes trained model artifacts safely."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.classifiers, os.path.join(directory, "classifiers.joblib"))
        joblib.dump(self.calibrators, os.path.join(directory, "calibrators.joblib"))
        joblib.dump(self.rtt_regressor, os.path.join(directory, "rtt_regressor.joblib"))
        joblib.dump(self.loss_regressor, os.path.join(directory, "loss_regressor.joblib"))

    def load(self, directory: str = settings.MODEL_ARTIFACTS_DIR) -> bool:
        """Loads model artifacts if present."""
        clf_path = os.path.join(directory, "classifiers.joblib")
        if not os.path.exists(clf_path):
            return False
        self.classifiers = joblib.load(clf_path)
        self.calibrators = joblib.load(os.path.join(directory, "calibrators.joblib"))
        self.rtt_regressor = joblib.load(os.path.join(directory, "rtt_regressor.joblib"))
        self.loss_regressor = joblib.load(os.path.join(directory, "loss_regressor.joblib"))
        self.is_trained = True
        return True
