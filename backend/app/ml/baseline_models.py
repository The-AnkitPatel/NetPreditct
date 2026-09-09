import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class PersistenceBaseline:
    """Predicts future state will match the current observed state."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PersistenceBaseline":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Heuristic: if current queue > 75% or loss > 0.5%, persist state as 1
        is_high = (X["queue_occupancy_pct"] > 75.0) | (X["packet_loss_pct"] > 0.5)
        probs = np.where(is_high, 0.85, 0.15)
        return np.column_stack([1 - probs, probs])


class MovingAverageThresholdBaseline:
    """Thresholds recent 15-minute moving average of queue depth and utilization."""

    def __init__(self, queue_threshold: float = 70.0):
        self.queue_threshold = queue_threshold

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MovingAverageThresholdBaseline":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Uses 15m rolling queue mean
        col = "queue_mean_15m" if "queue_mean_15m" in X.columns else "queue_occupancy_pct"
        scores = np.clip(X[col] / 100.0, 0.05, 0.95)
        return np.column_stack([1 - scores, scores])


class LogisticRegressionBaseline:
    """Standard L2-regularized logistic regression over scaled features."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogisticRegressionBaseline":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
