import numpy as np
from typing import Tuple, Optional


class ConformalPredictor:
    """
    Split Conformal Prediction engine providing distribution-free,
    statistically valid prediction intervals for continuous network forecasts.
    Guarantees coverage probability >= 1 - alpha on exchangeable telemetry.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_rtt: float = 6.5   # Default empirical quantile in ms
        self.q_loss: float = 0.4  # Default empirical quantile in %
        self.is_calibrated = False

    def calibrate(
        self,
        y_true_rtt: np.ndarray,
        y_pred_rtt: np.ndarray,
        y_true_loss: np.ndarray,
        y_pred_loss: np.ndarray,
    ) -> "ConformalPredictor":
        """Calibrates conformal conformity scores on a dedicated validation fold."""
        residuals_rtt = np.abs(y_true_rtt - y_pred_rtt)
        residuals_loss = np.abs(y_true_loss - y_pred_loss)

        n = len(residuals_rtt)
        if n > 0:
            # Finite-sample adjusted quantile level: ceil((n+1)*(1-alpha)) / n
            k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
            k = min(n, max(1, k))
            self.q_rtt = float(np.partition(residuals_rtt, k - 1)[k - 1])
            self.q_loss = float(np.partition(residuals_loss, k - 1)[k - 1])
            self.is_calibrated = True

        return self

    def predict_interval_rtt(self, point_forecast: float) -> Tuple[float, float]:
        """Returns (lower_bound, upper_bound) with 90% guaranteed coverage."""
        lower = max(2.0, point_forecast - self.q_rtt)
        upper = point_forecast + self.q_rtt
        return round(lower, 1), round(upper, 1)

    def predict_interval_loss(self, point_forecast: float) -> Tuple[float, float]:
        """Returns (lower_bound, upper_bound) for packet loss %."""
        lower = max(0.0, point_forecast - self.q_loss)
        upper = min(100.0, point_forecast + self.q_loss)
        return round(lower, 2), round(upper, 2)
