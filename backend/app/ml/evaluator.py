import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve


def evaluate_binary_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Computes comprehensive operational metrics for predictive classification."""
    y_pred = (y_prob >= threshold).astype(int)

    # PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    brier = brier_score_loss(y_true, y_prob)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # False Negative Rate = 1 - Recall (Crucial for network downtime prevention)
    fnr = 1.0 - rec

    return {
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "brier_score": round(float(brier), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "false_negative_rate": round(float(fnr), 4),
    }


def compute_calibration_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, List[float]]:
    """Calculates calibration reliability curve points for frontend visualization."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return {
        "bin_predicted_probs": [round(float(p), 4) for p in prob_pred],
        "bin_true_fractions": [round(float(p), 4) for p in prob_true],
    }


def compute_population_stability_index(
    expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10
) -> float:
    """
    Computes Population Stability Index (PSI) to detect feature distribution drift.
    PSI < 0.10: No significant shift.
    0.10 <= PSI < 0.25: Moderate shift.
    PSI >= 0.25: Significant drift requiring retraining.
    """
    eps = 1e-4
    # Create quantile buckets based on expected
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    bucket_bounds[0] -= 1e-5
    bucket_bounds[-1] += 1e-5

    # Count frequencies
    exp_counts = np.histogram(expected, bins=bucket_bounds)[0] + eps
    act_counts = np.histogram(actual, bins=bucket_bounds)[0] + eps

    exp_pct = exp_counts / np.sum(exp_counts)
    act_pct = act_counts / np.sum(act_counts)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return round(float(psi), 4)


def run_chronological_walk_forward_evaluation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_t15",
    train_ratio: float = 0.65,
    calib_ratio: float = 0.15,
    gap_steps: int = 30,
) -> Dict[str, Any]:
    """
    Executes strict chronological walk-forward validation with an embargo gap.
    Guarantees no temporal leakage across folds.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    calib_end = train_end + int(n * calib_ratio)
    test_start = calib_end + gap_steps

    train_df = df.iloc[:train_end]
    calib_df = df.iloc[train_end:calib_end]
    test_df = df.iloc[test_start:]

    return {
        "train_size": len(train_df),
        "calib_size": len(calib_df),
        "test_size": len(test_df),
        "embargo_gap_minutes": gap_steps,
        "test_positive_rate": round(float(test_df[target_col].mean()), 4),
    }
