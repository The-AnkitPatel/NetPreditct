import pandas as pd
import numpy as np
from backend.app.services.dataset_loader import ensure_dataset_exists
from backend.app.services.feature_extractor import extract_features_from_dataframe
from backend.app.services.prediction_service import prediction_service
from backend.app.services.model_manager import initialize_and_train_system
from backend.app.ml.evaluator import evaluate_binary_predictions
from backend.app.ml.baseline_models import (
    PersistenceBaseline,
    MovingAverageThresholdBaseline,
    LogisticRegressionBaseline,
)


def run_benchmark():
    initialize_and_train_system()
    df = ensure_dataset_exists()
    feat_df = extract_features_from_dataframe(df)

    n = len(df)
    train_idx = int(n * 0.65)
    test_idx = int(n * 0.80)

    X_train = feat_df.iloc[:train_idx]
    y_train = df["target_t15"].iloc[:train_idx].values

    X_test = feat_df.iloc[test_idx:]
    y_test = df["target_t15"].iloc[test_idx:].values

    print(f"\n=======================================================")
    print(f"NETPREDICT BENCHMARK EVALUATION (T+15m Horizon)")
    print(f"Test Set Size: {len(X_test)} samples | Incident Rate: {y_test.mean()*100:.2f}%")
    print(f"=======================================================\n")

    # 1. Calibrated LightGBM
    lgb_probs = []
    for i in range(len(X_test)):
        _, p = prediction_service.predictor.predict_horizon(X_test.iloc[i], 15)
        lgb_probs.append(p)
    lgb_metrics = evaluate_binary_predictions(y_test, np.array(lgb_probs))

    # 2. Logistic Regression
    lr = LogisticRegressionBaseline().fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_binary_predictions(y_test, lr_probs)

    # 3. Moving Average
    ma = MovingAverageThresholdBaseline().fit(X_train, y_train)
    ma_probs = ma.predict_proba(X_test)[:, 1]
    ma_metrics = evaluate_binary_predictions(y_test, ma_probs)

    # 4. Persistence
    pers = PersistenceBaseline().fit(X_train, y_train)
    pers_probs = pers.predict_proba(X_test)[:, 1]
    pers_metrics = evaluate_binary_predictions(y_test, pers_probs)

    results = {
        "LightGBM (Calibrated)": lgb_metrics,
        "Logistic Regression": lr_metrics,
        "Moving Average (15m)": ma_metrics,
        "Naive Persistence": pers_metrics,
    }

    results_df = pd.DataFrame(results).T[
        ["pr_auc", "roc_auc", "brier_score", "f1_score", "false_negative_rate"]
    ]
    print(results_df.to_string())
    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()
