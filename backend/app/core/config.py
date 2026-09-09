from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "NetPredict"
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*",
    ]

    # Telemetry Sliding Window Configuration
    SLIDING_WINDOW_CAPACITY: int = 720  # 720 steps (12 hours at 1m interval)
    MIN_STEPS_FOR_FEATURES: int = 30    # Minimum 30 steps to compute rolling stats

    # Operational Risk Thresholds
    RISK_NORMAL_MAX: float = 0.35
    RISK_ELEVATED_MAX: float = 0.70
    ANOMALY_THRESHOLD: float = 0.65

    # Prediction Horizons in minutes
    HORIZONS_MINUTES: List[int] = [5, 15, 30]
    PRIMARY_HORIZON_MINUTES: int = 15

    # Model Configuration
    RANDOM_SEED: int = 42
    MODEL_ARTIFACTS_DIR: str = "backend/data/models"
    DATASET_PATH: str = "backend/data/telemetry_trace.csv"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
