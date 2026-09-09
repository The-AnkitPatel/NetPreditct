from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from backend.app.domain.prediction_schemas import PredictionResponse, HistoricalIncident
from backend.app.services.prediction_service import prediction_service
from backend.app.services.sliding_window import window_manager
from backend.app.services.stream_service import stream_service

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.get("/horizons", response_model=PredictionResponse)
def get_multi_horizon_prediction(
    device_id: str = "core-router-alpha", interface_id: str = "xe-0/0/1"
):
    """
    Computes real-time multi-horizon predictive forecasts (T+5m, T+15m, T+30m).
    Distinguishes present anomaly score from future predicted failure risks.
    """
    latest_rec = window_manager.get_latest(device_id, interface_id)
    if not latest_rec:
        latest_rec, pred = stream_service.advance_step()
        return pred

    pred = prediction_service.process_telemetry(latest_rec)
    return pred


@router.get("/incidents", response_model=List[HistoricalIncident])
def get_historical_incidents():
    """Returns validated historical incident timeline for prediction-vs-actual analysis."""
    return prediction_service.historical_incidents
