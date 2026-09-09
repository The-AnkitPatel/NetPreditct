from fastapi import APIRouter, HTTPException
from backend.app.domain.explain_schemas import ShapExplanationResponse
from backend.app.services.prediction_service import prediction_service
from backend.app.services.sliding_window import window_manager
from backend.app.services.feature_extractor import extract_single_step_features
from backend.app.core.config import settings

router = APIRouter(prefix="/explain", tags=["Explainability"])


@router.get("/latest", response_model=ShapExplanationResponse)
def get_latest_explanation(
    device_id: str = "core-router-alpha", interface_id: str = "xe-0/0/1"
):
    """Generates TreeSHAP feature attributions and root-cause explanation for current network state."""
    if not prediction_service.explainer:
        raise HTTPException(status_code=503, detail="TreeSHAP explainer not yet initialized")

    history_df = window_manager.to_dataframe(device_id, interface_id)
    if len(history_df) == 0:
        raise HTTPException(status_code=404, detail="No telemetry available for explanation")

    features = extract_single_step_features(history_df)
    raw_risk, cal_risk = prediction_service.predictor.predict_horizon(
        features, settings.PRIMARY_HORIZON_MINUTES
    )

    explanation = prediction_service.explainer.explain_sample(
        features=features,
        prediction_id="LATEST",
        predicted_risk=raw_risk,
        calibrated_risk=cal_risk,
        horizon_minutes=settings.PRIMARY_HORIZON_MINUTES,
    )
    return explanation


@router.get("/{prediction_id}", response_model=ShapExplanationResponse)
def get_explanation_by_id(prediction_id: str):
    """Retrieves TreeSHAP explanation for a specific historical prediction ID."""
    cached = prediction_service.cached_predictions.get(prediction_id)
    if not cached:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found in cache")

    pred_resp, features = cached
    primary = pred_resp.primary_horizon

    explanation = prediction_service.explainer.explain_sample(
        features=features,
        prediction_id=prediction_id,
        predicted_risk=primary.failure_risk_score,
        calibrated_risk=primary.calibrated_probability,
        horizon_minutes=primary.horizon_minutes,
    )
    return explanation
