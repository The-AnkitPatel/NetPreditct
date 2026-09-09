from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from backend.app.domain.telemetry_schemas import (
    RawTelemetryRecord,
    TelemetryIngestBatch,
    TelemetrySnapshot,
)
from backend.app.services.sliding_window import window_manager
from backend.app.services.stream_service import stream_service
from backend.app.services.prediction_service import prediction_service
from backend.app.collectors.real_interface_collector import real_collector

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.post("/ingest", response_model=Dict[str, Any])
def ingest_telemetry(batch: TelemetryIngestBatch):
    """Ingests a batch of raw network telemetry points."""
    if not batch.records:
        raise HTTPException(status_code=400, detail="Empty telemetry batch")
    for rec in batch.records:
        prediction_service.process_telemetry(rec)
    return {"status": "ok", "ingested_count": len(batch.records)}


@router.get("/latest", response_model=Dict[str, Any])
def get_latest_telemetry(
    device_id: str = "core-router-alpha", interface_id: str = "xe-0/0/1"
):
    """Retrieves current telemetry state and present anomaly status."""
    latest_rec = window_manager.get_latest(device_id, interface_id)
    if not latest_rec:
        # Fallback to advancing step if uninitialized
        latest_rec, _ = stream_service.advance_step()

    history_df = window_manager.to_dataframe(device_id, interface_id)
    anom_score, is_anom, anom_status = prediction_service.anomaly_detector.score_single(
        history_df.iloc[-1]
    )

    return {
        "telemetry": latest_rec.model_dump(),
        "anomaly": {
            "is_anomaly": is_anom,
            "anomaly_score": anom_score,
            "status": anom_status,
        },
    }


@router.get("/history", response_model=List[Dict[str, Any]])
def get_telemetry_history(
    device_id: str = "core-router-alpha",
    interface_id: str = "xe-0/0/1",
    limit: int = 60,
):
    """Returns the last N historical telemetry records for time-series charts."""
    records = window_manager.get_window_records(device_id, interface_id, count=limit)
    return [r.model_dump() for r in records]


@router.post("/stream/step", response_model=Dict[str, Any])
def step_telemetry_stream():
    """Manually steps the telemetry simulation forward by 1 minute interval."""
    record, pred = stream_service.advance_step()
    return {
        "record": record.model_dump(),
        "prediction": pred.model_dump(),
        "progress": stream_service.get_progress(),
    }


@router.post("/stream/jump-incident", response_model=Dict[str, Any])
def jump_to_incident():
    """Jumps timeline forward to a high-stress congestion event."""
    record, pred = stream_service.jump_to_incident()
    return {
        "record": record.model_dump(),
        "prediction": pred.model_dump(),
        "progress": stream_service.get_progress(),
    }


@router.post("/stream/reset", response_model=Dict[str, Any])
def reset_telemetry_stream():
    """Resets stream playhead back to baseline start."""
    record, pred = stream_service.reset_stream()
    return {
        "record": record.model_dump(),
        "prediction": pred.model_dump(),
        "progress": stream_service.get_progress(),
    }


@router.post("/live-poll", response_model=Dict[str, Any])
def poll_live_hardware_interface():
    """Polls real network adapter counters and runs live ML prediction."""
    record = real_collector.collect()
    pred = prediction_service.process_telemetry(record)
    return {
        "source": "live_hardware_adapter",
        "record": record.model_dump(),
        "prediction": pred.model_dump(),
    }

