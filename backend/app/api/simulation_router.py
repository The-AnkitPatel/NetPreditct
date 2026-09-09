from fastapi import APIRouter
from backend.app.domain.simulation_schemas import (
    SimulationScenarioRequest,
    SimulationResponse,
)
from backend.app.services.simulation_service import simulation_service

router = APIRouter(prefix="/simulate", tags=["Simulation"])


@router.post("", response_model=SimulationResponse)
def simulate_intervention_scenario(req: SimulationScenarioRequest):
    """
    Evaluates candidate operator interventions (traffic reroute %, shaping, queue expansion)
    counterfactually through the calibrated predictive models.
    """
    return simulation_service.evaluate_scenario(req)


@router.post("/prescribe")
def prescribe_optimal_mitigation(
    device_id: str = "core-router-alpha",
    interface_id: str = "xe-0/0/1",
    horizon_minutes: int = 15,
):
    """
    Autonomous Prescription: Runs bounded multi-objective optimization over the surrogate
    model to determine the mathematically optimal mitigation policy.
    """
    return simulation_service.prescribe_optimal_intervention(
        device_id=device_id, interface_id=interface_id, horizon_minutes=horizon_minutes
    )

