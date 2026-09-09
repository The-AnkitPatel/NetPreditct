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
