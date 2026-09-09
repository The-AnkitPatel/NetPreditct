from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.core.logging_config import logger
from backend.app.services.model_manager import initialize_and_train_system
from backend.app.api.telemetry_router import router as telemetry_router
from backend.app.api.predict_router import router as predict_router
from backend.app.api.explain_router import router as explain_router
from backend.app.api.simulation_router import router as simulation_router
from backend.app.api.health_router import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown events."""
    logger.info(f"Starting {settings.PROJECT_NAME} Predictive Engine...")
    initialize_and_train_system()
    logger.info(f"{settings.PROJECT_NAME} ready to receive telemetry.")
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME} services.")


from fastapi.responses import RedirectResponse

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "**ML-Based Network Failure & Congestion Prediction System**\n\n"
        "--- \n"
        "### 📚 Canonical Fumadocs Platform Handbook\n"
        "The full interactive architectural documentation portal (Fumadocs) is hosted at: "
        "[**http://localhost:3000/docs**](http://localhost:3000/docs)\n\n"
        "--- \n"
        "*(This page is the raw OpenAPI / Swagger interface for direct REST debugging)*"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/portal", include_in_schema=False)
def redirect_to_fumadocs():
    """Redirects directly to the Fumadocs documentation portal."""
    return RedirectResponse(url="http://localhost:3000/docs")


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
api_v1_prefix = settings.API_V1_STR
app.include_router(telemetry_router, prefix=api_v1_prefix)
app.include_router(predict_router, prefix=api_v1_prefix)
app.include_router(explain_router, prefix=api_v1_prefix)
app.include_router(simulation_router, prefix=api_v1_prefix)
app.include_router(health_router, prefix=api_v1_prefix)


@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "docs": "/docs",
        "api_v1": settings.API_V1_STR,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
