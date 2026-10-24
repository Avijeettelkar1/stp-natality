"""FastAPI application serving the selected birth-weight model."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.service import ModelService
from src.models.evaluate import LBW_THRESHOLD, MACROSOMIA_THRESHOLD

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="STP Natality Birth Weight API",
    version="1.0.0",
    description="University research prototype for newborn weight prediction.",
    docs_url=None,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
model_service = ModelService()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the prediction form."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/docs", include_in_schema=False)
def swagger_ui() -> object:
    """Serve Swagger UI from vendored assets so docs work without a CDN."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - Swagger UI",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png",
    )


def _risk_category(weight_grams: float) -> str:
    if weight_grams < LBW_THRESHOLD:
        return "low_birth_weight"
    if weight_grams > MACROSOMIA_THRESHOLD:
        return "macrosomia"
    return "normal"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ready" if model_service.ready else "model_unavailable",
        model_ready=model_service.ready,
        version=os.getenv("GIT_SHA", "local"),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**model_service.info())


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        predicted_weight = model_service.predict(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(
        predicted_weight_grams=predicted_weight,
        risk_category=_risk_category(predicted_weight),
        model_version=model_service.info()["model_version"],
    )
