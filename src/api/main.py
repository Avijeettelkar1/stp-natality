"""FastAPI application serving the selected birth-weight model."""

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.service import ModelService
from src.models.evaluate import LBW_THRESHOLD, MACROSOMIA_THRESHOLD

app = FastAPI(
    title="STP Natality Birth Weight API",
    version="1.0.0",
    description="University research prototype for newborn weight prediction.",
)
model_service = ModelService()


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
