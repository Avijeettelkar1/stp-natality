"""FastAPI application serving the selected birth-weight model."""

import logging
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from src.api.monitoring_store import MonitoringStore
from src.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    SupportRequest,
    SupportResponse,
)
from src.api.service import ModelService
from src.models.evaluate import (
    LBW_THRESHOLD,
    MACROSOMIA_THRESHOLD,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


app = FastAPI(
    title="STP Natality Birth Weight API",
    version="1.1.0",
    description="University research prototype for newborn weight prediction.",
    docs_url=None,
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


# ----------------------------
# Models
# ----------------------------

# Current production model.
model_service = ModelService()

# Optional shadow/challenger model.
challenger_service = ModelService(
    env_var="CHALLENGER_MODEL_PATH",
    use_default=False,
)


# ----------------------------
# Database
# ----------------------------

monitoring_store = MonitoringStore()


# ----------------------------
# Prometheus metrics
# ----------------------------

HTTP_REQUESTS = Counter(
    "stp_http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status"],
)

HTTP_LATENCY = Histogram(
    "stp_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "stp_predictions_total",
    "Prediction requests.",
    ["result"],
)

SHADOW_PREDICTION_DIFFERENCE = Histogram(
    "stp_shadow_prediction_difference_grams",
    "Absolute difference between champion and challenger predictions.",
)

MONITORING_WRITE_ERRORS = Counter(
    "stp_monitoring_write_errors_total",
    "Monitoring database write failures.",
    ["operation"],
)


@app.middleware("http")
async def prometheus_middleware(
    request: Request,
    call_next,
):
    started = perf_counter()

    response = await call_next(request)

    duration = perf_counter() - started

    HTTP_REQUESTS.labels(
        method=request.method,
        path=request.url.path,
        status=response.status_code,
    ).inc()

    HTTP_LATENCY.labels(
        method=request.method,
        path=request.url.path,
    ).observe(duration)

    return response


# ----------------------------
# UI
# ----------------------------


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/docs", include_in_schema=False)
def swagger_ui() -> object:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - Swagger UI",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png",
    )


# ----------------------------
# Helpers
# ----------------------------


def _risk_category(
    weight_grams: float,
) -> str:
    if weight_grams < LBW_THRESHOLD:
        return "low_birth_weight"

    if weight_grams > MACROSOMIA_THRESHOLD:
        return "macrosomia"

    return "normal"


# ----------------------------
# Health
# ----------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(
        status=("ready" if model_service.ready else "model_unavailable"),
        model_ready=model_service.ready,
        challenger_ready=challenger_service.ready,
        database_ready=monitoring_store.ready,
        version=os.getenv(
            "GIT_SHA",
            "local",
        ),
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**model_service.info())


# ----------------------------
# Prediction
# ----------------------------


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:

    prediction_started = perf_counter()

    try:
        champion_prediction = model_service.predict(request)
    except RuntimeError as exc:
        PREDICTIONS.labels(result="champion_error").inc()

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    champion_info = model_service.info()

    challenger_prediction = None
    challenger_info = None

    # Shadow inference:
    # Challenger failure must NEVER affect the user's prediction.
    if challenger_service.ready:
        try:
            challenger_prediction = challenger_service.predict(request)

            challenger_info = challenger_service.info()

            SHADOW_PREDICTION_DIFFERENCE.observe(
                abs(champion_prediction - challenger_prediction)
            )

        except Exception:
            logger.exception("Challenger prediction failed.")

    latency_ms = (perf_counter() - prediction_started) * 1000

    risk_category = _risk_category(champion_prediction)

    prediction_id = uuid4()

    # Monitoring failure should NOT break production prediction.
    if monitoring_store.ready:
        try:
            monitoring_store.log_prediction(
                prediction_id=prediction_id,
                inputs=request.model_dump(mode="json"),
                champion_name=champion_info["model_name"],
                champion_version=champion_info["model_version"],
                champion_prediction_g=(champion_prediction),
                challenger_name=(
                    challenger_info["model_name"] if challenger_info else None
                ),
                challenger_version=(
                    challenger_info["model_version"] if challenger_info else None
                ),
                challenger_prediction_g=(challenger_prediction),
                risk_category=risk_category,
                latency_ms=latency_ms,
                app_version=os.getenv(
                    "GIT_SHA",
                    "local",
                ),
            )

        except Exception:
            MONITORING_WRITE_ERRORS.labels(operation="prediction").inc()

            logger.exception("Could not write prediction " "monitoring record.")

    PREDICTIONS.labels(result="success").inc()

    return PredictionResponse(
        prediction_id=prediction_id,
        predicted_weight_grams=(champion_prediction),
        risk_category=risk_category,
        model_version=champion_info["model_version"],
    )


# ----------------------------
# Ground-truth feedback
# ----------------------------


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
)
def feedback(
    request: FeedbackRequest,
) -> FeedbackResponse:

    if not monitoring_store.ready:
        raise HTTPException(
            status_code=503,
            detail=("Monitoring database " "is not configured."),
        )

    try:
        monitoring_store.save_feedback(
            prediction_id=(request.prediction_id),
            actual_weight_g=(request.actual_weight_g),
        )

    except Exception as exc:
        MONITORING_WRITE_ERRORS.labels(operation="feedback").inc()

        logger.exception("Could not save feedback.")

        raise HTTPException(
            status_code=500,
            detail="Could not save feedback.",
        ) from exc

    return FeedbackResponse(status="saved")


# ----------------------------
# Support requests
# ----------------------------


@app.post(
    "/support",
    response_model=SupportResponse,
)
def support(
    request: SupportRequest,
) -> SupportResponse:

    if not monitoring_store.ready:
        raise HTTPException(
            status_code=503,
            detail=("Monitoring database " "is not configured."),
        )

    try:
        support_id = monitoring_store.save_support_request(
            category=request.category,
            message=request.message,
        )

    except Exception as exc:
        MONITORING_WRITE_ERRORS.labels(operation="support").inc()

        logger.exception("Could not save support request.")

        raise HTTPException(
            status_code=500,
            detail=("Could not save " "support request."),
        ) from exc

    return SupportResponse(
        id=support_id,
        status="saved",
    )


# ----------------------------
# Prometheus
# ----------------------------


@app.get(
    "/metrics",
    include_in_schema=False,
)
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
