"""HTTP request and response contracts for model serving."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Features available during a late-pregnancy or delivery assessment."""

    gestation_weeks: float = Field(ge=20, le=44)
    mother_age: float = Field(ge=10, le=60)
    weight_gain_pounds: float | None = Field(default=None, ge=0, le=150)
    born_alive_alive: int = Field(default=0, ge=0, le=30)
    born_alive_dead: int = Field(default=0, ge=0, le=30)
    born_dead: int = Field(default=0, ge=0, le=30)
    ever_born: int = Field(default=1, ge=1, le=30)
    father_age: float | None = Field(default=None, ge=10, le=100)
    plurality: int = Field(default=1, ge=1, le=9)
    year_fix: int = Field(ge=1986, le=2100)
    month: int = Field(ge=1, le=12)
    is_male: bool
    mother_married: bool | None = None
    lmp_known: bool = True


class PredictionResponse(BaseModel):
    predicted_weight_grams: float
    risk_category: str
    model_version: str
    intended_use_note: str = (
        "Research decision-support prototype; not a clinical diagnosis."
    )


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    data_contract: str
    features: list[str]
