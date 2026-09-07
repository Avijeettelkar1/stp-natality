"""Model loading and inference logic shared by the API and tests."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.api.schemas import PredictionRequest
from src.models.pipeline import MODEL_FEATURES

DEFAULT_MODEL_PATH = Path("models/birth_weight_pipeline.joblib")


def prepare_features(request: PredictionRequest) -> pd.DataFrame:
    """Convert validated raw inputs to the model's exact feature contract."""
    values = request.model_dump()
    month = values["month"]
    gestation = values["gestation_weeks"]
    mother_age = values["mother_age"]
    plurality = values["plurality"]

    values.update(
        {
            "parity": (
                values["born_alive_alive"]
                + values["born_alive_dead"]
                + values["born_dead"]
            ),
            "birth_month_sin": np.sin(2 * np.pi * month / 12),
            "birth_month_cos": np.cos(2 * np.pi * month / 12),
            "is_multiple_birth": int(plurality > 1),
            "gestation_preterm": int(gestation < 37),
            "gestation_post_term": int(gestation > 41),
            "mother_age_group": (
                0
                if mother_age <= 19
                else 1 if mother_age <= 29 else 2 if mother_age <= 39 else 3
            ),
        }
    )
    values["is_male"] = int(values["is_male"])
    values["mother_married"] = (
        np.nan if values["mother_married"] is None else int(values["mother_married"])
    )
    values["lmp_known"] = int(values["lmp_known"])
    return pd.DataFrame(
        [{feature: values.get(feature, np.nan) for feature in MODEL_FEATURES}]
    ).astype("float64")


class ModelService:
    """Load a versioned sklearn pipeline and expose stable prediction methods."""

    def __init__(self, model_path: Path | None = None):
        configured = os.getenv("MODEL_PATH")
        self.model_path = model_path or Path(configured or DEFAULT_MODEL_PATH)
        self.metadata_path = self.model_path.with_name("model_metadata.json")
        self.model = None
        self.metadata: dict = {}
        self.load()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.model_path.exists():
            return
        self.model = joblib.load(self.model_path)
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def predict(self, request: PredictionRequest) -> float:
        if self.model is None:
            raise RuntimeError(
                "Model artifact is unavailable. Run scripts/train.py first."
            )
        prediction = self.model.predict(prepare_features(request))[0]
        return round(float(prediction), 1)

    def info(self) -> dict:
        return {
            "model_name": self.metadata.get("model_name", "unavailable"),
            "model_version": self.metadata.get("model_version", "unavailable"),
            "data_contract": self.metadata.get("data_contract", "natality-26-v1"),
            "features": self.metadata.get("features", MODEL_FEATURES),
        }
