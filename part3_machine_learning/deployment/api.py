"""
deployment/api.py — Part 3, Task 6.3: Deployment Simulation (FastAPI)
=========================================================================
A minimal FastAPI service simulating how the traffic volume regression model
and the accident-risk classification model could be served in production.
This is a deployment SIMULATION for the capstone, not a hardened production
service (see "Limitations" in mlops/model_versioning.md).

Run with:   uvicorn api:app --reload --port 8000
Then try:   curl -X POST http://127.0.0.1:8000/predict/volume \
              -H "Content-Type: application/json" \
              -d '{"hour": 8, "day_type": "weekday", "weather": "Clear", "is_holiday": false}'
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

for _candidate in (Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts"):
    if (_candidate / "common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
from common import FEATURE_COLUMNS, MODELS_DIR, WEATHER_OPTIONS, build_feature_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart City Traffic Intelligence API (deployment simulation)",
    description="Serves the Part 3 traffic-volume regression and accident-risk classification models.",
    version="1.0.0",
)

_volume_model = None
_risk_model = None


def get_volume_model():
    global _volume_model
    if _volume_model is None:
        _volume_model = joblib.load(MODELS_DIR / "regression_GradientBoostingRegressor.joblib")
        logger.info("Loaded regression model: regression_GradientBoostingRegressor.joblib")
    return _volume_model


def get_risk_model():
    global _risk_model
    if _risk_model is None:
        _risk_model = joblib.load(MODELS_DIR / "classification_RandomForestClassifier.joblib")
        logger.info("Loaded classification model: classification_RandomForestClassifier.joblib")
    return _risk_model


class ScenarioRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of day, 0-23")
    day_type: str = Field(..., description="'weekday' or 'weekend'")
    weather: str = Field("Clear", description=f"One of {WEATHER_OPTIONS}")
    is_holiday: bool = Field(False)


class VolumeResponse(BaseModel):
    predicted_traffic_volume: float


class RiskResponse(BaseModel):
    high_risk_probability: float
    high_risk_prediction: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/volume", response_model=VolumeResponse)
def predict_volume(req: ScenarioRequest):
    try:
        row = build_feature_frame([req.hour], req.day_type, req.weather, req.is_holiday)
    except ValueError as exc:
        logger.error("Invalid request to /predict/volume: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))

    model = get_volume_model()
    prediction = float(model.predict(row[FEATURE_COLUMNS])[0])
    logger.info("Predicted volume for %s: %.1f", req.model_dump(), prediction)
    return VolumeResponse(predicted_traffic_volume=round(prediction, 1))


@app.post("/predict/risk", response_model=RiskResponse)
def predict_risk(req: ScenarioRequest):
    try:
        row = build_feature_frame([req.hour], req.day_type, req.weather, req.is_holiday)
    except ValueError as exc:
        logger.error("Invalid request to /predict/risk: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))

    model = get_risk_model()
    proba = float(model.predict_proba(row[FEATURE_COLUMNS])[0, 1])
    logger.info("Predicted risk for %s: %.3f", req.model_dump(), proba)
    return RiskResponse(high_risk_probability=round(proba, 4), high_risk_prediction=proba >= 0.5)
