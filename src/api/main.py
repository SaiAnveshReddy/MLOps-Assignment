"""
CardioRisk serving API (Task 6 - Model Containerization).

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from api.logging_config import get_logger
from api.metrics import PREDICTION_COUNT, REQUEST_COUNT, REQUEST_LATENCY
from heart_mlops.inference import INPUT_FIELDS, predict_risk

logger = get_logger()

app = FastAPI(
    title="CardioRisk API",
    description="Heart disease risk classifier - BITS AIMLCZG523 Assignment 1",
    version="0.1.0",
)


class PatientInput(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=0, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0)
    slope: int = Field(..., ge=1, le=3)
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., description="3 = normal, 6 = fixed defect, 7 = reversible defect")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 57, "sex": 1, "cp": 4, "trestbps": 140, "chol": 289,
                "fbs": 0, "restecg": 2, "thalach": 145, "exang": 1,
                "oldpeak": 2.0, "slope": 2, "ca": 1, "thal": 7,
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: int
    risk_label: str
    confidence: float
    probability_disease: float


@app.middleware("http")
async def log_and_measure_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    REQUEST_COUNT.labels(
        method=request.method, path=request.url.path, status_code=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(path=request.url.path).observe(duration)

    logger.info(
        "request_handled",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
        },
    )
    return response


@app.get("/")
def root():
    return {"service": "cardiorisk-api", "status": "ok", "predict_fields": INPUT_FIELDS}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    try:
        result = predict_risk(patient.model_dump())
    except (FileNotFoundError, ValueError) as exc:
        logger.error("prediction_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=400, detail=str(exc))

    PREDICTION_COUNT.labels(risk_label=result["risk_label"]).inc()
    return result
