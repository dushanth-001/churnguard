import json
import os
import secrets
import sys
import threading
import time
from typing import Annotated

import joblib
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from inference import Predictor
from logger import setup_logger

app = FastAPI(title="ChurnGuard API")
logger = setup_logger()
model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl'))
predictor = Predictor(model)

API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class Stats:
    """In-memory request counters. They reset when the container restarts or scales to zero."""

    def __init__(self):
        self._lock = threading.Lock()
        self._started = time.time()
        self.requests = 0
        self.errors = 0
        self.positives = 0
        self._latency_sum = 0.0
        self._latency_max = 0.0

    def record(self, latency_ms, prediction=None, error=False):
        with self._lock:
            self.requests += 1
            self._latency_sum += latency_ms
            self._latency_max = max(self._latency_max, latency_ms)
            if error:
                self.errors += 1
            elif prediction == 1:
                self.positives += 1

    def snapshot(self):
        with self._lock:
            ok = self.requests - self.errors
            return {
                "uptime_seconds": round(time.time() - self._started, 1),
                "requests": self.requests,
                "errors": self.errors,
                "predicted_churn_rate": round(self.positives / ok, 4) if ok else None,
                "avg_latency_ms": round(self._latency_sum / self.requests, 2) if self.requests else None,
                "max_latency_ms": round(self._latency_max, 2),
            }


stats = Stats()


def verify_api_key(key: str = Security(api_key_header)):
    """Reject requests without the correct API key (fails closed if API_KEY is not set)."""
    if not API_KEY or not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


Binary = Annotated[int, Field(ge=0, le=1)]


class CustomerData(BaseModel):
    SeniorCitizen: Binary
    tenure: float
    Contract: Annotated[int, Field(ge=0, le=2)]
    MonthlyCharges: float
    TotalCharges: float
    TenureGroup: Annotated[int, Field(ge=0, le=3)]
    gender_Male: Binary
    Partner_Yes: Binary
    Dependents_Yes: Binary
    PhoneService_Yes: Binary
    MultipleLines_Yes: Binary
    InternetService_Fiber_optic: Binary
    InternetService_No: Binary
    OnlineSecurity_Yes: Binary
    OnlineBackup_Yes: Binary
    DeviceProtection_Yes: Binary
    TechSupport_Yes: Binary
    StreamingTV_Yes: Binary
    StreamingMovies_Yes: Binary
    PaperlessBilling_Yes: Binary
    PaymentMethod_Credit_card: Binary
    PaymentMethod_Electronic_check: Binary
    PaymentMethod_Mailed_check: Binary


@app.get("/")
def home():
    return {"message": "ChurnGuard API is running!"}


@app.get("/stats", dependencies=[Depends(verify_api_key)])
def get_stats():
    return stats.snapshot()


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(data: CustomerData):
    start = time.perf_counter()
    try:
        d = data.model_dump()
        numeric = {k: d[k] for k in ("tenure", "MonthlyCharges", "TotalCharges", "Contract")}
        prediction, probability = predictor.predict_one(d)
    except Exception as e:
        stats.record((time.perf_counter() - start) * 1000, error=True)
        logger.error(json.dumps({"event": "prediction_error", "error": str(e)}))
        raise HTTPException(status_code=500, detail="Prediction failed")

    latency_ms = (time.perf_counter() - start) * 1000
    stats.record(latency_ms, prediction=prediction)
    logger.info(json.dumps({
        "event": "prediction",
        "prediction": prediction,
        "probability": round(probability, 4),
        "latency_ms": round(latency_ms, 2),
        **numeric,
    }))
    return {"churn_prediction": prediction, "churn_probability": probability}
