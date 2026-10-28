import os
import sys
import secrets
from typing import Annotated

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from logger import setup_logger

app = FastAPI(title="ChurnGuard API")
logger = setup_logger()
model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl'))

API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


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


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(data: CustomerData):
    try:
        d = data.model_dump()
        d['InternetService_Fiber optic'] = d.pop('InternetService_Fiber_optic')
        d['PaymentMethod_Credit card (automatic)'] = d.pop('PaymentMethod_Credit_card')
        d['PaymentMethod_Electronic check'] = d.pop('PaymentMethod_Electronic_check')
        d['PaymentMethod_Mailed check'] = d.pop('PaymentMethod_Mailed_check')
        input_df = pd.DataFrame([d])[model.feature_names_in_]
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        logger.info(f"Prediction made: {prediction}, Probability: {probability}")
        return {"churn_prediction": int(prediction), "churn_probability": float(probability)}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
