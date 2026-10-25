
import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from logger import setup_logger

app = FastAPI(title="ChurnGuard API")
logger = setup_logger()
model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl'))

class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: float
    Contract: int
    MonthlyCharges: float
    TotalCharges: float
    TenureGroup: int
    gender_Male: int
    Partner_Yes: int
    Dependents_Yes: int
    PhoneService_Yes: int
    MultipleLines_Yes: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    OnlineSecurity_Yes: int
    OnlineBackup_Yes: int
    DeviceProtection_Yes: int
    TechSupport_Yes: int
    StreamingTV_Yes: int
    StreamingMovies_Yes: int
    PaperlessBilling_Yes: int
    PaymentMethod_Credit_card: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

@app.get("/")
def home():
    return {"message": "ChurnGuard API is running!"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        d = data.dict()
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
        return {"error": str(e)}
