import os
import sys

os.environ["API_KEY"] = "test-key"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

PAYLOAD = {
    "SeniorCitizen": 0, "tenure": -1.2, "Contract": 0, "MonthlyCharges": 1.5,
    "TotalCharges": -0.8, "TenureGroup": 0, "gender_Male": 1, "Partner_Yes": 0,
    "Dependents_Yes": 0, "PhoneService_Yes": 1, "MultipleLines_Yes": 0,
    "InternetService_Fiber_optic": 1, "InternetService_No": 0, "OnlineSecurity_Yes": 0,
    "OnlineBackup_Yes": 0, "DeviceProtection_Yes": 0, "TechSupport_Yes": 0,
    "StreamingTV_Yes": 0, "StreamingMovies_Yes": 0, "PaperlessBilling_Yes": 1,
    "PaymentMethod_Credit_card": 0, "PaymentMethod_Electronic_check": 1,
    "PaymentMethod_Mailed_check": 0,
}


def test_health_is_open():
    assert client.get("/").status_code == 200


def test_predict_requires_key():
    assert client.post("/predict", json=PAYLOAD).status_code == 401


def test_predict_rejects_wrong_key():
    r = client.post("/predict", json=PAYLOAD, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_predict_with_valid_key():
    r = client.post("/predict", json=PAYLOAD, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    body = r.json()
    assert body["churn_prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_rejects_invalid_input():
    bad = dict(PAYLOAD, gender_Male=5)
    r = client.post("/predict", json=bad, headers={"X-API-Key": "test-key"})
    assert r.status_code == 422
