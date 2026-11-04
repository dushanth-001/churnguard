"""Single-request inference helper used by the API."""
import numpy as np
import pandas as pd

MODE = "numpy_booster"
API_TO_MODEL = {
    "InternetService_Fiber_optic": "InternetService_Fiber optic",
    "PaymentMethod_Credit_card": "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check",
}
MODEL_TO_API = {v: k for k, v in API_TO_MODEL.items()}


class Predictor:
    """Turns one API payload into (label, probability).

    Modes:
      legacy           DataFrame, predict and predict_proba (two model calls); kept for benchmarking
      dataframe_proba  DataFrame, one predict_proba call, label = probability > threshold
      numpy_booster    NumPy array and Booster.inplace_predict, label = probability > threshold
    """

    def __init__(self, model, mode=MODE, threshold=0.5):
        self.model = model
        self.mode = mode
        self.threshold = threshold
        self.features = list(model.feature_names_in_)
        self.api_names = [MODEL_TO_API.get(f, f) for f in self.features]
        self.booster = model.get_booster() if mode == "numpy_booster" else None

    def predict_one(self, fields):
        row = [fields[name] for name in self.api_names]
        if self.mode == "legacy":
            df = pd.DataFrame([row], columns=self.features)
            return int(self.model.predict(df)[0]), float(self.model.predict_proba(df)[0][1])
        if self.mode == "numpy_booster":
            arr = np.asarray([row], dtype=np.float32)
            proba = float(self.booster.inplace_predict(arr, validate_features=False)[0])
        else:
            df = pd.DataFrame([row], columns=self.features)
            proba = float(self.model.predict_proba(df)[0][1])
        return int(proba > self.threshold), proba
