"""Build models/reference.csv: validation-set inputs and predicted probabilities."""
import os
import sys

import joblib
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from data_preprocessing import load_data
from evaluate import DATA_URL, SCALE_COLS, build_features

MONITORED_INPUTS = ["tenure", "MonthlyCharges", "TotalCharges", "Contract"]


def main():
    df = build_features(load_data(DATA_URL))
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    model = joblib.load(os.path.join(BASE_DIR, "models", "xgboost_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

    X_val = X_val.copy()
    X_val[SCALE_COLS] = scaler.transform(X_val[SCALE_COLS])
    X_val = X_val[model.feature_names_in_]

    ref = X_val[MONITORED_INPUTS].copy()
    ref["probability"] = model.predict_proba(X_val)[:, 1]
    out = os.path.join(BASE_DIR, "models", "reference.csv")
    ref.to_csv(out, index=False)
    print(f"Wrote {out} with {len(ref)} rows")


if __name__ == "__main__":
    main()
