"""Evaluate the deployed model on the held-out test set and write metrics.json."""
import datetime
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from data_preprocessing import clean_data, engineer_features, load_data

DATA_URL = ('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/'
            'master/data/Telco-Customer-Churn.csv')
FIX_COLS = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']
ONEHOT_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
               'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
               'PaymentMethod']
SCALE_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']


def build_features(df):
    """Clean the raw data and build model-ready features."""
    df = engineer_features(clean_data(df))
    for col in FIX_COLS:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})
    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=True)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def main():
    df = build_features(load_data(DATA_URL))
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))

    X_test = X_test.copy()
    X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])
    X_test = X_test[model.feature_names_in_]

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        'evaluated_on': datetime.date.today().isoformat(),
        'dataset': 'Telco Customer Churn (IBM), 7043 customers',
        'split': '70/15/15 stratified, random_state=42',
        'test_set_size': int(len(y_test)),
        'threshold': 0.5,
        'recall': round(float(recall_score(y_test, pred)), 4),
        'precision': round(float(precision_score(y_test, pred)), 4),
        'f1': round(float(f1_score(y_test, pred)), 4),
        'accuracy': round(float(accuracy_score(y_test, pred)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, proba)), 4),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }
    with open(os.path.join(BASE_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
