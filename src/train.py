
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

def split_data(df, target_col='Churn'):
    """ Split data into stratified Train, Validation, and Test sets."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_xgboost(X_train, y_train, scale_weight):
    """ Tune hyperparameters and train the XGBoost model using RandomizedSearchCV."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0]
    }

    xgb_base = XGBClassifier(random_state=42, scale_pos_weight=scale_weight, eval_metric='logloss')

    search = RandomizedSearchCV(
        estimator=xgb_base, param_distributions=param_grid,
        n_iter=20, scoring='recall', cv=3, random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def save_model(model, path='churnguard/models/xgboost_model.pkl'):
    """ Serialize and save the trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")
