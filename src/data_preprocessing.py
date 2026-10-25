
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(url):
    """ Load dataset from the specified URL."""
    df = pd.read_csv(url)
    return df

def clean_data(df):
    """ Clean data by converting TotalCharges to numeric and handling missing values."""
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

def engineer_features(df):
    """Feature Engineering - Drop ID, Encode Churn, Create TenureGroup"""
    df = df.drop('customerID', axis=1)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    def tenure_group(tenure):
        if tenure <= 12:
            return 0
        elif tenure <= 24:
            return 1
        elif tenure <= 48:
            return 2
        else:
            return 3

    df['TenureGroup'] = df['tenure'].apply(tenure_group)

    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    df['Contract'] = df['Contract'].map(contract_map)

    return df

def preprocess_data(df, cols_to_scale):
    """ Apply standard feature scaling to continuous numerical columns."""
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df, scaler
