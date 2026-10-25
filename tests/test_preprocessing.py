import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import clean_data, engineer_features


def test_clean_data_fills_missing_values():
    df = pd.DataFrame({'TotalCharges': [' ', '100.5'], 'tenure': [0, 5]})
    result = clean_data(df)
    assert result['TotalCharges'].iloc[0] == 0.0
    assert result['TotalCharges'].iloc[1] == 100.5


def test_engineer_features_encodes_churn_and_drops_id():
    df = pd.DataFrame({
        'customerID': ['A1', 'B2'],
        'Churn': ['Yes', 'No'],
        'tenure': [5, 60],
        'Contract': ['Month-to-month', 'Two year']
    })
    result = engineer_features(df)
    assert 'customerID' not in result.columns
    assert list(result['Churn']) == [1, 0]
    assert list(result['TenureGroup']) == [0, 3]
    assert list(result['Contract']) == [0, 2]
