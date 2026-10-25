
import pandas as pd
import sys
sys.path.append('src')
from data_preprocessing import clean_data

def test_clean_data_fills_missing_values():
    """ Verify that missing or blank TotalCharges values are imputed with 0.0."""

    # Create synthetic test dataset 
    test_df = pd.DataFrame({
        'TotalCharges': [' ', '100.5', '200.0'],
        'tenure': [0, 5, 10]
    })

    result = clean_data(test_df)

    # Assert that missing/blank values are converted and imputed to 0.0 
    assert result['TotalCharges'].iloc[0] == 0.0, "Missing value should be filled with 0"

    # Assert that valid numeric strings are accurately converted to float 
    assert result['TotalCharges'].iloc[1] == 100.5, "Value should remain 100.5"

    print("Test Passed! ✅")

# Execute test assertion 
test_clean_data_fills_missing_values()
