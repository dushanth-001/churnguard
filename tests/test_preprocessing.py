
import pandas as pd
import sys
sys.path.append('src')
from data_preprocessing import clean_data

def test_clean_data_fills_missing_values():
    """Test பண்றோம்: Missing TotalCharges, 0 ஆ Fill ஆகுதா?"""

    # Fake, Small Test Data Create பண்றோம்
    test_df = pd.DataFrame({
        'TotalCharges': [' ', '100.5', '200.0'],
        'tenure': [0, 5, 10]
    })

    result = clean_data(test_df)

    # Check பண்றோம்: Missing Value 0 ஆ Fill ஆச்சா?
    assert result['TotalCharges'].iloc[0] == 0.0, "Missing value should be filled with 0"

    # Check பண்றோம்: Other Values Correctly Convert ஆச்சா?
    assert result['TotalCharges'].iloc[1] == 100.5, "Value should remain 100.5"

    print("Test Passed! ✅")

# Test Run பண்றோம்
test_clean_data_fills_missing_values()
