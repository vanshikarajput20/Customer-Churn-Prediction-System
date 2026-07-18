import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import preprocess_data

def test_preprocess_data_handles_missing_total_charges():
    """Test that missing values (e.g., empty strings) in TotalCharges are handled properly by being coerced to NaN and dropped."""
    # Create a small dummy dataframe that mimics the structure of our raw data
    data = {
        'customerID': ['1', '2', '3'],
        'TotalCharges': ['100.5', ' ', '200.0'],
        'MonthlyCharges': [50.0, 50.0, 100.0],
        'PhoneService': ['Yes', 'Yes', 'No'],
        'MultipleLines': ['No', 'Yes', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['No', 'Yes', 'No internet service'],
        'DeviceProtection': ['No', 'No', 'No internet service'],
        'TechSupport': ['Yes', 'No', 'No internet service'],
        'StreamingTV': ['Yes', 'No', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service']
    }
    df = pd.DataFrame(data)
    
    # Preprocess
    cleaned_df = preprocess_data(df)
    
    # Assertions
    # 1. The row with empty TotalCharges (index 1) should be dropped
    assert len(cleaned_df) == 2
    
    # 2. customerID should be dropped
    assert 'customerID' not in cleaned_df.columns
    
    # 3. TotalCharges should now be a numeric type
    assert pd.api.types.is_numeric_dtype(cleaned_df['TotalCharges'])

def test_preprocess_data_feature_engineering():
    """Test that Total_Services and Monthly_to_Total_Ratio are calculated correctly."""
    data = {
        'TotalCharges': ['100.0', '300.0'],
        'MonthlyCharges': [50.0, 100.0],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['Yes', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No'],
        'OnlineBackup': ['No', 'Yes'],
        'DeviceProtection': ['No', 'No'],
        'TechSupport': ['Yes', 'No'],
        'StreamingTV': ['Yes', 'Yes'],
        'StreamingMovies': ['No', 'Yes']
    }
    df = pd.DataFrame(data)
    
    cleaned_df = preprocess_data(df)
    
    # First customer services: Phone (Yes), Multilines (Yes), Internet (DSL is not 'No'), 
    # Security (Yes), Backup (No), Protection (No), Support (Yes), TV (Yes), Movies (No)
    # Active: Phone, Multilines, Internet, Security, Support, TV -> 6 services
    # Wait, 'InternetService' isn't explicitly counted in our utils.py list! 
    # Let's check utils.py: 
    # services = ['PhoneService', 'MultipleLines', 'InternetService', 
    #             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    #             'TechSupport', 'StreamingTV', 'StreamingMovies']
    # If InternetService == 'DSL' it counts as 1.
    # Total for customer 1: Phone(1) + Multi(1) + Internet(1) + Security(1) + Support(1) + TV(1) = 6
    assert cleaned_df.iloc[0]['Total_Services'] == 6
    
    # Ratio check
    # Customer 1 ratio: 50.0 / 100.0 = 0.5
    assert cleaned_df.iloc[0]['Monthly_to_Total_Ratio'] == 0.5
    
    # Customer 2 ratio: 100.0 / 300.0 = 0.3333333333333333
    assert np.isclose(cleaned_df.iloc[1]['Monthly_to_Total_Ratio'], 1/3)
