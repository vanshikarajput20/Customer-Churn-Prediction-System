import os
import joblib
import pandas as pd
import numpy as np

def load_prediction_assets():
    """Load the saved model, scaler, and encoders."""
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    numerical_cols = joblib.load("models/numerical_cols.pkl")
    categorical_cols = joblib.load("models/categorical_cols.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    
    return model, scaler, label_encoders, numerical_cols, categorical_cols, feature_columns

def predict_churn(customer_data: dict):
    """
    Takes a dictionary of customer data, preprocesses it exactly like the training data,
    and returns a prediction and probability.
    """
    try:
        model, scaler, label_encoders, numerical_cols, categorical_cols, feature_columns = load_prediction_assets()
    except FileNotFoundError:
        return {"error": "Model files not found. Please run train.py first."}
        
    df = pd.DataFrame([customer_data])
    
    # 1. Feature Engineering (match what utils.py does)
    services = ['PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df['Total_Services'] = df[services].apply(lambda x: (x != 'No') & (x != 'No internet service') & (x != 'No phone service')).sum(axis=1)
    
    # Ensure TotalCharges is numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    
    df['Monthly_to_Total_Ratio'] = np.where(df['TotalCharges'] == 0, 0, df['MonthlyCharges'] / df['TotalCharges'])
    
    # 2. Encoding
    for col in categorical_cols:
        if col in df.columns:
            # Handle unseen labels by setting to 0 (or most frequent)
            le = label_encoders[col]
            # If the user input is not in classes, we pick the first class to prevent error
            if df[col].iloc[0] not in le.classes_:
                df[col] = 0
            else:
                df[col] = le.transform(df[col])
        else:
            df[col] = 0
            
    # 3. Scaling
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Ensure column order matches training exactly
    X = df[feature_columns]
    
    # Predict
    prob = model.predict_proba(X)[0][1]
    pred = int(prob > 0.5)
    
    return {
        "prediction": pred,
        "probability": float(prob)
    }

if __name__ == "__main__":
    # Test prediction
    sample = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 95.0,
        'TotalCharges': 1140.0
    }
    print("Testing prediction on sample data:")
    result = predict_churn(sample)
    print(result)
