import os
import joblib
import pandas as pd
import numpy as np
import shap

def load_prediction_assets():
    """
    Loads the trained model, scaler, encoders, and feature structures from disk.
    
    Returns:
        tuple: (model, scaler, label_encoders, numerical_cols, categorical_cols, feature_columns)
    """
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    numerical_cols = joblib.load("models/numerical_cols.pkl")
    categorical_cols = joblib.load("models/categorical_cols.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    
    return model, scaler, label_encoders, numerical_cols, categorical_cols, feature_columns

def predict_churn(customer_data: dict):
    """
    Takes a dictionary of raw customer data, applies the same preprocessing steps 
    as the training pipeline, outputs a churn prediction, and calculates SHAP values.
    
    Args:
        customer_data (dict): Single customer profile.
        
    Returns:
        dict: Contains prediction (1 or 0), probability score, and top SHAP features.
    """
    try:
        model, scaler, label_encoders, numerical_cols, categorical_cols, feature_columns = load_prediction_assets()
    except FileNotFoundError:
        return {"error": "Model files not found. Please run train.py first to generate the necessary files."}
        
    df = pd.DataFrame([customer_data])
    
    # 1. Apply Feature Engineering
    services = [
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Aggregate services count
    df['Total_Services'] = df[services].apply(
        lambda x: (x != 'No') & (x != 'No internet service') & (x != 'No phone service')
    ).sum(axis=1)
    
    # Ensure TotalCharges is numeric, handle zeros to avoid division by zero
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    df['Monthly_to_Total_Ratio'] = np.where(
        df['TotalCharges'] == 0, 0, df['MonthlyCharges'] / df['TotalCharges']
    )
    
    # 2. Apply Encoding
    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders[col]
            if df[col].iloc[0] not in le.classes_:
                df[col] = le.transform([le.classes_[0]])[0]
            else:
                df[col] = le.transform(df[col])
        else:
            df[col] = 0
            
    # 3. Apply Scaling
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Ensure columns match training shape exactly
    X = df[feature_columns]
    
    # Generate prediction and probability
    prob = model.predict_proba(X)[0][1]
    pred = int(prob > 0.5)
    
    # 4. Calculate SHAP values for Explainability
    try:
        # Determine explainer type based on model
        if hasattr(model, 'feature_importances_'):
            # Tree-based model
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            # Depending on model (e.g. Random Forest), shap_values might be a list for multi-class/binary
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Take the positive class
        else:
            # Linear model (e.g. Logistic Regression)
            # Use a background dataset of zeros (mean of scaled data)
            background = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X)
            
        # Ensure array shape is flat
        shap_values = np.array(shap_values).flatten()
        
        # Zip features with their shap values and sort by absolute impact
        feature_impacts = list(zip(feature_columns, shap_values))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top 3 features
        top_3_features = [{"feature": f, "impact": float(val)} for f, val in feature_impacts[:3]]
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        top_3_features = []
    
    return {
        "prediction": pred,
        "probability": float(prob),
        "top_features": top_3_features
    }

if __name__ == "__main__":
    # Smoke test
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
    print("Testing prediction on sample data...")
    result = predict_churn(sample)
    print(f"Prediction result: {result}")
