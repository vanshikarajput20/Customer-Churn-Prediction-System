import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import urllib.request
import joblib

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = "data/Telco-Customer-Churn.csv"

def load_data():
    """Download data if it doesn't exist and load it."""
    if not os.path.exists("data"):
        os.makedirs("data")
        
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(DATA_URL, context=ctx) as u, open(DATA_PATH, 'wb') as f:
            f.write(u.read())
        print("Download complete.")
        
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess_data(df):
    """
    Handle missing values, clean data, and do feature engineering.
    """
    # 1. Handle missing values
    # TotalCharges is object because of some blank spaces ' '
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop the 11 rows with NaN TotalCharges (they have 0 tenure and haven't paid yet)
    df = df.dropna(subset=['TotalCharges'])
    
    # Remove duplicates if any
    df = df.drop_duplicates()
    
    # Drop customerID as it is not useful for modeling
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # 2. Feature Engineering
    # Feature 1: Total Services Count
    # Customers with more services might be more engaged and less likely to churn.
    services = ['PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # We count how many services they have active (ignoring "No internet service" or "No phone service")
    df['Total_Services'] = df[services].apply(lambda x: (x != 'No') & (x != 'No internet service') & (x != 'No phone service')).sum(axis=1)

    # Feature 2: Monthly Charges to Total Charges Ratio
    # This might show if they are on a high monthly plan but recently joined
    df['Monthly_to_Total_Ratio'] = df['MonthlyCharges'] / df['TotalCharges']
    
    return df

def prepare_modeling_data(df):
    """
    Encode categorical variables, scale numerical features, and split the data.
    """
    # Separate target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    # Label Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
    # Scale numerical columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save the scaler and encoders for prediction later
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(numerical_cols, "models/numerical_cols.pkl")
    joblib.dump(categorical_cols, "models/categorical_cols.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()
