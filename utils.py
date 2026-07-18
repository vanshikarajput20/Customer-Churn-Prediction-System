import os
import urllib.request
import ssl
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Data source URL
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = "data/Telco-Customer-Churn.csv"

def load_data():
    """
    Downloads the Telco Customer Churn dataset if it doesn't exist locally,
    and loads it into a Pandas DataFrame.
    
    Returns:
        pd.DataFrame: Raw dataset.
    """
    if not os.path.exists("data"):
        os.makedirs("data")
        
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(DATA_URL, context=ctx) as response, open(DATA_PATH, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete.")
        
    return pd.read_csv(DATA_PATH)

def preprocess_data(df):
    """
    Cleans the raw data, handles missing values, and applies feature engineering.
    
    Feature Engineering details:
    - Total_Services: Aggregates the number of active services a customer has. 
      Customers heavily invested in multiple services are typically less likely to churn.
    - Monthly_to_Total_Ratio: Captures the proportion of monthly charge relative to the total charge. 
      This highlights newly joined customers with high plans, who might be high flight risks.
      
    Args:
        df (pd.DataFrame): Raw dataset.
        
    Returns:
        pd.DataFrame: Cleaned and augmented dataset.
    """
    # 1. Handle missing values
    # TotalCharges is read as an object due to blank spaces (' ') for some customers
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with NaN in TotalCharges (typically new customers with 0 tenure)
    df = df.dropna(subset=['TotalCharges'])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Drop customerID as it contains no predictive power
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # 2. Feature Engineering
    services = [
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Count how many services are active (ignoring negative/null indicators)
    df['Total_Services'] = df[services].apply(
        lambda x: (x != 'No') & (x != 'No internet service') & (x != 'No phone service')
    ).sum(axis=1)

    # Calculate ratio of monthly charges to total charges
    df['Monthly_to_Total_Ratio'] = df['MonthlyCharges'] / df['TotalCharges']
    
    return df

def prepare_modeling_data(df):
    """
    Prepares data for machine learning models by encoding categorical variables, 
    scaling numerical features, and splitting into train/test sets. 
    It also saves the encoders and scaler for deployment.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_columns)
    """
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Identify feature types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    # Encode categorical columns (Label Encoding for simplicity)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
    # Scale numerical columns to ensure standard normal distribution
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split dataset (stratified to maintain churn distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save the preprocessing artifacts for inference
    if not os.path.exists("models"):
        os.makedirs("models")
        
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(numerical_cols, "models/numerical_cols.pkl")
    joblib.dump(categorical_cols, "models/categorical_cols.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()
