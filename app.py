import streamlit as st
import pandas as pd
from predict import predict_churn

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Customer Churn Prediction System")
st.markdown("Enter the customer's details below to predict if they are likely to churn (leave your service).")

# Create layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    
with col2:
    st.subheader("Account Details")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", 
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
with col3:
    st.subheader("Financials")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)

st.divider()

st.subheader("Services Subscribed")
col4, col5, col6, col7 = st.columns(4)

with col4:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multi_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    
with col5:
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

with col6:
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

with col7:
    support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    stream_mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# Build the payload
customer_data = {
    'gender': gender,
    'SeniorCitizen': 1 if senior == "Yes" else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone,
    'MultipleLines': multi_lines,
    'InternetService': internet,
    'OnlineSecurity': security,
    'OnlineBackup': backup,
    'DeviceProtection': protection,
    'TechSupport': support,
    'StreamingTV': stream_tv,
    'StreamingMovies': stream_mov,
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

st.divider()
submit = st.button("Predict Churn Probability", type="primary", use_container_width=True)

if submit:
    with st.spinner("Analyzing customer profile..."):
        result = predict_churn(customer_data)
        
        if "error" in result:
            st.error(result["error"])
        else:
            prob = result["probability"]
            pred = result["prediction"]
            
            st.subheader("Prediction Results")
            
            # Display metrics side by side
            rc1, rc2 = st.columns(2)
            with rc1:
                if pred == 1:
                    st.error("⚠️ HIGH RISK OF CHURN")
                else:
                    st.success("✅ LIKELY TO STAY")
            with rc2:
                st.metric(label="Churn Probability", value=f"{prob * 100:.1f}%")
                
            st.progress(prob)
