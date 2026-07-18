import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predict import predict_churn

# Configure the Streamlit page
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, professional look
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
    .risk-high {
        color: #DC2626;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-medium {
        color: #D97706;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-low {
        color: #16A34A;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Customer Churn Prediction System")
st.markdown("""
Welcome to the Customer Churn Prediction dashboard. This tool utilizes advanced machine learning 
to evaluate customer profiles and predict the likelihood of service cancellation. 
Please configure the customer's attributes in the sidebar to generate a live risk assessment.
""")

st.divider()

# Sidebar for inputs
st.sidebar.header("Customer Profile")
st.sidebar.markdown("Configure the demographics, account details, and services below.")

# 1. Demographics
st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

# 2. Account Details
st.sidebar.subheader("Account Details")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", 
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=10.0, max_value=200.0, value=75.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=900.0)

# 3. Services Subscribed
st.sidebar.subheader("Services Subscribed")
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multi_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
stream_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
stream_mov = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

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

# Main Panel layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Action")
    submit = st.button("Generate Risk Assessment", type="primary", use_container_width=True)

    if submit:
        with st.spinner("Analyzing profile..."):
            result = predict_churn(customer_data)
            
            if "error" in result:
                st.error(result["error"])
            else:
                prob = result["probability"]
                pred = result["prediction"]
                
                # Determine Risk Level
                if prob > 0.70:
                    risk_level = "High Risk"
                    risk_class = "risk-high"
                    action_msg = "Immediate retention efforts recommended. Consider offering a customized discount or reaching out via a dedicated success manager."
                elif prob > 0.40:
                    risk_level = "Medium Risk"
                    risk_class = "risk-medium"
                    action_msg = "Monitor closely. Engage with targeted marketing campaigns to increase product adoption."
                else:
                    risk_level = "Low Risk"
                    risk_class = "risk-low"
                    action_msg = "Customer is stable. Continue providing standard excellent service."
                
                st.markdown("### Assessment Results")
                st.metric(label="Calculated Churn Probability", value=f"{prob * 100:.1f}%")
                st.progress(prob)
                
                st.markdown(f"**Risk Category:** <span class='{risk_class}'>{risk_level}</span>", unsafe_allow_html=True)
                st.info(f"**Recommended Action:** {action_msg}")

with col2:
    st.subheader("Profile Overview")
    if submit and "error" not in result:
        # Display key factors chart
        st.markdown("This chart illustrates the individual customer's selected financial parameters.")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        categories = ['Monthly Charges', 'Average Past Months']
        
        # Avoid division by zero
        avg_past = total_charges / tenure if tenure > 0 else monthly_charges
        
        values = [monthly_charges, avg_past]
        
        colors = ['#1E3A8A', '#94A3B8']
        ax.bar(categories, values, color=colors, width=0.5)
        ax.set_ylabel("Amount ($)")
        ax.set_title("Financial Snapshot")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v + 2, f"${v:.1f}", ha='center', va='bottom', fontweight='bold')
            
        st.pyplot(fig)
    else:
        st.info("Configure the profile on the left sidebar and click 'Generate Risk Assessment' to view the analysis.")

st.divider()

st.markdown("""
<div style='text-align: center; color: #64748B; font-size: 0.9rem;'>
    Designed for American Express | Machine Learning Internship Project Evaluation
</div>
""", unsafe_allow_html=True)
