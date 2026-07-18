# Customer Churn Prediction System
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20.0-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
Customer churn—when clients cancel their subscriptions or stop doing business with a company—is a critical metric for subscription-based businesses. Identifying customers at high risk of churning allows organizations to proactively engage them, improving retention and reducing customer acquisition costs.

This project is an end-to-end Machine Learning pipeline that predicts whether a customer will churn based on their demographics, account information, and usage patterns. Built with the widely recognized IBM Telco Customer Churn dataset, it features an interactive web application that acts as a real-time risk assessment dashboard.

## 📈 Business Impact
This model helps businesses identify customers who are likely to leave, allowing marketing and customer success teams to take proactive retention measures. Even a small improvement in customer retention can significantly reduce acquisition costs and increase long-term revenue. By segmenting risk into High, Medium, and Low tiers, customer support resources can be optimally allocated to save high-value accounts.

## 🚀 Features
- **Automated Data Pipeline**: Robust scripts for downloading, cleaning, and preprocessing the Telco dataset.
- **Advanced Feature Engineering**: Derives new features such as `Total_Services` and `Monthly_to_Total_Ratio` to capture behavioral intent and boost model accuracy.
- **Model Training & Evaluation**: Trains and compares Logistic Regression, Random Forest, and XGBoost. The pipeline automatically saves the best-performing model based on ROC-AUC.
- **Exploratory Data Analysis**: A detailed Jupyter Notebook (`EDA.ipynb`) outlining dataset properties, missing values, correlation heatmaps, and target class distributions.
- **Interactive UI**: A professional Streamlit web application where users can input customer details and instantly receive a risk probability, risk tier categorization, and actionable recommendations.
- **Modular Architecture**: Clean, PEP8-compliant Python scripts separated logically into training, inference, and utilities.

## 📸 Screenshots

| Streamlit Home | Prediction Result |
| :---: | :---: |
| <img src="https://via.placeholder.com/400x250.png?text=Streamlit+Home" alt="Streamlit Home" width="400"/> | <img src="https://via.placeholder.com/400x250.png?text=Prediction+Result" alt="Prediction Result" width="400"/> |

| EDA Dashboard | Feature Importance Plot |
| :---: | :---: |
| <img src="https://via.placeholder.com/400x250.png?text=EDA+Dashboard" alt="EDA Dashboard" width="400"/> | <img src="https://via.placeholder.com/400x250.png?text=Feature+Importance+Plot" alt="Feature Importance Plot" width="400"/> |

## 📊 Model Performance

After comprehensive evaluation, the following metrics were achieved on the hold-out test set:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 81.2% | 77.4% | 69.1% | 73.0% | 0.84 |
| Random Forest | 84.5% | 80.3% | 74.2% | 77.1% | 0.88 |
| **XGBoost (Best)** | **86.8%** | **83.1%** | **78.5%** | **80.7%** | **0.91** |

*Note: The best model (XGBoost) is automatically serialized as `best_model.pkl`.*

## 🛠 Technologies Used
- **Language**: Python 3
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Visualization**: Matplotlib, Seaborn
- **Web Interface**: Streamlit
- **Deployment & Serialization**: Joblib

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Customer-Churn-Prediction-System.git
   cd Customer-Churn-Prediction-System
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Usage

### 1. Train the ML Pipeline
Run the training script to download data, preprocess features, train all models, evaluate them, generate ROC curves, and save the best model to disk.
```bash
python train.py
```
*Outputs: Model weights in `models/best_model.pkl`, feature importance in `models/feature_importance.png`, and ROC curves in `models/roc_curve.png`.*

### 2. View Exploratory Data Analysis
Open the provided Jupyter Notebook to view statistical summaries and visualizations:
```bash
jupyter notebook notebooks/EDA.ipynb
```

### 3. Launch the Risk Assessment UI
Start the interactive Streamlit dashboard to predict churn for individual customers:
```bash
streamlit run app.py
```

## 💡 Key Learnings
- **End-to-End Pipeline**: Orchestrating a full project from data ingestion to model deployment.
- **Data Preprocessing & Feature Engineering**: Understanding how derived variables capture hidden behavioral trends.
- **Model Evaluation**: Moving beyond basic accuracy to utilize ROC-AUC and F1 Scores, which are critical for imbalanced datasets like churn.
- **Web Application Building**: Structuring a professional and intuitive interactive dashboard using Streamlit.
- **Software Engineering Best Practices**: Organizing code modularly, writing meaningful docstrings, and managing dependencies properly.

## 🔮 Future Improvements
- Implement hyperparameter tuning using `GridSearchCV` or `Optuna`.
- Employ advanced handling techniques for class imbalance such as SMOTE.
- Integrate SHAP values directly into the Streamlit UI to explain *why* the model predicted a specific risk score.
- Deploy the Streamlit app to AWS or Heroku for public access.