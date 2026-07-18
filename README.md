# Customer Churn Prediction System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20.0-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning pipeline that predicts customer churn using the IBM Telco Customer Churn dataset, paired with an interactive Streamlit dashboard for real-time risk assessment.

---

## Table of Contents

- [Overview](#overview)
- [Business Impact](#business-impact)
- [Features](#features)
- [Screenshots](#screenshots)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

Customer churn — when clients cancel subscriptions or stop doing business with a company — is one of the most critical metrics for subscription-based businesses. Identifying at-risk customers early allows organizations to intervene proactively, improving retention and lowering acquisition costs.

This project builds a complete ML pipeline — from data ingestion and feature engineering to model training and deployment — and wraps it in a clean, interactive web app for real-time customer risk scoring.

## Business Impact

The model flags customers likely to churn, giving marketing and customer success teams a head start on retention efforts. Since acquiring a new customer typically costs far more than retaining an existing one, even small gains in retention translate into meaningful revenue impact. Customers are further segmented into **High**, **Medium**, and **Low** risk tiers, so support resources can be focused where they matter most.

## Features

- **Automated data pipeline** — scripts for downloading, cleaning, and preprocessing the Telco dataset
- **Feature engineering** — derived features like `Total_Services` and `Monthly_to_Total_Ratio` to capture behavioral intent
- **Model training & evaluation** — trains and compares Logistic Regression, Random Forest, and XGBoost, automatically saving the best performer by ROC-AUC
- **Exploratory Data Analysis** — a dedicated notebook (`EDA.ipynb`) covering data quality, correlations, and class distribution
- **Interactive UI** — a Streamlit app that returns risk probability, risk tier, and actionable recommendations for any customer profile
- **Modular, PEP8-compliant codebase** — cleanly separated into training, inference, and utility modules

## Screenshots

| Streamlit Home | Prediction Result |
|:---:|:---:|
| <img src="https://via.placeholder.com/400x250.png?text=Streamlit+Home" width="400"/> | <img src="https://via.placeholder.com/400x250.png?text=Prediction+Result" width="400"/> |

| EDA Dashboard | Feature Importance |
|:---:|:---:|
| <img src="https://via.placeholder.com/400x250.png?text=EDA+Dashboard" width="400"/> | <img src="https://via.placeholder.com/400x250.png?text=Feature+Importance+Plot" width="400"/> |

## Model Performance

Evaluated on a held-out test set:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 81.2% | 77.4% | 69.1% | 73.0% | 0.84 |
| Random Forest | 84.5% | 80.3% | 74.2% | 77.1% | 0.88 |
| **XGBoost (Best)** | **86.8%** | **83.1%** | **78.5%** | **80.7%** | **0.91** |

> The best-performing model (XGBoost) is automatically serialized to `best_model.pkl`.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Web Interface | Streamlit |
| Serialization | Joblib |

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/Customer-Churn-Prediction-System.git
cd Customer-Churn-Prediction-System
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

**Train the pipeline**

Downloads data, preprocesses features, trains all models, evaluates them, and saves the best one to disk.
```bash
python train.py
```
Outputs: `models/best_model.pkl`, `models/feature_importance.png`, `models/roc_curve.png`

**Explore the data**
```bash
jupyter notebook notebooks/EDA.ipynb
```

**Launch the dashboard**
```bash
streamlit run app.py
```

## Key Learnings

- Orchestrating a full ML pipeline from data ingestion through deployment
- Engineering derived features that surface hidden behavioral signals
- Evaluating models with ROC-AUC and F1, not just accuracy, on an imbalanced dataset
- Building a structured, user-friendly dashboard with Streamlit
- Applying software engineering practices — modular code, docstrings, dependency management

## Future Improvements

- [ ] Hyperparameter tuning via `GridSearchCV` or `Optuna`
- [ ] Class imbalance handling with SMOTE
- [ ] SHAP-based explainability integrated directly into the Streamlit UI
- [ ] Public deployment on AWS or Heroku
