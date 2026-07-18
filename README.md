# Customer Churn Prediction System

This project is an end-to-end Machine Learning pipeline that predicts whether a customer will churn (cancel their service) based on demographics, account information, and usage patterns. It uses the IBM Telco Customer Churn dataset and includes a fully interactive web application built with Streamlit.

## 🚀 Features

- **Automated Data Pipeline**: Scripts for downloading, cleaning, and preprocessing the Telco dataset.
- **Feature Engineering**: Derives new features such as `Total_Services` and `Monthly_to_Total_Ratio` to boost model performance.
- **Model Evaluation**: Trains and compares Logistic Regression, Decision Tree, Random Forest, and XGBoost, automatically selecting the best one based on ROC-AUC score.
- **Exploratory Data Analysis**: A Jupyter Notebook (`EDA.ipynb`) detailing dataset properties, missing values, correlation heatmaps, and target class distributions.
- **Interactive UI**: A Streamlit web application where users can input customer details and instantly receive a churn probability.
- **Modular Code**: Clean, well-documented python scripts organized into training, inference, and utilities.

## 🛠 Technologies Used

- **Python 3**
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Visualization**: Matplotlib, Seaborn
- **Web Interface**: Streamlit
- **Model Deployment**: Joblib

## 📁 Project Structure

```text
customer-churn-prediction/
├── data/
│   └── Telco-Customer-Churn.csv       # Downloaded automatically during training
├── notebooks/
│   └── EDA.ipynb                      # Exploratory Data Analysis notebook
├── models/
│   ├── best_model.pkl                 # Trained ML model (XGBoost/LogReg/etc.)
│   ├── feature_importance.png         # Bar plot of top features
│   └── *.pkl                          # Encoders and scalers
├── app.py                             # Streamlit Web App
├── train.py                           # Training and evaluation script
├── predict.py                         # Prediction interface
├── utils.py                           # Helper functions
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

## ⚙️ Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vanshikarajput20/Customer-Churn-Prediction-System.git
   cd Customer-Churn-Prediction-System
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Usage

### 1. Train the Model
To download the dataset, preprocess it, evaluate models, and save the best one, simply run:
```bash
python train.py
```
*This will create the `models/` and `data/` directories and output the model metrics.*

### 2. View Exploratory Data Analysis
You can open the Jupyter Notebook to see the EDA:
```bash
jupyter notebook notebooks/EDA.ipynb
```

### 3. Run the Web App
To launch the interactive prediction UI:
```bash
streamlit run app.py
```

## 📊 Feature Importance
The trained model relies heavily on features like `Tenure`, `Contract Type`, and `Total_Services`. You can find the generated feature importance plot at `models/feature_importance.png` after running the training script.

## 🔮 Future Improvements

- Incorporate hyperparameter tuning using `GridSearchCV` or `Optuna`.
- Add advanced handling of class imbalance (e.g., SMOTE).
- Deploy the Streamlit app to AWS, Heroku, or Streamlit Community Cloud.
- Add SHAP values for model explainability on the web interface.