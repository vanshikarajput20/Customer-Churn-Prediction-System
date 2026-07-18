import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:
    print(f"Warning: Could not load XGBoost ({e}). Falling back to GradientBoostingClassifier.")
    HAS_XGB = False
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_data, preprocess_data, prepare_modeling_data

def evaluate_model(y_true, y_pred, y_prob, model_name):
    print(f"--- {model_name} ---")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}\n")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc}

def train_models():
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    df_clean = preprocess_data(df)
    
    print("Preparing modeling data (encoding, scaling, splitting)...")
    X_train, X_test, y_train, y_test, feature_columns = prepare_modeling_data(df_clean)
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42),
    }
    
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        models["XGBoost (Fallback GBC)"] = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    results = {}
    best_model = None
    best_roc_auc = 0
    best_model_name = ""
    
    print("Training models...\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_prob, name)
        results[name] = metrics
        
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_model = model
            best_model_name = name
            
    print(f"Best Model Selected: {best_model_name} with ROC-AUC of {best_roc_auc:.4f}")
    
    # Save the best model
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(best_model, "models/best_model.pkl")
    print("Saved best model to 'models/best_model.pkl'")
    
    # Feature Importance Plot
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.zeros(len(feature_columns))
        
    feat_imp = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp.head(15))
    plt.title(f'Top 15 Feature Importances ({best_model_name})')
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    print("Saved feature importance plot to 'models/feature_importance.png'")

if __name__ == "__main__":
    train_models()
