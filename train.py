import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:
    print(f"Warning: Could not load XGBoost ({e}).")
    HAS_XGB = False

from utils import load_data, preprocess_data, prepare_modeling_data

def evaluate_model(y_true, y_pred, y_prob, model_name):
    """
    Computes key performance metrics for a model and prints them.
    Returns the computed metrics in a dictionary.
    """
    print(f"\n--- {model_name} ---")
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }
    
    for k, v in metrics.items():
        print(f"{k.capitalize():<12}: {v:.4f}")
        
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}\n")
    
    return metrics, cm

def plot_roc_curves(roc_data):
    """
    Plots the ROC curves for all trained models.
    """
    plt.figure(figsize=(10, 8))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if not os.path.exists("models"):
        os.makedirs("models")
    plt.savefig("models/roc_curve.png")
    plt.close()
    print("Saved ROC curve plot to 'models/roc_curve.png'")

def train_models():
    """
    Main pipeline function to load data, preprocess, tune hyperparameters, 
    evaluate them, and save the best performing model.
    """
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    df_clean = preprocess_data(df)
    
    print("Preparing modeling data (encoding, scaling, splitting)...")
    X_train, X_test, y_train, y_test, feature_columns = prepare_modeling_data(df_clean)
    
    # Define models and their hyperparameter grids
    model_configs = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [5, 10, None]
            }
        }
    }
    
    if HAS_XGB:
        model_configs["XGBoost"] = {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1]
            }
        }
    
    results = {}
    roc_data = {}
    best_overall_model = None
    best_overall_roc_auc = 0
    best_overall_name = ""
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nStarting Hyperparameter Tuning and Cross-Validation...")
    for name, config in model_configs.items():
        print(f"\nTuning {name}...")
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best params for {name}: {grid_search.best_params_}")
        print(f"Cross-Validated ROC-AUC: {grid_search.best_score_:.4f}")
        
        # Evaluate on the hold-out test set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        metrics, cm = evaluate_model(y_test, y_pred, y_prob, name)
        results[name] = metrics
        
        # Store ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, metrics['roc_auc'])
        
        # Track the best overall model based on test set ROC-AUC
        if metrics['roc_auc'] > best_overall_roc_auc:
            best_overall_roc_auc = metrics['roc_auc']
            best_overall_model = best_model
            best_overall_name = name
            
    print(f"\n=======================================================")
    print(f"Best Model Selected: {best_overall_name} with Test ROC-AUC of {best_overall_roc_auc:.4f}")
    print(f"=======================================================\n")
    
    # Save ROC curves
    plot_roc_curves(roc_data)
    
    # Save the best model
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(best_overall_model, "models/best_model.pkl")
    print("Saved best model to 'models/best_model.pkl'")
    
    # Feature Importance Plot
    if hasattr(best_overall_model, 'feature_importances_'):
        importances = best_overall_model.feature_importances_
    elif hasattr(best_overall_model, 'coef_'):
        importances = np.abs(best_overall_model.coef_[0])
    else:
        importances = np.zeros(len(feature_columns))
        
    feat_imp = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feat_imp.head(15), palette="viridis", legend=False)
    plt.title(f'Top 15 Feature Importances ({best_overall_name})')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    plt.close()
    print("Saved feature importance plot to 'models/feature_importance.png'")

if __name__ == "__main__":
    train_models()
