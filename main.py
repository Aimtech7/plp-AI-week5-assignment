"""
main.py
GitHub-ready example for: Predicting 30-day hospital readmission risk using a Neural Network (MLP).
This script uses a synthetic dataset for demonstration. Replace data generation with real EHR loading.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Synthetic data generator (for demonstration only) ---
def generate_synthetic_data(n=5000, seed=42):
    rng = np.random.RandomState(seed)
    age = rng.randint(18, 95, size=n)
    gender = rng.choice(['M', 'F'], size=n, p=[0.48, 0.52])
    prior_admissions = rng.poisson(1.2, size=n)
    comorbidity_score = np.clip(rng.normal(2.5 + 0.05*(age-50), 1.0, size=n), 0, 10)
    length_of_stay = np.clip(rng.exponential(3.0, size=n).round(), 1, 30)
    lab_abnormal_flag = rng.binomial(1, 0.25, size=n)
    social_vulnerability_index = np.clip(rng.normal(0.5, 0.2, size=n), 0, 1)
    # base risk
    base = 0.03 + 0.01*prior_admissions + 0.02*(comorbidity_score/10) + 0.01*(length_of_stay/10) + 0.1*lab_abnormal_flag
    # add noise and clip to [0,1]
    readmit_prob = np.clip(base + rng.normal(0, 0.03, size=n), 0, 1)
    readmitted = rng.binomial(1, readmit_prob)
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'prior_admissions': prior_admissions,
        'comorbidity_score': comorbidity_score,
        'length_of_stay': length_of_stay,
        'lab_abnormal_flag': lab_abnormal_flag,
        'svi': social_vulnerability_index,
        'readmitted_30d': readmitted
    })
    # Introduce some missingness
    for col in ['comorbidity_score', 'svi']:
        df.loc[rng.choice(n, size=int(0.05*n), replace=False), col] = np.nan
    return df

def build_preprocessing_pipeline(num_cols, cat_cols):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return preprocessor

def train_evaluate(df):
    # split features and target
    X = df.drop(columns=['readmitted_30d'])
    y = df['readmitted_30d']
    # categorical / numerical columns
    cat_cols = ['gender']
    num_cols = [c for c in X.columns if c not in cat_cols]
    # train/val/test split stratified
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)
    # 0.1765 * 0.85 ≈ 0.15 -> so final splits 70/15/15
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    # MLP classifier pipeline
    clf = Pipeline([
        ('pre', preprocessor),
        ('mlp', MLPClassifier(max_iter=300, random_state=42))
    ])
    # Hyperparameter grid (small example)
    param_grid = {
        'mlp__hidden_layer_sizes': [(32,), (64,), (64,32)],
        'mlp__alpha': [1e-4, 1e-3],
        'mlp__learning_rate_init': [1e-3, 5e-4]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(clf, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)
    # Evaluate on validation
    y_val_pred = best.predict(X_val)
    y_val_proba = best.predict_proba(X_val)[:,1]
    print("Validation AUC:", roc_auc_score(y_val, y_val_proba))
    print("Validation Recall:", recall_score(y_val, y_val_pred))
    print("Validation Precision:", precision_score(y_val, y_val_pred))
    # Final test evaluation
    y_test_pred = best.predict(X_test)
    y_test_proba = best.predict_proba(X_test)[:,1]
    print("Test AUC:", roc_auc_score(y_test, y_test_proba))
    print("Test Recall:", recall_score(y_test, y_test_pred))
    print("Test Precision:", precision_score(y_test, y_test_pred))
    print("Confusion Matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))
    # Save model artifact
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(best, 'artifacts/model.joblib')
    print("Model saved to artifacts/model.joblib")
    # Save confusion matrix figure
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('artifacts/confusion_matrix.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    df = generate_synthetic_data(n=3000)
    print("Dataset shape:", df.shape)
    train_evaluate(df)
