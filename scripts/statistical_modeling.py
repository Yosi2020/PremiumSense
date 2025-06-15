# scripts/statistical_modeling.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import shap


def preprocess_features(df: pd.DataFrame, target_reg: str, target_clf: str=None):
    """
    Splits into X, y_regression, y_classification (if requested),
    and builds a preprocessing pipeline.
    """
    df = df.copy()
    # Define features & targets
    y_reg = df[target_reg]
    if target_clf:
        df["has_claim"] = (df["TotalClaims"] > 0).astype(int)
        y_clf = df["has_claim"]
    else:
        y_clf = None

    X = df.drop(columns=[target_reg, "has_claim"] if target_clf else [target_reg])

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","bool","category"]).columns.tolist()

    # Build transformers
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    return X, y_reg, y_clf, preprocessor

def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_regression_models(X_train, y_train, preprocessor):
    """Returns dict of fitted regression pipelines."""
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest":    RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost":         XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="rmse")
    }
    pipelines = {}
    for name, model in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
    return pipelines

def evaluate_regression(models: dict, X_test, y_test):
    """Compute RMSE and R2 for each model."""
    results = {}
    for name, pipe in models.items():
        preds = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        results[name] = {"RMSE": rmse, "R2": r2}
    return pd.DataFrame(results).T

def train_classification_models(X_train, y_train, preprocessor):
    """Returns dict of fitted classification pipelines."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":            XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)
    }
    pipelines = {}
    for name, model in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
    return pipelines

def evaluate_classification(models: dict, X_test, y_test):
    """Compute common classification metrics."""
    results = {}
    for name, pipe in models.items():
        preds = pipe.predict(X_test)
        results[name] = {
            "Accuracy":  accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall":    recall_score(y_test, preds),
            "F1":        f1_score(y_test, preds)
        }
    return pd.DataFrame(results).T

def compute_shap_importance(model_pipe, X_sample: pd.DataFrame) -> pd.DataFrame:
    # 1. Extract preprocessor and model
    preprocessor = model_pipe.named_steps['prep']
    model        = model_pipe.named_steps['model']
    
    # 2. Preprocess the sample
    X_pre = preprocessor.transform(X_sample)
    # Convert sparse to dense if needed
    if hasattr(X_pre, "toarray"):
        X_pre = X_pre.toarray().astype(float)
    
    # 3. Create a TreeExplainer and compute SHAP values
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_pre)
    # If shap_values is a list (e.g. multi-output), take the first output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # 4. Reconstruct feature names
    try:
        # scikit-learn ≥1.0
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Older versions: build manually
        # numeric feature names
        numeric_features = preprocessor.transformers_[0][2]
        # categorical features and one-hot names
        cat_transformer  = preprocessor.transformers_[1][1]
        cat_cols         = preprocessor.transformers_[1][2]
        ohe              = cat_transformer.named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        feature_names     = np.concatenate([numeric_features, cat_feature_names])
    
    # 5. Compute mean absolute SHAP per feature
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    })
    
    # 6. Sort and return
    return shap_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

def compute_risk_based_premium(pipe_clf, pipe_reg, X):
    """
    Premium = P(claim)*E[claim_amt] from models.
    Returns array of premiums.
    """
    p_claim = pipe_clf.predict_proba(X)[:,1]
    e_claim = pipe_reg.predict(X)
    premium = p_claim * e_claim
    return premium
