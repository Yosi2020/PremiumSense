import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    y_reg = df[target_reg]
    if target_clf:
        df["has_claim"] = (df["TotalClaims"] > 0).astype(int)
        y_clf = df["has_claim"]
    else:
        y_clf = None

    X = df.drop(columns=[target_reg, "has_claim"] if target_clf else [target_reg])

    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","bool","category"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
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
        "XGBoost":         XGBRegressor(
                              n_estimators=100,
                              random_state=42,
                              use_label_encoder=False,
                              eval_metric="rmse"
                          )
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
    """Compute RMSE and R² for each regression model."""
    results = {}
    for name, pipe in models.items():
        preds = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        results[name] = {"RMSE": rmse, "R2": r2}
    return pd.DataFrame(results).T

def train_classification_models(X_train, y_train, preprocessor):
    """
    Returns dict of fitted classification pipelines.
    Uses class_weight or scale_pos_weight to handle imbalance.
    """
    # compute imbalance ratio
    neg, pos = np.bincount(y_train)
    scale_pos = neg / pos if pos > 0 else 1.0

    models = {
        "LogisticRegression": LogisticRegression(
                                  class_weight='balanced',
                                  max_iter=1000
                              ),
        "RandomForest": RandomForestClassifier(
                             n_estimators=100,
                             random_state=42,
                             class_weight='balanced'
                         ),
        "XGBoost": XGBClassifier(
                       n_estimators=100,
                       use_label_encoder=False,
                       eval_metric="logloss",
                       scale_pos_weight=scale_pos,
                       random_state=42
                   )
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

def evaluate_classification(models: dict, X_test, y_test, threshold: float = 0.5):
    """
    Compute classification metrics with optional threshold on predict_proba.
    Uses zero_division=0 to avoid undefined metrics.
    """
    results = {}
    for name, pipe in models.items():
        # if model supports probabilities, apply threshold
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_test)[:, 1]
            preds = (probs > threshold).astype(int)
        else:
            preds = pipe.predict(X_test)

        results[name] = {
            "Accuracy":  accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall":    recall_score(y_test, preds, zero_division=0),
            "F1":        f1_score(y_test, preds, zero_division=0)
        }
    return pd.DataFrame(results).T

def compute_shap_importance(model_pipe, X_sample: pd.DataFrame) -> pd.DataFrame:
    # unchanged from your original
    preprocessor = model_pipe.named_steps['prep']
    model        = model_pipe.named_steps['model']
    X_pre = preprocessor.transform(X_sample)
    if hasattr(X_pre, "toarray"):
        X_pre = X_pre.toarray().astype(float)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_pre)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        numeric_feats = preprocessor.transformers_[0][2]
        cat_trans     = preprocessor.transformers_[1][1]
        cat_cols      = preprocessor.transformers_[1][2]
        ohe           = cat_trans.named_steps['onehot']
        cat_names     = ohe.get_feature_names_out(cat_cols)
        feature_names = np.concatenate([numeric_feats, cat_names])
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    })
    return shap_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

def compute_risk_based_premium(pipe_clf, pipe_reg, X):
    """
    Premium = P(claim)*E[claim_amt] from models.
    """
    p_claim = pipe_clf.predict_proba(X)[:,1]
    e_claim = pipe_reg.predict(X)
    return p_claim * e_claim
