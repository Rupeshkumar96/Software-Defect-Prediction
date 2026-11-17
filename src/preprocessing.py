"""
Preprocessing: cleaning, splitting, scaling.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def clean_features(df, target_col):
    """Remove non-numeric features and handle constant columns."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Keep only numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[numeric_cols].copy()

    # Drop constant columns
    constant_cols = [c for c in X_num.columns if X_num[c].nunique() <= 1]
    X_num = X_num.drop(columns=constant_cols)

    return X_num, y, numeric_cols, constant_cols


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor():
    """Return imputer + scaler pipeline pieces."""
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    return imputer, scaler


def fit_transform_train(X_train, X_test):
    """Fit imputer+scaler on train, transform both."""
    imputer, scaler = build_preprocessor()
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)
    return X_train_sc, X_test_sc, imputer, scaler