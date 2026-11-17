"""
Model training and comparison.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)


def get_models():
    """Return dict of model name -> estimator."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42
        ),
    }


def train_all_models(X_train, y_train):
    models = get_models()
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_models(trained_models, X_test, y_test):
    """Return comparison DataFrame and per-model prediction artifacts."""
    rows = []
    artifacts = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        rows.append({
            'Model': name,
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4),
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        prec, rec, thr = precision_recall_curve(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        artifacts[name] = {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'cm': cm,
            'fpr': fpr, 'tpr': tpr,
            'precision_curve': prec, 'recall_curve': rec,
            'threshold_curve': thr,
            'avg_precision': average_precision_score(y_test, y_proba),
        }

    comparison_df = pd.DataFrame(rows)
    return comparison_df, artifacts


def pick_best_model(comparison_df):
    """Pick best model by F1 (balanced metric for imbalanced data)."""
    return comparison_df.sort_values('F1', ascending=False).iloc[0]['Model']