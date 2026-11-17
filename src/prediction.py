"""
Individual module prediction with SHAP explanation.
"""
import numpy as np
import pandas as pd
import shap


def predict_single(model, explainer, scaler, imputer, input_df, feature_names,
                   threshold=0.5):
    """
    input_df: single-row DataFrame with same columns as training X.
    Returns dict with probability, prediction, and top SHAP contributions.
    """
    X = input_df[feature_names].copy()
    X_imp = imputer.transform(X)
    X_sc = scaler.transform(X_imp)

    proba = model.predict_proba(X_sc)[0, 1]
    pred = int(proba >= threshold)

    # SHAP on scaled input
    sv = explainer.shap_values(X_sc)
    if isinstance(sv, list):
        sv = sv[1]
    if hasattr(sv, 'ndim') and sv.ndim == 3:
        sv = sv[:, :, 1]
    sv_row = sv[0]

    contribs = pd.DataFrame({
        'Feature': feature_names,
        'Value': X.iloc[0].values,
        'SHAP': sv_row
    }).sort_values('SHAP', key=np.abs, ascending=False)

    return {
        'probability': float(proba),
        'prediction': pred,
        'label': 'Defective' if pred == 1 else 'Non-Defective',
        'top_contributions': contribs.head(10).reset_index(drop=True),
    }