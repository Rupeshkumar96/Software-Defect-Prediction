"""
Evaluation plots: confusion matrix, ROC, PR, threshold sensitivity.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score
)


def confusion_matrix_fig(cm, model_name='Model'):
    labels = ['Non-Defective', 'Defective']
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f'Pred: {l}' for l in labels],
        y=[f'Actual: {l}' for l in labels],
        text=cm, texttemplate='%{text}', textfont={'size': 20},
        colorscale='Blues', showscale=True
    ))
    fig.update_layout(
        title=f'Confusion Matrix — {model_name}',
        height=450, width=550
    )
    return fig


def roc_curve_fig(artifacts):
    fig = go.Figure()
    for name, a in artifacts.items():
        fig.add_trace(go.Scatter(
            x=a['fpr'], y=a['tpr'],
            mode='lines', name=f"{name} (AUC={_auc(a):.3f})"
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Random',
        line=dict(dash='dash', color='gray')
    ))
    fig.update_layout(
        title='ROC Curves — All Models',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    return fig


def _auc(a):
    # trapezoid integration
    return np.trapezoid(a['tpr'], a['fpr'])


def pr_curve_fig(artifacts):
    fig = go.Figure()
    for name, a in artifacts.items():
        fig.add_trace(go.Scatter(
            x=a['recall_curve'], y=a['precision_curve'],
            mode='lines', name=f"{name} (AP={a['avg_precision']:.3f})"
        ))
    fig.update_layout(
        title='Precision-Recall Curves',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=500
    )
    return fig


def threshold_sensitivity(model, X_test, y_test):
    """Compute metrics across thresholds."""
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            'Threshold': round(t, 2),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
        })
    return rows


def threshold_curve_fig(rows, model_name):
    import pandas as pd
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for metric in ['Precision', 'Recall', 'F1']:
        fig.add_trace(go.Scatter(
            x=df['Threshold'], y=df[metric],
            mode='lines+markers', name=metric
        ))
    fig.update_layout(
        title=f'Threshold Sensitivity — {model_name}',
        xaxis_title='Decision Threshold',
        yaxis_title='Score',
        height=500
    )
    return fig