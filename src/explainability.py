"""
SHAP explanations.
"""
import numpy as np
import shap
import plotly.graph_objects as go
import pandas as pd


def build_explainer(model, X_train, feature_names):
    """Create the correct SHAP explainer for the model type."""
    # Tree-based models
    if hasattr(model, 'estimators_') or model.__class__.__name__ in (
        'RandomForestClassifier', 'GradientBoostingClassifier'
    ):
        return shap.TreeExplainer(model)
    # Linear / other
    return shap.LinearExplainer(model, X_train)


def shap_values_for_class(explainer, X):
    """Extract SHAP values for class=1 (defective)."""
    sv = explainer.shap_values(X)
    # TreeExplainer on classifiers returns list [class0, class1]
    if isinstance(sv, list):
        return sv[1]
    # Some versions return 3D array
    if hasattr(sv, 'ndim') and sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def global_importance_df(shap_values, feature_names):
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs
    }).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
    return df


def feature_importance_bar(shap_values, feature_names, top_n=15):
    df = global_importance_df(shap_values, feature_names).head(top_n)
    fig = go.Figure(go.Bar(
        x=df['Mean |SHAP|'],
        y=df['Feature'],
        orientation='h',
        marker_color='#3498db'
    ))
    fig.update_layout(
        title=f'Top {top_n} Features by Mean |SHAP|',
        xaxis_title='Mean |SHAP value|',
        yaxis=dict(autorange='reversed'),
        height=600
    )
    return fig


def beeswarm_fig(shap_values, X_df):
    """Plotly version of SHAP beeswarm (global impact)."""
    import pandas as pd
    n_samples, n_features = shap_values.shape
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]

    fig = go.Figure()
    for rank, feat_idx in enumerate(order):
        vals = shap_values[:, feat_idx]
        feat_vals = X_df.iloc[:, feat_idx].values
        # Normalize feature values for coloring
        fmin, fmax = np.nanmin(feat_vals), np.nanmax(feat_vals)
        if fmax > fmin:
            color_vals = (feat_vals - fmin) / (fmax - fmin)
        else:
            color_vals = np.zeros_like(feat_vals, dtype=float)
        y_jitter = rank + (np.random.rand(n_samples) - 0.5) * 0.4

        fig.add_trace(go.Scatter(
            x=vals, y=y_jitter, mode='markers',
            marker=dict(
                color=color_vals, colorscale='RdBu',
                size=6, showscale=(rank == 0),
                colorbar=dict(title='Feature value') if rank == 0 else None
            ),
            name=X_df.columns[feat_idx],
            text=X_df.columns[feat_idx],
            hovertemplate='Feature: %{text}<br>SHAP: %{x:.3f}<extra></extra>'
        ))

    fig.update_layout(
        title='SHAP Beeswarm — Global Feature Impact',
        xaxis_title='SHAP value (impact on defect probability)',
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(order))),
            ticktext=[X_df.columns[i] for i in order]
        ),
        height=700, showlegend=False
    )
    return fig


def waterfall_fig_for_sample(explainer, X_df, sample_idx, feature_names):
    """Waterfall plot for one module."""
    sv = shap_values_for_class(explainer, X_df.iloc[[sample_idx]])
    sv_row = sv[0]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = base[1] if len(np.atleast_1d(base)) > 1 else base[0]

    feat_vals = X_df.iloc[sample_idx].values
    contributions = sv_row
    order = np.argsort(np.abs(contributions))[::-1][:12]

    labels = [f"{feature_names[i]} = {feat_vals[i]:.2f}" for i in order]
    vals = contributions[order]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker_color=['#e74c3c' if v > 0 else '#3498db' for v in vals]
    ))
    fig.update_layout(
        title=f'SHAP Waterfall — Module #{sample_idx} (base={base:.3f})',
        xaxis_title='SHAP contribution',
        yaxis=dict(autorange='reversed'),
        height=550
    )
    return fig, base, float(sv_row.sum())


def dependence_fig(shap_values, X_df, feature_idx):
    feat_name = X_df.columns[feature_idx]
    feat_vals = X_df.iloc[:, feature_idx].values
    sv = shap_values[:, feature_idx]

    fig = go.Figure(go.Scatter(
        x=feat_vals, y=sv, mode='markers',
        marker=dict(color=sv, colorscale='RdBu', size=7)
    ))
    fig.update_layout(
        title=f'SHAP Dependence — {feat_name}',
        xaxis_title=f'{feat_name} value',
        yaxis_title=f'SHAP value for {feat_name}',
        height=500
    )
    return fig