"""
Exploratory Data Analysis helpers.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def missing_values_df(df):
    mv = df.isnull().sum()
    pct = (mv / len(df) * 100).round(2)
    return pd.DataFrame({
        'Column': mv.index,
        'Missing Count': mv.values,
        'Missing %': pct.values
    })


def descriptive_stats(df, numeric_cols):
    return df[numeric_cols].describe().T.round(3)


def correlation_matrix(df, numeric_cols):
    return df[numeric_cols].corr().round(3)


def correlation_heatmap(corr_df):
    fig = px.imshow(
        corr_df,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title='Feature Correlation Heatmap'
    )
    fig.update_layout(height=700)
    return fig


def class_distribution_chart(df, target_col):
    counts = df[target_col].value_counts().sort_index()
    labels = ['Non-Defective (0)', 'Defective (1)']
    colors = ['#2ecc71', '#e74c3c']
    fig = go.Figure(data=[
        go.Bar(x=labels, y=counts.values, marker_color=colors, text=counts.values,
               textposition='auto')
    ])
    fig.update_layout(
        title='Class Distribution',
        xaxis_title='Class',
        yaxis_title='Count',
        height=400
    )
    return fig


def feature_distribution_box(df, numeric_cols, top_n=12):
    """Box plot of top features by variance."""
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top_cols = variances.head(top_n).index.tolist()
    fig = px.box(df, y=top_cols, title=f'Distribution of Top {top_n} Features')
    fig.update_layout(height=600)
    return fig