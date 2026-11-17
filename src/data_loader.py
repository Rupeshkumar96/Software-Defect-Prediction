"""
Data loading, validation, and target column detection.
"""
import pandas as pd
import numpy as np


def load_dataset(file_path_or_buffer):
    """Load CSV dataset and perform basic normalization."""
    df = pd.read_csv(file_path_or_buffer)
    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    return df


def detect_target_column(df):
    """
    Try to auto-detect the target column.
    Common candidates: 'defects', 'defect', 'target', 'label', 'class', 'bug'.
    """
    candidates = ['defects', 'defect', 'target', 'label', 'class', 'bug', 'is_defective']
    for c in candidates:
        for col in df.columns:
            if col.lower() == c:
                return col
    return None


def encode_target(series):
    """
    Encode target column to 0/1.
    Handles: bool, 'true'/'false', 'yes'/'no', 0/1, 'Y'/'N'.
    Returns (encoded_series, mapping_description).
    """
    s = series.copy()

    # Boolean
    if s.dtype == bool:
        return s.astype(int), "bool -> {False: 0, True: 1}"

    # Numeric
    if pd.api.types.is_numeric_dtype(s):
        unique = sorted(s.dropna().unique().tolist())
        if set(unique).issubset({0, 1}):
            return s.astype(int), f"numeric -> {{0: 0, 1: 1}}"
        # If more than 2 unique numeric values, treat >0 as defective
        return (s > 0).astype(int), f"numeric -> {{0: 0, >0: 1}}"

    # String
    s_str = s.astype(str).str.strip().str.lower()
    pos = {'true', 'yes', 'y', '1', 'defective', 'bug', 'buggy'}
    neg = {'false', 'no', 'n', '0', 'non-defective', 'clean', 'ok'}

    if set(s_str.unique()).issubset(pos | neg):
        encoded = s_str.isin(pos).astype(int)
        return encoded, "string -> {positive: 1, negative: 0}"

    # Fallback: factorize
    codes, uniques = pd.factorize(s)
    # Assume higher frequency class is non-defective (0)
    if len(uniques) == 2:
        counts = s.value_counts()
        majority = counts.idxmax()
        encoded = (s != majority).astype(int)
        return encoded, f"factorized -> majority='{majority}'=0, other=1"

    raise ValueError(
        f"Could not automatically encode target column. Unique values: {s.unique()[:10]}"
    )


def dataset_summary(df, target_col):
    """Return a dict of high-level dataset statistics."""
    return {
        'total_records': len(df),
        'total_features': df.shape[1],
        'feature_columns': [c for c in df.columns if c != target_col],
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': int(df.duplicated().sum()),
        'defective_count': int((df[target_col] == 1).sum()),
        'non_defective_count': int((df[target_col] == 0).sum()),
    }