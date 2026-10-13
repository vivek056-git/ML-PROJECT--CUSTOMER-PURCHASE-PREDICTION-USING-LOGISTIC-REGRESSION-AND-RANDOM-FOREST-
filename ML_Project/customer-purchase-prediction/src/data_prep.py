"""
data_prep.py
------------
Data loading, exploration, cleaning, encoding, and splitting.

Each function has explicit type hints and a docstring explaining what it does
and what it returns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATA_FILE, ID_COL, RANDOM_STATE, TARGET, TEST_SIZE


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """Read the CSV dataset and return it as a DataFrame.

    Prints the shape so it is visible in the pipeline banner.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns intact.
    """
    df = pd.read_csv(DATA_FILE)
    print(f"  Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Explore
# ---------------------------------------------------------------------------


def explore(df: pd.DataFrame) -> dict[str, Any]:
    """Return a summary dictionary describing the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataframe (before cleaning).

    Returns
    -------
    dict
        Keys: shape, dtypes, missing_per_column, target_distribution,
        feature_target_correlation.
    """
    target_counts = df[TARGET].value_counts().sort_index()
    target_pct = (target_counts / len(df) * 100).round(2)
    target_dist = {
        int(k): {"count": int(v), "pct": float(target_pct[k])}
        for k, v in target_counts.items()
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)

    corr_series = (
        df[numeric_cols + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values(ascending=False)
    )

    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_column": df.isnull().sum().to_dict(),
        "target_distribution": target_dist,
        "feature_target_correlation": corr_series.round(4).to_dict(),
    }


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ID column, then impute missing values.

    Numeric columns: imputed with the column **median** (robust to outliers).
    Categorical columns: imputed with the column **mode** (most frequent value).

    Prints how many values were filled per column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe without the ID column.
    """
    df = df.copy()

    # Drop identifier — it carries no predictive signal
    df = df.drop(columns=[ID_COL], errors="ignore")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    filled_report: dict[str, int] = {}

    for col in numeric_cols:
        n_missing = int(df[col].isnull().sum())
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            filled_report[col] = n_missing

    for col in categorical_cols:
        n_missing = int(df[col].isnull().sum())
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            filled_report[col] = n_missing

    if filled_report:
        print("  Imputed missing values:")
        for col, n in filled_report.items():
            print(f"    {col}: {n} value(s) filled")
    else:
        print("  No missing values found.")

    return df


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features as numeric.

    - Gender: Female -> 0, Male -> 1 (binary, single column).
    - Customer_Segment: one-hot encoded with drop_first=True to avoid
      multicollinearity (drops the first dummy level).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    pd.DataFrame
        Fully numeric dataframe ready for modelling.
    """
    df = df.copy()

    # Binary encode Gender
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    # One-hot encode Customer_Segment; drop_first removes one redundant dummy
    df = pd.get_dummies(df, columns=["Customer_Segment"], drop_first=True)

    # Ensure dummy columns are integer dtype for cleanliness
    dummy_cols = [c for c in df.columns if c.startswith("Customer_Segment_")]
    df[dummy_cols] = df[dummy_cols].astype(int)

    print(f"  Encoded features. Final column count: {df.shape[1]}")
    return df


# ---------------------------------------------------------------------------
# Split and scale
# ---------------------------------------------------------------------------


def split_and_scale(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    list[str],
]:
    """Stratified train/test split followed by feature scaling.

    The scaler is **fit on the training set only** and then applied to both
    train and test sets.  Fitting on the full dataset before splitting would
    allow test-set statistics to leak into training — a form of data leakage
    that inflates evaluation metrics.

    Logistic Regression receives the scaled arrays; Random Forest receives
    the unscaled DataFrames (tree-based splits are invariant to monotonic
    feature scaling, so scaling adds nothing and is omitted deliberately).

    Parameters
    ----------
    df : pd.DataFrame
        Encoded, cleaned dataframe.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series (unscaled)
    X_train_scaled, X_test_scaled    : np.ndarray (scaled, for LR)
    scaler                            : fitted StandardScaler
    feature_names                     : list of feature column names
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    feature_names: list[str] = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # preserve class ratio in both splits
    )

    # Fit scaler on training data only — no leakage from test set
    scaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    X_test_scaled: np.ndarray = scaler.transform(X_test)

    print(
        f"  Train size: {len(X_train)} | Test size: {len(X_test)}"
        f" | Features: {len(feature_names)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
        feature_names,
    )
