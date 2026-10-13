"""
train.py
--------
Model training, hyperparameter search, cross-validation, and artifact saving.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src.config import ARTIFACTS, RANDOM_STATE


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------


def train_logistic(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
) -> LogisticRegression:
    """Train a Logistic Regression classifier.

    class_weight='balanced' compensates for the 2:1 class imbalance by
    up-weighting the minority class (Purchase=1) during training.
    max_iter=1000 prevents convergence warnings on this dataset.

    Parameters
    ----------
    X_train_scaled : np.ndarray
        Standardised training features.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    LogisticRegression
        Fitted model.
    """
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train_scaled, y_train)
    print("  Logistic Regression trained.")
    return lr


# ---------------------------------------------------------------------------
# Random Forest with grid search
# ---------------------------------------------------------------------------


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train a Random Forest classifier with a small hyperparameter grid search.

    A deliberately conservative grid is used to keep runtime reasonable while
    still comparing meaningful structural variations.  F1 is the search
    objective because accuracy is misleading under class imbalance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Unscaled training features.  Tree-based models are invariant to
        monotonic feature scaling, so StandardScaler is intentionally omitted.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    RandomForestClassifier
        Best estimator selected by GridSearchCV.
    """
    base_rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [1, 3, 5],
    }

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"  Random Forest best parameters: {best_params}")
    print(f"  Grid search best CV F1: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_model(
    model: LogisticRegression | RandomForestClassifier,
    X: np.ndarray | pd.DataFrame,
    y: pd.Series,
    label: str,
) -> dict[str, float]:
    """Evaluate a model with 5-fold stratified cross-validation.

    Cross-validation guards against lucky or unlucky test splits by averaging
    performance across multiple held-out folds.

    Parameters
    ----------
    model : fitted sklearn estimator
    X : features (scaled array for LR, unscaled DataFrame for RF)
    y : labels
    label : human-readable model name for printing

    Returns
    -------
    dict with keys: acc_mean, acc_std, f1_mean, f1_std
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1"],
        n_jobs=-1,
    )

    result = {
        "acc_mean": float(np.mean(scores["test_accuracy"])),
        "acc_std": float(np.std(scores["test_accuracy"])),
        "f1_mean": float(np.mean(scores["test_f1"])),
        "f1_std": float(np.std(scores["test_f1"])),
    }

    print(
        f"  {label} CV -- "
        f"Accuracy: {result['acc_mean']:.4f} +/- {result['acc_std']:.4f}  "
        f"F1: {result['f1_mean']:.4f} +/- {result['f1_std']:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------


def save_artifacts(
    lr_model: LogisticRegression,
    rf_model: RandomForestClassifier,
    scaler: object,
    feature_names: list[str],
) -> None:
    """Persist models, scaler, and feature names to disk with joblib.

    Parameters
    ----------
    lr_model : fitted LogisticRegression
    rf_model : fitted RandomForestClassifier
    scaler   : fitted StandardScaler
    feature_names : list of feature column names in training order
    """
    joblib.dump(lr_model, ARTIFACTS["lr_model"])
    joblib.dump(rf_model, ARTIFACTS["rf_model"])
    joblib.dump(scaler, ARTIFACTS["scaler"])
    joblib.dump(feature_names, ARTIFACTS["feature_names"])

    print("  Artifacts saved:")
    for key, path in ARTIFACTS.items():
        print(f"    {key}: {path}")
