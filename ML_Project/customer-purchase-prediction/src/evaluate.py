"""
evaluate.py
-----------
Model evaluation: metrics, terminal report, plots, RESULTS.md, metrics.json.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import METRICS_FILE, PALETTE, PLOTS_DIR, RESULTS_FILE
from src.plot_style import apply_style, get_two_colors


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, float]:
    """Compute classification metrics on the test set.

    Parameters
    ----------
    model      : fitted sklearn estimator
    X_test     : test features (scaled for LR, unscaled for RF)
    y_test     : true labels
    model_name : label for display

    Returns
    -------
    dict with accuracy, precision, recall, f1, roc_auc
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def baseline_accuracy(y_test: pd.Series) -> float:
    """Majority-class baseline: predict the most frequent class for every sample."""
    majority = y_test.mode()[0]
    return float((y_test == majority).mean())


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------


def print_report(
    lr_metrics: dict[str, float],
    rf_metrics: dict[str, float],
    baseline_acc: float,
    lr_model: Any,
    rf_model: Any,
    X_test_scaled: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    lr_cv: dict[str, float],
    rf_cv: dict[str, float],
) -> None:
    """Print a formatted comparison report to the terminal."""

    sep = "-" * 72

    print(f"\n{sep}")
    print("  MODEL COMPARISON REPORT")
    print(sep)

    # Comparison table header
    col_w = 18
    print(
        f"\n  {'Metric':<20} {'Baseline':>{col_w}} {'Logistic Reg':>{col_w}} {'Random Forest':>{col_w}}"
    )
    print(f"  {'-'*20} {'-'*col_w} {'-'*col_w} {'-'*col_w}")

    metrics_rows = [
        ("Accuracy", baseline_acc, lr_metrics["accuracy"], rf_metrics["accuracy"]),
        ("Precision", None, lr_metrics["precision"], rf_metrics["precision"]),
        ("Recall", None, lr_metrics["recall"], rf_metrics["recall"]),
        ("F1 Score", None, lr_metrics["f1"], rf_metrics["f1"]),
        ("ROC-AUC", None, lr_metrics["roc_auc"], rf_metrics["roc_auc"]),
    ]

    for name, base, lr_v, rf_v in metrics_rows:
        base_str = f"{base:.4f}" if base is not None else "  --  "
        print(
            f"  {name:<20} {base_str:>{col_w}} {lr_v:>{col_w}.4f} {rf_v:>{col_w}.4f}"
        )

    # Cross-validation
    print(f"\n  Cross-Validation (5-fold Stratified)")
    print(f"  {'-'*55}")
    print(
        f"  {'':20} {'Accuracy (mean+/-std)':>25} {'F1 (mean+/-std)':>22}"
    )
    for label, cv in [("Logistic Regression", lr_cv), ("Random Forest", rf_cv)]:
        acc_str = f"{cv['acc_mean']:.4f} +/- {cv['acc_std']:.4f}"
        f1_str = f"{cv['f1_mean']:.4f} +/- {cv['f1_std']:.4f}"
        print(f"  {label:<20} {acc_str:>25} {f1_str:>22}")

    # Confusion matrices
    print(f"\n  Confusion Matrices (test set)")
    print(f"  {'-'*55}")
    for name, mdl, X in [
        ("Logistic Regression", lr_model, X_test_scaled),
        ("Random Forest", rf_model, X_test),
    ]:
        cm = confusion_matrix(y_test, mdl.predict(X))
        print(f"\n  {name}")
        print(f"  Predicted:      No Purchase   Purchase")
        print(f"  Actual No Buy:  {cm[0,0]:>11}   {cm[0,1]:>8}")
        print(f"  Actual Buy:     {cm[1,0]:>11}   {cm[1,1]:>8}")

    # Classification reports
    print(f"\n  Classification Reports")
    print(f"  {'-'*55}")
    for name, mdl, X in [
        ("Logistic Regression", lr_model, X_test_scaled),
        ("Random Forest", rf_model, X_test),
    ]:
        y_pred = mdl.predict(X)
        print(f"\n  {name}")
        report = classification_report(y_test, y_pred, target_names=["No Purchase", "Purchase"])
        for line in report.splitlines():
            print(f"    {line}")

    # Verdict
    print(f"\n  {sep}")
    print("  VERDICT")
    print(f"  {sep}")

    lr_f1 = lr_metrics["f1"]
    rf_f1 = rf_metrics["f1"]
    f1_margin = abs(rf_f1 - lr_f1)
    winner = "Random Forest" if rf_f1 >= lr_f1 else "Logistic Regression"
    winner_f1 = max(rf_f1, lr_f1)
    winner_acc = lr_metrics["accuracy"] if winner == "Logistic Regression" else rf_metrics["accuracy"]
    acc_gap = winner_acc - baseline_acc

    print(f"  Winner by F1 score: {winner} (F1 = {winner_f1:.4f})")
    print(f"  F1 margin over runner-up: {f1_margin:.4f}")
    print(f"  Winner accuracy vs majority-class baseline: {winner_acc:.4f} vs {baseline_acc:.4f}")

    if acc_gap >= 0.05:
        print(
            f"  The winning model's accuracy exceeds the majority-class baseline by"
            f" {acc_gap:.4f} ({acc_gap*100:.1f} pp)."
        )
    elif acc_gap >= 0:
        print(
            f"  Note: the winning model's accuracy ({winner_acc:.4f}) is only"
            f" {acc_gap:.4f} ({acc_gap*100:.1f} pp) above the baseline -- below the 5 pp threshold."
            f" This is consistent with weak feature-target correlations and is an honest result."
        )
    else:
        print(
            f"  Note: the winning model's accuracy ({winner_acc:.4f}) is"
            f" {abs(acc_gap):.4f} ({abs(acc_gap)*100:.1f} pp) below the majority-class baseline."
            f" class_weight='balanced' trades raw accuracy for better minority-class recall;"
            f" F1 score and ROC-AUC are more informative metrics for this imbalanced task."
        )

    print(f"\n{sep}\n")



# ---------------------------------------------------------------------------
# Metrics JSON
# ---------------------------------------------------------------------------


def save_metrics_json(
    lr_metrics: dict[str, float],
    rf_metrics: dict[str, float],
    baseline_acc: float,
    lr_cv: dict[str, float],
    rf_cv: dict[str, float],
    explore_summary: dict[str, Any],
) -> None:
    """Write all computed metrics to artifacts/metrics.json."""
    payload = {
        "baseline_accuracy": baseline_acc,
        "logistic_regression": {**lr_metrics, "cross_validation": lr_cv},
        "random_forest": {**rf_metrics, "cross_validation": rf_cv},
        "dataset": {
            "shape": list(explore_summary["shape"]),
            "target_distribution": explore_summary["target_distribution"],
            "missing_per_column": explore_summary["missing_per_column"],
        },
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Metrics saved: {METRICS_FILE}")


# ---------------------------------------------------------------------------
# RESULTS.md
# ---------------------------------------------------------------------------


def write_results_md(
    lr_metrics: dict[str, float],
    rf_metrics: dict[str, float],
    baseline_acc: float,
    lr_cv: dict[str, float],
    rf_cv: dict[str, float],
    explore_summary: dict[str, Any],
    rf_model: Any,
    lr_model: Any,
    feature_names: list[str],
    X_test_scaled: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Generate outputs/RESULTS.md with all computed values interpolated."""

    dist = explore_summary["target_distribution"]
    n0 = dist[0]["count"]
    n1 = dist[1]["count"]
    p0 = dist[0]["pct"]
    p1 = dist[1]["pct"]
    total_rows = n0 + n1

    missing_dict = explore_summary["missing_per_column"]
    total_missing = sum(missing_dict.values())
    missing_cols = {k: v for k, v in missing_dict.items() if v > 0}

    shape = explore_summary["shape"]

    # Confusion matrices
    cm_lr = confusion_matrix(y_test, lr_model.predict(X_test_scaled))
    cm_rf = confusion_matrix(y_test, rf_model.predict(X_test))

    # Feature importances (RF)
    rf_importances = pd.Series(
        rf_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    # LR coefficients
    lr_coefs = pd.Series(
        lr_model.coef_[0], index=feature_names
    ).reindex(pd.Series(np.abs(lr_model.coef_[0]), index=feature_names).sort_values(ascending=False).index)

    # Verdict
    lr_f1 = lr_metrics["f1"]
    rf_f1 = rf_metrics["f1"]
    winner = "Random Forest" if rf_f1 >= lr_f1 else "Logistic Regression"
    winner_f1 = max(rf_f1, lr_f1)
    margin = abs(rf_f1 - lr_f1)

    lines: list[str] = []

    lines.append("# Customer Purchase Prediction -- Results")
    lines.append("")
    lines.append("All values in this document are computed from a live pipeline run.")
    lines.append("No numbers have been manually entered.")
    lines.append("")

    # Dataset summary
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(f"| Property | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| File | fixCustomer_Purchase_Prediction_500_Members.csv |")
    lines.append(f"| Rows | {shape[0]} |")
    lines.append(f"| Columns (raw) | {shape[1]} |")
    lines.append(f"| Total missing values | {total_missing} |")
    lines.append("")

    # Missing values
    lines.append("## Missing Value Treatment")
    lines.append("")
    lines.append("Numeric columns were imputed with the column **median**.")
    lines.append("Categorical columns were imputed with the column **mode**.")
    lines.append("")
    if missing_cols:
        lines.append("| Column | Missing Count |")
        lines.append("|---|---|")
        for col, cnt in missing_cols.items():
            lines.append(f"| {col} | {cnt} |")
    else:
        lines.append("No missing values detected.")
    lines.append("")

    # Class distribution
    lines.append("## Class Distribution")
    lines.append("")
    lines.append(f"| Class | Label | Count | Percentage |")
    lines.append(f"|---|---|---|---|")
    lines.append(f"| 0 | No Purchase | {n0} | {p0:.1f}% |")
    lines.append(f"| 1 | Purchase | {n1} | {p1:.1f}% |")
    lines.append(f"| -- | Total | {total_rows} | 100.0% |")
    lines.append(f"")
    lines.append(f"Majority-class baseline accuracy: **{baseline_acc:.4f}**")
    lines.append("")

    # Metrics table
    lines.append("## Model Comparison")
    lines.append("")
    lines.append(
        "| Metric | Baseline | Logistic Regression | Random Forest |"
    )
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Accuracy | {baseline_acc:.4f} | {lr_metrics['accuracy']:.4f} | {rf_metrics['accuracy']:.4f} |"
    )
    lines.append(
        f"| Precision | -- | {lr_metrics['precision']:.4f} | {rf_metrics['precision']:.4f} |"
    )
    lines.append(
        f"| Recall | -- | {lr_metrics['recall']:.4f} | {rf_metrics['recall']:.4f} |"
    )
    lines.append(
        f"| F1 Score | -- | {lr_metrics['f1']:.4f} | {rf_metrics['f1']:.4f} |"
    )
    lines.append(
        f"| ROC-AUC | -- | {lr_metrics['roc_auc']:.4f} | {rf_metrics['roc_auc']:.4f} |"
    )
    lines.append("")

    # Cross-validation
    lines.append("## Cross-Validation Results (5-fold Stratified)")
    lines.append("")
    lines.append("| Model | Accuracy mean | Accuracy std | F1 mean | F1 std |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Logistic Regression | {lr_cv['acc_mean']:.4f} | {lr_cv['acc_std']:.4f} | {lr_cv['f1_mean']:.4f} | {lr_cv['f1_std']:.4f} |"
    )
    lines.append(
        f"| Random Forest | {rf_cv['acc_mean']:.4f} | {rf_cv['acc_std']:.4f} | {rf_cv['f1_mean']:.4f} | {rf_cv['f1_std']:.4f} |"
    )
    lines.append("")

    # Confusion matrices
    lines.append("## Confusion Matrices (Test Set)")
    lines.append("")
    lines.append("### Logistic Regression")
    lines.append("")
    lines.append("| | Predicted No Purchase | Predicted Purchase |")
    lines.append("|---|---|---|")
    lines.append(f"| Actual No Purchase | {cm_lr[0,0]} | {cm_lr[0,1]} |")
    lines.append(f"| Actual Purchase | {cm_lr[1,0]} | {cm_lr[1,1]} |")
    lines.append("")
    lines.append("### Random Forest")
    lines.append("")
    lines.append("| | Predicted No Purchase | Predicted Purchase |")
    lines.append("|---|---|---|")
    lines.append(f"| Actual No Purchase | {cm_rf[0,0]} | {cm_rf[0,1]} |")
    lines.append(f"| Actual Purchase | {cm_rf[1,0]} | {cm_rf[1,1]} |")
    lines.append("")

    # Feature importances
    lines.append("## Top Feature Importances (Random Forest)")
    lines.append("")
    lines.append("| Feature | Importance |")
    lines.append("|---|---|")
    for feat, imp in rf_importances.head(10).items():
        lines.append(f"| {feat} | {imp:.4f} |")
    lines.append("")

    lines.append("## Top Logistic Regression Coefficients (standardised scale)")
    lines.append("")
    lines.append("| Feature | Coefficient |")
    lines.append("|---|---|")
    for feat, coef in lr_coefs.head(10).items():
        lines.append(f"| {feat} | {coef:.4f} |")
    lines.append("")

    # Verdict
    winner_acc = lr_metrics["accuracy"] if winner == "Logistic Regression" else rf_metrics["accuracy"]
    acc_gap = winner_acc - baseline_acc

    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"The winning model by F1 score is **{winner}** "
        f"(F1 = {winner_f1:.4f}, F1 margin over runner-up = {margin:.4f})."
    )
    lines.append(
        f"Winner accuracy: {winner_acc:.4f} vs majority-class baseline: {baseline_acc:.4f} "
        f"(gap = {acc_gap:+.4f})."
    )
    if acc_gap >= 0.05:
        lines.append(
            f"\nThe winning model's accuracy exceeds the majority-class baseline by "
            f"{acc_gap:.4f} ({acc_gap*100:.1f} percentage points)."
        )
    elif acc_gap >= 0:
        lines.append(
            f"\nThe winning model's accuracy ({winner_acc:.4f}) is only "
            f"{acc_gap:.4f} ({acc_gap*100:.1f} pp) above the baseline -- below the 5 pp threshold. "
            f"This is consistent with weak feature-target correlations and is an honest result."
        )
    else:
        lines.append(
            f"\nThe winning model's accuracy ({winner_acc:.4f}) is "
            f"{abs(acc_gap):.4f} ({abs(acc_gap)*100:.1f} pp) below the majority-class baseline. "
            f"class_weight='balanced' trades raw accuracy for better minority-class recall; "
            f"F1 and ROC-AUC are more informative metrics for this imbalanced task."
        )
    lines.append("")


    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  RESULTS.md saved: {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_target_distribution(
    df_raw: pd.DataFrame,
    y_test: pd.Series | None = None,
) -> None:
    """01_target_distribution.png -- bar chart of class counts with % labels."""
    apply_style()
    colors = [PALETTE["negative"], PALETTE["positive"]]

    counts = df_raw["Purchase"].value_counts().sort_index()
    total = counts.sum()
    labels = ["No Purchase", "Purchase"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, width=0.5)

    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{count}  ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["ink"],
        )

    ax.set_ylabel("Count")
    ax.set_title("Target Class Distribution")
    ax.spines["bottom"].set_color(PALETTE["hairline"])
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    _save(fig, "01_target_distribution.png")


def plot_correlation_heatmap(df_encoded: pd.DataFrame) -> None:
    """02_correlation_heatmap.png -- annotated numeric correlation matrix."""
    apply_style()

    numeric_df = df_encoded.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        ax=ax,
        cmap=sns.diverging_palette(145, 20, s=60, l=40, as_cmap=True),
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7, "color": PALETTE["ink"]},
        linewidths=0.4,
        linecolor=PALETTE["hairline"],
        square=True,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title("Feature Correlation Matrix")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    _save(fig, "02_correlation_heatmap.png")


def plot_confusion_matrices(
    lr_model: Any,
    rf_model: Any,
    X_test_scaled: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """03_confusion_matrices.png -- both models side by side."""
    apply_style()
    colors = [PALETTE["negative"], PALETTE["positive"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pairs = [
        ("Logistic Regression", lr_model, X_test_scaled),
        ("Random Forest", rf_model, X_test),
    ]

    for ax, (name, mdl, X) in zip(axes, pairs):
        cm = confusion_matrix(y_test, mdl.predict(X))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Purchase", "Purchase"],
        )
        disp.plot(
            ax=ax,
            colorbar=False,
            cmap="Greens",
        )
        # Override colours to match palette
        for text in ax.texts:
            text.set_color(PALETTE["ink"])
            text.set_fontsize(11)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle("Confusion Matrices (Test Set)", fontsize=12, fontweight="600")
    fig.tight_layout()
    _save(fig, "03_confusion_matrices.png")


def plot_roc_curves(
    lr_model: Any,
    rf_model: Any,
    X_test_scaled: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """04_roc_curves.png -- both ROC curves with AUC in legend."""
    apply_style()
    two_colors = get_two_colors()

    fig, ax = plt.subplots(figsize=(6, 5))

    for (name, mdl, X), color in zip(
        [
            ("Logistic Regression", lr_model, X_test_scaled),
            ("Random Forest", rf_model, X_test),
        ],
        two_colors,
    ):
        y_prob = mdl.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, label=f"{name}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], color=PALETTE["hairline"], linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    _save(fig, "04_roc_curves.png")


def plot_feature_importance(
    rf_model: Any,
    lr_model: Any,
    feature_names: list[str],
) -> None:
    """05_feature_importance.png -- RF importances and LR coefficients."""
    apply_style()
    two_colors = get_two_colors()

    # RF importances
    rf_imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values()
    # LR coefficients sorted by absolute magnitude
    lr_coef = pd.Series(lr_model.coef_[0], index=feature_names)
    lr_coef_sorted = lr_coef.reindex(lr_coef.abs().sort_values().index)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # RF panel
    axes[0].barh(rf_imp.index, rf_imp.values, color=two_colors[0], height=0.6)
    axes[0].set_xlabel("Mean Decrease in Impurity")
    axes[0].set_title("Random Forest Feature Importance")
    axes[0].tick_params(axis="y", labelsize=8)

    # LR panel
    bar_colors = [
        two_colors[0] if v >= 0 else two_colors[1] for v in lr_coef_sorted.values
    ]
    axes[1].barh(
        lr_coef_sorted.index,
        lr_coef_sorted.values,
        color=bar_colors,
        height=0.6,
    )
    axes[1].axvline(0, color=PALETTE["hairline"], linewidth=0.8)
    axes[1].set_xlabel("Coefficient (standardised feature scale)")
    axes[1].set_title("Logistic Regression Coefficients")
    axes[1].tick_params(axis="y", labelsize=8)

    fig.suptitle("Feature Contributions", fontsize=12, fontweight="600")
    fig.tight_layout()
    _save(fig, "05_feature_importance.png")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, filename: str) -> None:
    """Save a figure to the plots directory."""
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")
