"""
main.py
-------
Orchestrator: runs the full pipeline end-to-end.

Usage:
    python main.py
"""

from __future__ import annotations

import time

from src.config import ARTIFACTS_DIR, PLOTS_DIR, ensure_dirs
from src.data_prep import clean, encode, explore, load_data, split_and_scale
from src.evaluate import (
    baseline_accuracy,
    compute_metrics,
    plot_confusion_matrices,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_roc_curves,
    plot_target_distribution,
    print_report,
    save_metrics_json,
    write_results_md,
)
from src.train import (
    cross_validate_model,
    save_artifacts,
    train_logistic,
    train_random_forest,
)


def banner(title: str) -> None:
    """Print a section banner to make pipeline progress easy to follow."""
    width = 72
    print(f"\n{'='*width}")
    print(f"  {title.upper()}")
    print(f"{'='*width}")


def main() -> None:
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    banner("setup")
    ensure_dirs()
    print("  Output directories confirmed.")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    banner("load data")
    df_raw = load_data()

    # ------------------------------------------------------------------
    # Explore
    # ------------------------------------------------------------------
    banner("explore")
    summary = explore(df_raw)
    dist = summary["target_distribution"]
    print(f"  Target distribution:")
    print(f"    No Purchase (0): {dist[0]['count']}  ({dist[0]['pct']:.1f}%)")
    print(f"    Purchase (1):    {dist[1]['count']}  ({dist[1]['pct']:.1f}%)")
    print(f"  Total missing values: {sum(summary['missing_per_column'].values())}")
    print(f"  Top feature-target correlations:")
    for feat, corr in list(summary["feature_target_correlation"].items())[:5]:
        print(f"    {feat:<30} {corr:+.4f}")

    # ------------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------------
    banner("clean")
    df_clean = clean(df_raw)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    banner("encode")
    df_encoded = encode(df_clean)

    # ------------------------------------------------------------------
    # Split and scale
    # ------------------------------------------------------------------
    banner("split and scale")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
        feature_names,
    ) = split_and_scale(df_encoded)

    # ------------------------------------------------------------------
    # Train Logistic Regression
    # ------------------------------------------------------------------
    banner("train logistic regression")
    lr_model = train_logistic(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # Train Random Forest
    # ------------------------------------------------------------------
    banner("train random forest (grid search)")
    rf_model = train_random_forest(X_train, y_train)

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    banner("cross-validation")
    lr_cv = cross_validate_model(lr_model, X_train_scaled, y_train, "Logistic Regression")
    rf_cv = cross_validate_model(rf_model, X_train, y_train, "Random Forest")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    banner("evaluate")
    lr_metrics = compute_metrics(lr_model, X_test_scaled, y_test, "Logistic Regression")
    rf_metrics = compute_metrics(rf_model, X_test, y_test, "Random Forest")
    base_acc = baseline_accuracy(y_test)
    print(f"  Baseline accuracy (majority class): {base_acc:.4f}")
    print(f"  Logistic Regression -- Accuracy: {lr_metrics['accuracy']:.4f}  F1: {lr_metrics['f1']:.4f}  AUC: {lr_metrics['roc_auc']:.4f}")
    print(f"  Random Forest       -- Accuracy: {rf_metrics['accuracy']:.4f}  F1: {rf_metrics['f1']:.4f}  AUC: {rf_metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    banner("save artifacts")
    save_artifacts(lr_model, rf_model, scaler, feature_names)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    banner("generate plots")
    plot_target_distribution(df_raw)
    plot_correlation_heatmap(df_encoded)
    plot_confusion_matrices(lr_model, rf_model, X_test_scaled, X_test, y_test)
    plot_roc_curves(lr_model, rf_model, X_test_scaled, X_test, y_test)
    plot_feature_importance(rf_model, lr_model, feature_names)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    banner("write outputs")
    save_metrics_json(lr_metrics, rf_metrics, base_acc, lr_cv, rf_cv, summary)
    write_results_md(
        lr_metrics,
        rf_metrics,
        base_acc,
        lr_cv,
        rf_cv,
        summary,
        rf_model,
        lr_model,
        feature_names,
        X_test_scaled,
        X_test,
        y_test,
    )

    # ------------------------------------------------------------------
    # Terminal report
    # ------------------------------------------------------------------
    print_report(
        lr_metrics,
        rf_metrics,
        base_acc,
        lr_model,
        rf_model,
        X_test_scaled,
        X_test,
        y_test,
        lr_cv,
        rf_cv,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    banner("pipeline complete")
    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"  Artifacts:     {ARTIFACTS_DIR}")
    print(f"  Plots:         {PLOTS_DIR}")
    print(f"  Results:       outputs/RESULTS.md")
    print(f"\n  To launch the interactive app run:")
    print(f"    streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
