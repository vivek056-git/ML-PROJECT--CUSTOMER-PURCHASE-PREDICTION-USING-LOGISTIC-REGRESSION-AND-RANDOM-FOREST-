# Customer Purchase Prediction

Binary classification comparing Logistic Regression and Random Forest on a
synthetic dataset of 500 retail customers. The goal is an honest head-to-head
comparison of the two algorithms, not a competition for the highest possible
accuracy score.

---

## File Structure

```
customer-purchase-prediction/
├── data/
│   └── fixCustomer_Purchase_Prediction_500_Members.csv
├── src/
│   ├── __init__.py          Package marker
│   ├── config.py            Central constants, paths, palette, ensure_dirs()
│   ├── plot_style.py        Global matplotlib style (matches app CSS)
│   ├── data_prep.py         Load, explore, clean, encode, split, scale
│   ├── train.py             Fit models, grid search, cross-validate, save
│   └── evaluate.py          Metrics, terminal report, plots, RESULTS.md, metrics.json
├── artifacts/               Created at runtime by main.py
├── outputs/
│   └── plots/               Created at runtime by main.py
├── main.py                  Pipeline orchestrator
├── app.py                   Streamlit application
├── requirements.txt         Pinned dependencies
└── README.md                This file
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

Place `fixCustomer_Purchase_Prediction_500_Members.csv` in the `data/` folder,
then run:

```bash
python main.py
```

The script will:

1. Load and describe the dataset.
2. Impute missing values.
3. Encode categorical features.
4. Perform a stratified 80/20 train/test split.
5. Train Logistic Regression and Random Forest (with GridSearchCV).
6. Run 5-fold cross-validation for both models.
7. Evaluate on the held-out test set.
8. Save trained models, the scaler, and the feature-name list to `artifacts/`.
9. Generate five plots to `outputs/plots/`.
10. Write `outputs/RESULTS.md` and `artifacts/metrics.json`.
11. Print a comparison table and verdict to the terminal.

Total runtime is typically under 60 seconds.

---

## Running the Application

After the pipeline has completed:

```bash
streamlit run app.py
```

The app opens three tabs:

- **Dataset** -- shape, missing values, class balance, data preview, plots.
- **Model Comparison** -- metrics table, cross-validation, confusion matrices,
  ROC curves, feature importances, verdict.
- **Predict** -- enter a customer profile and receive predictions from both
  models side by side.

---

## Module Descriptions

| Module | Purpose |
|---|---|
| `src/config.py` | Single source of truth for paths, constants (`RANDOM_STATE`, `TEST_SIZE`, `TARGET`), the colour palette, and directory creation. |
| `src/plot_style.py` | Sets matplotlib rcParams to match the app's CSS so embedded figures look native. |
| `src/data_prep.py` | `load_data`, `explore`, `clean`, `encode`, `split_and_scale` -- each a pure function with docstrings and type hints. |
| `src/train.py` | `train_logistic`, `train_random_forest` (with GridSearchCV), `cross_validate_model`, `save_artifacts`. |
| `src/evaluate.py` | `compute_metrics`, `baseline_accuracy`, `print_report`, `save_metrics_json`, `write_results_md`, five plot functions. |
| `main.py` | Calls the above in order with section banners; nothing else. |
| `app.py` | Streamlit UI; loads cached artifacts, never retrains. |

---

## Methodology

### Missing Value Imputation

Twenty-eight values are missing across both numeric and categorical columns.
Numeric columns are imputed with the column **median** because it is robust to
outliers and preserves the marginal distribution better than the mean when data
are skewed. Categorical columns are imputed with the column **mode** (most
frequent value) as the only statistically coherent single-value fill for
nominal data.

### Leakage-Free Scaling

`StandardScaler` is fitted exclusively on the **training** set.  The same
fitted scaler is then used to transform the test set.  Fitting on the full
dataset before splitting would allow test-set statistics (mean and standard
deviation) to influence the training process, artificially inflating
evaluation metrics -- a form of data leakage.

### Class Imbalance

The target is imbalanced at approximately 2:1 (334 negatives, 166 positives).
Both models are trained with `class_weight="balanced"`, which instructs
scikit-learn to automatically up-weight the minority class during loss
computation.  This prevents the classifier from ignoring the minority class
to maximise raw accuracy.

### Why Random Forest Receives Unscaled Features

Decision-tree-based models, including Random Forest, determine splits by
comparing feature values to thresholds.  Monotonic transformations such as
standard scaling do not change the relative ordering of values within a
feature, so they cannot change which threshold is optimal or how the tree
splits.  Passing unscaled data to Random Forest is therefore equivalent to
passing scaled data, and skipping the step avoids unnecessary computation and
keeps the code explicit about what each model actually receives.

### Accuracy Expectations

Feature-target correlations are weak (strongest is Cart_Additions at +0.18).
A realistic test-set accuracy for both models is 68-78 percent.  The
majority-class baseline is 66.8 percent.  Results in that range are correct
and expected; they reflect the information content of the features, not a
model or implementation problem.

---

For computed results, see `outputs/RESULTS.md` after running the pipeline.
