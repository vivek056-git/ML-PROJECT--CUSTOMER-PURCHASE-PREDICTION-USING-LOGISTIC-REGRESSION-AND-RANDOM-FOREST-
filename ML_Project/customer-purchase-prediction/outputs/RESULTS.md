# Customer Purchase Prediction -- Results

All values in this document are computed from a live pipeline run.
No numbers have been manually entered.

## Dataset Summary

| Property | Value |
|---|---|
| File | fixCustomer_Purchase_Prediction_500_Members.csv |
| Rows | 5000 |
| Columns (raw) | 13 |
| Total missing values | 45 |

## Missing Value Treatment

Numeric columns were imputed with the column **median**.
Categorical columns were imputed with the column **mode**.

| Column | Missing Count |
|---|---|
| Age | 5 |
| Gender | 5 |
| Annual_Income | 5 |
| Customer_Segment | 4 |
| Previous_Purchases | 4 |
| Website_Visits | 4 |
| Pages_Viewed | 5 |
| Time_on_Website_Min | 4 |
| Emails_Opened | 3 |
| Cart_Additions | 3 |
| Days_Since_Last_Purchase | 3 |

## Class Distribution

| Class | Label | Count | Percentage |
|---|---|---|---|
| 0 | No Purchase | 2921 | 58.4% |
| 1 | Purchase | 2079 | 41.6% |
| -- | Total | 5000 | 100.0% |

Majority-class baseline accuracy: **0.5840**

## Model Comparison

| Metric | Baseline | Logistic Regression | Random Forest |
|---|---|---|---|
| Accuracy | 0.5840 | 0.5910 | 0.5750 |
| Precision | -- | 0.5071 | 0.4901 |
| Recall | -- | 0.6010 | 0.5361 |
| F1 Score | -- | 0.5501 | 0.5121 |
| ROC-AUC | -- | 0.6311 | 0.6156 |

## Cross-Validation Results (5-fold Stratified)

| Model | Accuracy mean | Accuracy std | F1 mean | F1 std |
|---|---|---|---|---|
| Logistic Regression | 0.5880 | 0.0098 | 0.5466 | 0.0106 |
| Random Forest | 0.5813 | 0.0108 | 0.5218 | 0.0243 |

## Confusion Matrices (Test Set)

### Logistic Regression

| | Predicted No Purchase | Predicted Purchase |
|---|---|---|
| Actual No Purchase | 341 | 243 |
| Actual Purchase | 166 | 250 |

### Random Forest

| | Predicted No Purchase | Predicted Purchase |
|---|---|---|
| Actual No Purchase | 352 | 232 |
| Actual Purchase | 193 | 223 |

## Top Feature Importances (Random Forest)

| Feature | Importance |
|---|---|
| Cart_Additions | 0.2161 |
| Website_Visits | 0.1366 |
| Days_Since_Last_Purchase | 0.1336 |
| Pages_Viewed | 0.0975 |
| Emails_Opened | 0.0967 |
| Annual_Income | 0.0953 |
| Previous_Purchases | 0.0827 |
| Time_on_Website_Min | 0.0746 |
| Age | 0.0546 |
| Customer_Segment_New | 0.0061 |

## Top Logistic Regression Coefficients (standardised scale)

| Feature | Coefficient |
|---|---|
| Cart_Additions | 0.2816 |
| Website_Visits | 0.1914 |
| Days_Since_Last_Purchase | -0.1675 |
| Pages_Viewed | 0.1454 |
| Emails_Opened | 0.1421 |
| Previous_Purchases | 0.1393 |
| Annual_Income | 0.0703 |
| Customer_Segment_Regular | 0.0218 |
| Customer_Segment_New | -0.0079 |
| Time_on_Website_Min | -0.0072 |

## Verdict

The winning model by F1 score is **Logistic Regression** (F1 = 0.5501, F1 margin over runner-up = 0.0380).
Winner accuracy: 0.5910 vs majority-class baseline: 0.5840 (gap = +0.0070).

The winning model's accuracy (0.5910) is only 0.0070 (0.7 pp) above the baseline -- below the 5 pp threshold. This is consistent with weak feature-target correlations and is an honest result.

