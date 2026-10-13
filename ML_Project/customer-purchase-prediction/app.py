"""
app.py
------
Streamlit application: Dataset exploration, model comparison, and live prediction.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import ARTIFACTS, ARTIFACTS_DIR, DATA_FILE, PLOTS_DIR, RESULTS_FILE

# ---------------------------------------------------------------------------
# Page configuration -- must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Customer Purchase Prediction",
    page_icon=None,
)

# ---------------------------------------------------------------------------
# Design injection
# ---------------------------------------------------------------------------
FONT_IMPORT = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"

CUSTOM_CSS = f"""
<style>
@import url('{FONT_IMPORT}');

/* ---- Reset and base ---- */
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    color: #16161A;
    background-color: #F7F6F3;
}}

/* ---- Main background ---- */
.stApp {{
    background-color: #F7F6F3;
}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background-color: #FFFFFF;
    border-right: 1px solid #E2E0DA;
}}

/* ---- Headers ---- */
h1 {{
    font-size: 26px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #16161A;
    border-bottom: 1px solid #E2E0DA;
    padding-bottom: 12px;
    margin-bottom: 24px;
}}
h2 {{
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #16161A;
    margin-top: 32px;
    margin-bottom: 12px;
}}
h3 {{
    font-size: 15px;
    font-weight: 600;
    color: #16161A;
    margin-top: 20px;
    margin-bottom: 8px;
}}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {{
    background-color: #FFFFFF;
    border: 1px solid #E2E0DA;
    border-radius: 0px;
    padding: 16px 20px;
}}
[data-testid="stMetricLabel"] {{
    font-size: 11px;
    font-weight: 500;
    color: #6B6B73;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 22px;
    color: #1F5C4E;
}}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 1px solid #E2E0DA;
    background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 0;
    padding: 10px 24px;
    font-size: 13px;
    font-weight: 500;
    color: #6B6B73;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: #1F5C4E;
    border-bottom: 2px solid #1F5C4E;
    background: transparent;
}}

/* ---- Dataframes ---- */
.stDataFrame {{
    font-family: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 12px;
    border: 1px solid #E2E0DA;
    border-radius: 0;
}}

/* ---- Buttons ---- */
.stButton > button {{
    background-color: #1F5C4E;
    color: #FFFFFF;
    border: none;
    border-radius: 2px;
    padding: 10px 28px;
    font-size: 13px;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: background-color 0.15s ease;
}}
.stButton > button:hover {{
    background-color: #174a3e;
}}

/* ---- Inputs ---- */
.stNumberInput input, .stSelectbox select {{
    font-family: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 13px;
    border-radius: 0;
    border: 1px solid #E2E0DA;
}}

/* ---- Text ---- */
p, li {{
    font-size: 14px;
    line-height: 1.65;
    color: #16161A;
}}
small, .caption {{
    font-size: 12px;
    color: #6B6B73;
}}

/* ---- Prediction result cards ---- */
.pred-card {{
    background: #FFFFFF;
    border: 1px solid #E2E0DA;
    border-top: 3px solid #1F5C4E;
    padding: 20px 24px;
    margin-bottom: 16px;
}}
.pred-card.secondary {{
    border-top-color: #B4552D;
}}
.pred-model-name {{
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6B6B73;
    margin-bottom: 8px;
}}
.pred-class {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 500;
    color: #16161A;
    margin-bottom: 4px;
}}
.pred-prob {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #6B6B73;
}}

/* ---- Divider ---- */
hr {{
    border: none;
    border-top: 1px solid #E2E0DA;
    margin: 24px 0;
}}

/* ---- Remove Streamlit default padding on image ---- */
.stImage img {{
    border: 1px solid #E2E0DA;
}}

/* ---- Section label ---- */
.section-label {{
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6B6B73;
    border-bottom: 1px solid #E2E0DA;
    padding-bottom: 6px;
    margin-bottom: 16px;
    margin-top: 32px;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Check artifacts exist before doing anything else
# ---------------------------------------------------------------------------
_artifacts_missing = not ARTIFACTS_DIR.exists() or not any(ARTIFACTS_DIR.iterdir())
if _artifacts_missing:
    st.markdown(
        "Artifacts directory is empty or missing. "
        "Run `python main.py` from the project root to train the models first."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Load resources (cached so they are read only once)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_models() -> tuple:
    lr = joblib.load(ARTIFACTS["lr_model"])
    rf = joblib.load(ARTIFACTS["rf_model"])
    scaler = joblib.load(ARTIFACTS["scaler"])
    feature_names = joblib.load(ARTIFACTS["feature_names"])
    return lr, rf, scaler, feature_names


@st.cache_data
def load_dataframe() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_metrics() -> dict:
    with open(ARTIFACTS["lr_model"].parent / "metrics.json", "r") as f:
        return json.load(f)


lr_model, rf_model, scaler, feature_names = load_models()
df_raw = load_dataframe()
metrics = load_metrics()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("# Customer Purchase Prediction")
st.markdown(
    "<p style='color:#6B6B73;font-size:14px;margin-top:-16px;'>Logistic Regression vs Random Forest &nbsp;&middot;&nbsp; 5 000 customers &nbsp;&middot;&nbsp; Binary classification</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_dataset, tab_models, tab_predict = st.tabs(
    ["Dataset", "Model Comparison", "Predict"]
)

# ===========================================================================
# TAB 1 -- Dataset
# ===========================================================================
with tab_dataset:
    st.markdown("## Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df_raw.shape[0]}")
    c2.metric("Columns", f"{df_raw.shape[1]}")
    total_missing = int(df_raw.isnull().sum().sum())
    c3.metric("Missing Values", f"{total_missing}")
    n_purchase = int(df_raw["Purchase"].sum())
    c4.metric("Purchase rate", f"{n_purchase / len(df_raw) * 100:.1f}%")

    st.markdown('<div class="section-label">Missing Value Treatment</div>', unsafe_allow_html=True)
    missing_series = df_raw.isnull().sum()
    missing_cols = missing_series[missing_series > 0]
    if len(missing_cols) > 0:
        missing_df = pd.DataFrame(
            {
                "Column": missing_cols.index,
                "Missing Count": missing_cols.values,
                "Treatment": [
                    "Median (numeric)" if df_raw[c].dtype != object else "Mode (categorical)"
                    for c in missing_cols.index
                ],
            }
        )
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.markdown("No missing values detected in the dataset.")

    st.markdown('<div class="section-label">Class Distribution</div>', unsafe_allow_html=True)
    dist = metrics["dataset"]["target_distribution"]
    c1, c2, c3 = st.columns(3)
    c1.metric("No Purchase (0)", f"{dist['0']['count']}", f"{dist['0']['pct']:.1f}%")
    c2.metric("Purchase (1)", f"{dist['1']['count']}", f"{dist['1']['pct']:.1f}%")
    c3.metric("Baseline Accuracy", f"{metrics['baseline_accuracy']:.4f}")

    st.markdown('<div class="section-label">Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_raw, use_container_width=True, height=280)

    st.markdown('<div class="section-label">Visualisations</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        target_dist_path = PLOTS_DIR / "01_target_distribution.png"
        if target_dist_path.exists():
            st.image(str(target_dist_path), use_container_width=True)
    with v2:
        heatmap_path = PLOTS_DIR / "02_correlation_heatmap.png"
        if heatmap_path.exists():
            st.image(str(heatmap_path), use_container_width=True)


# ===========================================================================
# TAB 2 -- Model Comparison
# ===========================================================================
with tab_models:
    st.markdown("## Model Comparison")

    # Summary metrics
    lr_m = metrics["logistic_regression"]
    rf_m = metrics["random_forest"]
    baseline = metrics["baseline_accuracy"]

    metric_rows = [
        ("Accuracy", baseline, lr_m["accuracy"], rf_m["accuracy"]),
        ("Precision", None, lr_m["precision"], rf_m["precision"]),
        ("Recall", None, lr_m["recall"], rf_m["recall"]),
        ("F1 Score", None, lr_m["f1"], rf_m["f1"]),
        ("ROC-AUC", None, lr_m["roc_auc"], rf_m["roc_auc"]),
    ]

    table_data = []
    for name, base, lr_v, rf_v in metric_rows:
        base_str = f"{base:.4f}" if base is not None else "--"
        table_data.append(
            {
                "Metric": name,
                "Baseline": base_str,
                "Logistic Regression": f"{lr_v:.4f}",
                "Random Forest": f"{rf_v:.4f}",
                "Winner": "LR" if lr_v > rf_v else ("RF" if rf_v > lr_v else "Tie"),
            }
        )

    cmp_df = pd.DataFrame(table_data)

    def highlight_winner(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        winner = row["Winner"]
        col_map = {"LR": "Logistic Regression", "RF": "Random Forest"}
        if winner in col_map:
            col_name = col_map[winner]
            col_idx = list(row.index).index(col_name)
            styles[col_idx] = f"color: #1F5C4E; font-weight: 600;"
        return styles

    styled = cmp_df.style.apply(highlight_winner, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown(
        f"<p style='font-size:13px;color:#6B6B73;'>Majority-class baseline accuracy: "
        f"<span style='font-family:monospace;color:#16161A;'>{baseline:.4f}</span></p>",
        unsafe_allow_html=True,
    )

    # Cross-validation
    st.markdown('<div class="section-label">Cross-Validation (5-fold Stratified)</div>', unsafe_allow_html=True)
    lr_cv = lr_m["cross_validation"]
    rf_cv = rf_m["cross_validation"]
    cv_data = {
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy mean": [f"{lr_cv['acc_mean']:.4f}", f"{rf_cv['acc_mean']:.4f}"],
        "Accuracy std": [f"{lr_cv['acc_std']:.4f}", f"{rf_cv['acc_std']:.4f}"],
        "F1 mean": [f"{lr_cv['f1_mean']:.4f}", f"{rf_cv['f1_mean']:.4f}"],
        "F1 std": [f"{lr_cv['f1_std']:.4f}", f"{rf_cv['f1_std']:.4f}"],
    }
    st.dataframe(pd.DataFrame(cv_data), use_container_width=True, hide_index=True)

    # Plots
    st.markdown('<div class="section-label">Confusion Matrices</div>', unsafe_allow_html=True)
    cm_path = PLOTS_DIR / "03_confusion_matrices.png"
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)

    roc_col, fi_col = st.columns(2)
    with roc_col:
        st.markdown('<div class="section-label">ROC Curves</div>', unsafe_allow_html=True)
        roc_path = PLOTS_DIR / "04_roc_curves.png"
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
    with fi_col:
        st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
        fi_path = PLOTS_DIR / "05_feature_importance.png"
        if fi_path.exists():
            st.image(str(fi_path), use_container_width=True)

    # Verdict
    winner_name = "Random Forest" if rf_m["f1"] >= lr_m["f1"] else "Logistic Regression"
    winner_f1 = max(rf_m["f1"], lr_m["f1"])
    margin = abs(rf_m["f1"] - lr_m["f1"])
    winner_acc = lr_m["accuracy"] if winner_name == "Logistic Regression" else rf_m["accuracy"]
    acc_gap = winner_acc - baseline

    st.markdown('<div class="section-label">Verdict</div>', unsafe_allow_html=True)
    st.markdown(
        f"**{winner_name}** wins by F1 score "
        f"(F1 = {winner_f1:.4f}, margin = {margin:.4f} over the runner-up). "
        f"Winner accuracy vs majority-class baseline: {winner_acc:.4f} vs {baseline:.4f} "
        f"({acc_gap:+.4f})."
        + (
            " The accuracy gain is below the 5 pp threshold, consistent with the weak feature-target "
            "correlations in this dataset."
            if acc_gap < 0.05
            else ""
        )
    )


# ===========================================================================
# TAB 3 -- Predict
# ===========================================================================
with tab_predict:
    st.markdown("## Predict Purchase Probability")
    st.markdown(
        "Enter a customer profile below. Both models will return a predicted class "
        "and probability of purchase."
    )

    with st.form("prediction_form"):
        st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)

        # Dynamically derive limits from the dataset so they always match the loaded data
        def _max(col: str) -> float:
            return float(df_raw[col].max())

        def _min(col: str) -> float:
            return float(df_raw[col].min())

        def _mean_int(col: str) -> float:
            return float(round(df_raw[col].mean()))

        def _mean_f(col: str) -> float:
            return round(float(df_raw[col].mean()), 1)

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age",
                min_value=0.0,
                value=_mean_int("Age"), step=1.0,
            )
            gender = st.selectbox("Gender", ["Female", "Male"])
            annual_income = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=float(round(_mean_f("Annual_Income"), -3)), step=1000.0,
            )
            customer_segment = st.selectbox(
                "Customer Segment", ["Regular", "Loyal", "New"]
            )


        with col2:
            previous_purchases = st.number_input(
                "Previous Purchases",
                min_value=0.0,
                value=_mean_int("Previous_Purchases"), step=1.0,
            )
            website_visits = st.number_input(
                "Website Visits",
                min_value=0.0,
                value=_mean_int("Website_Visits"), step=1.0,
            )
            pages_viewed = st.number_input(
                "Pages Viewed",
                min_value=0.0,
                value=_mean_int("Pages_Viewed"), step=1.0,
            )
            time_on_website = st.number_input(
                "Time on Website (min)",
                min_value=0.0,
                value=_mean_f("Time_on_Website_Min"), step=0.5,
            )

        with col3:
            emails_opened = st.number_input(
                "Emails Opened",
                min_value=0.0,
                value=_mean_int("Emails_Opened"), step=1.0,
            )
            cart_additions = st.number_input(
                "Cart Additions",
                min_value=0.0,
                value=_mean_int("Cart_Additions"), step=1.0,
            )
            days_since_last = st.number_input(
                "Days Since Last Purchase",
                min_value=0.0,
                value=_mean_int("Days_Since_Last_Purchase"), step=1.0,
            )

        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        # Build raw input dict
        gender_enc = 0 if gender == "Female" else 1

        # One-hot encode Customer_Segment -- must match training columns
        seg_loyal = 1 if customer_segment == "Loyal" else 0
        seg_regular = 1 if customer_segment == "Regular" else 0
        # drop_first=True dropped the first alphabetically; check which was dropped
        # Training used pd.get_dummies with drop_first=True on ['Loyal','New','Regular']
        # Alphabetical order: Loyal, New, Regular -> 'Loyal' is dropped
        # So dummies are: Customer_Segment_New, Customer_Segment_Regular
        seg_new = 1 if customer_segment == "New" else 0

        raw_input = {
            "Age": age,
            "Gender": gender_enc,
            "Annual_Income": annual_income,
            "Previous_Purchases": previous_purchases,
            "Website_Visits": website_visits,
            "Pages_Viewed": pages_viewed,
            "Time_on_Website_Min": time_on_website,
            "Emails_Opened": emails_opened,
            "Cart_Additions": cart_additions,
            "Days_Since_Last_Purchase": days_since_last,
            "Customer_Segment_New": seg_new,
            "Customer_Segment_Regular": seg_regular,
        }

        # Build DataFrame and reindex against training feature order
        # This is the critical step: column order must exactly match training.
        input_df = pd.DataFrame([raw_input])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Scale for Logistic Regression
        input_scaled = scaler.transform(input_df.values)

        # Predictions
        lr_class = int(lr_model.predict(input_scaled)[0])
        lr_prob = float(lr_model.predict_proba(input_scaled)[0][1])

        rf_class = int(rf_model.predict(input_df)[0])
        rf_prob = float(rf_model.predict_proba(input_df)[0][1])

        lr_label = "Purchase" if lr_class == 1 else "No Purchase"
        rf_label = "Purchase" if rf_class == 1 else "No Purchase"

        st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)

        p1, p2 = st.columns(2)

        with p1:
            st.markdown(
                f"""
                <div class="pred-card">
                    <div class="pred-model-name">Logistic Regression</div>
                    <div class="pred-class">{lr_label}</div>
                    <div class="pred-prob">Purchase probability: {lr_prob:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with p2:
            st.markdown(
                f"""
                <div class="pred-card secondary">
                    <div class="pred-model-name">Random Forest</div>
                    <div class="pred-class">{rf_label}</div>
                    <div class="pred-prob">Purchase probability: {rf_prob:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if lr_class == rf_class:
            agreement_text = "Both models agree on the predicted class."
        else:
            agreement_text = (
                "The two models disagree. Logistic Regression predicts "
                f"{lr_label.lower()} while Random Forest predicts {rf_label.lower()}."
            )

        st.markdown(
            f"<p style='font-size:13px;color:#6B6B73;margin-top:8px;'>{agreement_text}</p>",
            unsafe_allow_html=True,
        )


