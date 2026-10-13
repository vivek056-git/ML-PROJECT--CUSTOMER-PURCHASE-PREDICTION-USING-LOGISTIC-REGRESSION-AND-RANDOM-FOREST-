"""
config.py
---------
Central configuration: paths, constants, colour palette, and directory setup.
All paths are resolved relative to the project root so scripts run from any
working directory.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root is one level above this file (src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

DATA_FILE = DATA_DIR / "Customer_Purchase_Prediction_5000_Members_30_Blank_15_NaN_30_Duplicates.csv"
METRICS_FILE = ARTIFACTS_DIR / "metrics.json"
RESULTS_FILE = OUTPUTS_DIR / "RESULTS.md"

ARTIFACTS = {
    "lr_model": ARTIFACTS_DIR / "logistic_regression.joblib",
    "rf_model": ARTIFACTS_DIR / "random_forest.joblib",
    "scaler": ARTIFACTS_DIR / "scaler.joblib",
    "feature_names": ARTIFACTS_DIR / "feature_names.joblib",
}

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
TARGET: str = "Purchase"
ID_COL: str = "Customer_ID"

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

PALETTE = {
    "canvas": "#F7F6F3",
    "surface": "#FFFFFF",
    "ink": "#16161A",
    "muted": "#6B6B73",
    "hairline": "#E2E0DA",
    "accent": "#1F5C4E",
    "accent_secondary": "#B4552D",
    "positive": "#1F5C4E",
    "negative": "#9A9A93",
}

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def ensure_dirs() -> None:
    """Create runtime output directories if they do not already exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
