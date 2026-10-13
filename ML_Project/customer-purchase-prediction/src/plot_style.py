"""
plot_style.py
-------------
Global matplotlib style settings aligned with the project design specification.
Import apply_style() before creating any figure so all plots share a consistent
visual identity.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

from src.config import PALETTE


def apply_style() -> None:
    """Apply the project-wide matplotlib style.

    Call once at the start of any plotting session.  The settings mirror the
    Streamlit app's CSS so figures look native when embedded in the interface.
    """
    mpl.rcParams.update(
        {
            # Figure
            "figure.facecolor": PALETTE["canvas"],
            "figure.edgecolor": PALETTE["canvas"],
            "figure.dpi": 100,
            "savefig.facecolor": PALETTE["canvas"],
            "savefig.edgecolor": PALETTE["canvas"],
            # Axes
            "axes.facecolor": PALETTE["canvas"],
            "axes.edgecolor": PALETTE["hairline"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": PALETTE["ink"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.titlecolor": PALETTE["ink"],
            # Grid: faint horizontal lines only
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": PALETTE["hairline"],
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            # Ticks
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            # Legend
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": PALETTE["ink"],
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "system-ui",
                "-apple-system",
                "Segoe UI",
                "Arial",
                "Helvetica",
            ],
            "font.size": 10,
            "text.color": PALETTE["ink"],
            # Lines
            "lines.linewidth": 1.8,
            "patch.edgecolor": PALETTE["canvas"],
        }
    )


def get_two_colors() -> list[str]:
    """Return the two primary accent colours used throughout the project."""
    return [PALETTE["accent"], PALETTE["accent_secondary"]]
