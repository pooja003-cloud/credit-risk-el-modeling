"""Central configuration: paths, modelling constants, and documented risk assumptions.

Every assumption that is *not* observed in the data lives here so that it can be
reviewed, challenged and changed in one place. This mirrors how model
documentation is reviewed in a model-risk-management setting.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
DOCS = ROOT / "docs"

RAW_FILE = DATA_RAW / "default_of_credit_card_clients.csv"

for _p in (DATA_PROCESSED, FIGURES, TABLES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Reproducibility and splitting
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # fraction of the full dataset, taken out of the post-test remainder

TARGET = "default"
ID_COL = "ID"

# --------------------------------------------------------------------------------------
# Probability-of-default floor and cap
# --------------------------------------------------------------------------------------
# Isotonic calibration is a step function and will assign a probability of exactly 0 or 1
# to accounts in its outermost steps. Neither is a defensible risk estimate: a PD of zero
# zeroes out that account's expected loss, and a PD of one claims certainty from a handful
# of observations. Supervisory frameworks address this with an explicit floor - Basel uses
# 0.03% for retail exposures - and the same device is applied here, with a symmetric cap.
PD_FLOOR = 0.0003
PD_CAP = 0.9997

# --------------------------------------------------------------------------------------
# Currency
# --------------------------------------------------------------------------------------
# The UCI dataset is denominated in New Taiwan dollars (NT$).
CURRENCY = "NT$"

# --------------------------------------------------------------------------------------
# Expected-loss assumptions  (ASSUMPTIONS, NOT OBSERVED VALUES)
# --------------------------------------------------------------------------------------
# The dataset contains no recovery, collection or charge-off information, so loss given
# default cannot be estimated from it. The values below are stated assumptions used to
# illustrate the EL framework. They are anchored on the broad range typically cited for
# unsecured revolving retail exposures and must not be presented as empirical estimates
# for this portfolio.
LGD_BASELINE = 0.65
LGD_MODERATE = 0.75
LGD_SEVERE = 0.85

# Credit conversion factor: the share of the currently undrawn credit line assumed to be
# drawn between today and the default event. Basel-style treatment for revolving retail
# exposures. Also an assumption, not an observation.
CCF_BASELINE = 0.35
CCF_MODERATE = 0.45
CCF_SEVERE = 0.55

# Scenario stress applied to the probability of default. The stress is applied on the
# log-odds scale (see src/scenarios.py) so that PDs stay inside (0, 1) and low-risk
# accounts deteriorate proportionally rather than by a flat additive amount.
PD_ODDS_MULTIPLIER_BASELINE = 1.00
PD_ODDS_MULTIPLIER_MODERATE = 1.50
PD_ODDS_MULTIPLIER_SEVERE = 2.25

SCENARIOS = {
    "Baseline": {
        "pd_odds_multiplier": PD_ODDS_MULTIPLIER_BASELINE,
        "lgd": LGD_BASELINE,
        "ccf": CCF_BASELINE,
        "narrative": (
            "Current conditions persist. Unemployment and interest rates stay near "
            "present levels; observed default behaviour continues into the next cycle."
        ),
    },
    "Moderate deterioration": {
        "pd_odds_multiplier": PD_ODDS_MULTIPLIER_MODERATE,
        "lgd": LGD_MODERATE,
        "ccf": CCF_MODERATE,
        "narrative": (
            "A mild recession: unemployment rises by roughly 2 percentage points and "
            "policy rates rise, raising default odds by 50% and reducing recoveries."
        ),
    },
    "Severe deterioration": {
        "pd_odds_multiplier": PD_ODDS_MULTIPLIER_SEVERE,
        "lgd": LGD_SEVERE,
        "ccf": CCF_SEVERE,
        "narrative": (
            "A severe recession comparable to a supervisory stress scenario: default "
            "odds more than double, recoveries fall sharply and borrowers draw more of "
            "their undrawn credit lines before defaulting."
        ),
    },
}

# --------------------------------------------------------------------------------------
# Decision-rule cost assumptions (used to test an approve / decline cutoff)
# --------------------------------------------------------------------------------------
# Cost of a false negative = lending to an account that defaults = LGD x EAD (loss).
# Cost of a false positive = declining an account that would have paid = forgone margin.
# The forgone margin is expressed as a share of exposure over the horizon and is, again,
# an assumption.
MARGIN_ON_GOOD_ACCOUNT = 0.06

# --------------------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------------------
N_RISK_BANDS = 10  # deciles
PLOT_STYLE = {
    "figure.figsize": (8, 5),
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}

# Categorical slots, in fixed order, from a colour-vision-deficiency validated palette.
# Hues are assigned by slot order and never cycled; charts that need more than three
# simultaneous categories use a sequential ramp or facet instead.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Single-hue sequential ramp (blue, light -> dark) for magnitude encodings.
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

PALETTE = {
    "primary": "#2a78d6",
    "secondary": "#eb6834",
    "accent": "#1baf7a",
    "good": "#1baf7a",
    "warn": "#eda100",
    "bad": "#e34948",
    "neutral": "#52514e",
    "grid": "#c9c8c3",
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
}

MODEL_ORDER = [
    "Baseline (prior)",
    "Logistic regression",
    "Decision tree",
    "Random forest",
    "Gradient boosting (LightGBM)",
    "Gradient boosting (XGBoost)",
]
