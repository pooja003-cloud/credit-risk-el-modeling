"""Data loading, validation, cleaning, feature engineering and splitting.

The module is deliberately explicit about two things that matter in a credit-risk
review:

1. **Data integrity** - the loaded file is checked against the counts published for the
   UCI "Default of Credit Card Clients" dataset before anything else happens.
2. **Data leakage** - two feature sets are produced. The *behavioural* set uses the
   six months of repayment history, which is what a bank observes when re-underwriting
   an existing account. The *origination* set uses only what would be known at account
   opening. Comparing them is the honest way to show what the repayment history is
   worth and to avoid quietly presenting a behavioural scorecard as an application
   scorecard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C

# --------------------------------------------------------------------------------------
# Expected characteristics of the source file (published UCI figures)
# --------------------------------------------------------------------------------------
EXPECTED_ROWS = 30_000
EXPECTED_DEFAULTS = 6_636
EXPECTED_DEFAULT_RATE = 0.2212

MONTHS = {1: "Sep-2005", 2: "Aug-2005", 3: "Jul-2005", 4: "Jun-2005", 5: "May-2005", 6: "Apr-2005"}

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

EDUCATION_MAP = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
MARRIAGE_MAP = {1: "Married", 2: "Single", 3: "Other"}
SEX_MAP = {1: "Male", 2: "Female"}


# --------------------------------------------------------------------------------------
# Load and validate
# --------------------------------------------------------------------------------------
def load_raw(path=None) -> pd.DataFrame:
    """Load the raw CSV exactly as distributed."""
    path = path or C.RAW_FILE
    df = pd.read_csv(path)
    return df


def validate_raw(df: pd.DataFrame) -> dict:
    """Check the loaded file against the published dataset statistics.

    Raises if the file does not match, so that a silently truncated or altered mirror
    can never flow through the rest of the pipeline.
    """
    target_raw = "default.payment.next.month"
    checks = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "defaults": int(df[target_raw].sum()),
        "default_rate": float(df[target_raw].mean()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_ids": int(df[C.ID_COL].duplicated().sum()),
        "duplicate_rows_excl_id": int(df.drop(columns=[C.ID_COL]).duplicated().sum()),
    }
    problems = []
    if checks["rows"] != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS} rows, found {checks['rows']}")
    if checks["defaults"] != EXPECTED_DEFAULTS:
        problems.append(f"expected {EXPECTED_DEFAULTS} defaults, found {checks['defaults']}")
    if abs(checks["default_rate"] - EXPECTED_DEFAULT_RATE) > 5e-4:
        problems.append(f"default rate {checks['default_rate']:.4f} != {EXPECTED_DEFAULT_RATE}")
    if problems:
        raise ValueError("Raw data failed validation: " + "; ".join(problems))
    return checks


# --------------------------------------------------------------------------------------
# Clean
# --------------------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename, recode undocumented category codes, and drop exact duplicates.

    Decisions taken, and why:

    * ``PAY_0`` is renamed ``PAY_1`` so that the repayment-status columns line up with
      the bill and payment columns (index 1 = September 2005, the most recent month).
    * ``EDUCATION`` codes 0, 5 and 6 are undocumented and ``4`` is already "other";
      all four are collapsed into "Other" rather than treated as distinct levels.
    * ``MARRIAGE`` code 0 is undocumented and is collapsed into "Other" (code 3).
    * Rows that are exact duplicates once the ``ID`` column is removed are dropped.
      Duplicate observations would otherwise appear in both training and test folds and
      inflate measured performance.
    * ``PAY_*`` values of -2 (no consumption) and -1 (paid in full) are kept as-is but a
      separate non-negative "months past due" version is engineered, because the raw
      scale is not monotone in risk between -2 and 0.
    """
    log: dict = {"rows_in": int(len(df))}
    out = df.copy()

    out = out.rename(columns={"PAY_0": "PAY_1", "default.payment.next.month": C.TARGET})

    edu_recoded = int((~out["EDUCATION"].isin([1, 2, 3])).sum())
    out["EDUCATION"] = out["EDUCATION"].where(out["EDUCATION"].isin([1, 2, 3]), 4)

    mar_recoded = int((~out["MARRIAGE"].isin([1, 2])).sum())
    out["MARRIAGE"] = out["MARRIAGE"].where(out["MARRIAGE"].isin([1, 2]), 3)

    before = len(out)
    out = out.drop_duplicates(subset=[c for c in out.columns if c != C.ID_COL], keep="first")
    dropped_dupes = before - len(out)

    log.update(
        {
            "education_codes_recoded_to_other": edu_recoded,
            "marriage_codes_recoded_to_other": mar_recoded,
            "duplicate_rows_dropped": int(dropped_dupes),
            "rows_out": int(len(out)),
            "default_rate_after_clean": float(out[C.TARGET].mean()),
            "negative_bill_amounts": int((out[BILL_COLS] < 0).to_numpy().sum()),
        }
    )
    return out.reset_index(drop=True), log


# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create credit-risk features from limits, bills, payments and repayment status.

    All features use information dated on or before September 2005. The target is
    default in October 2005, so nothing here is observed after the outcome.
    """
    out = df.copy()
    eps = 1.0  # guards divisions by a zero bill

    # --- Exposure and utilisation -----------------------------------------------------
    out["log_limit_bal"] = np.log1p(out["LIMIT_BAL"])
    for i in range(1, 7):
        out[f"util_{i}"] = out[f"BILL_AMT{i}"].clip(lower=0) / out["LIMIT_BAL"]
    util = out[[f"util_{i}" for i in range(1, 7)]]
    out["util_mean"] = util.mean(axis=1)
    out["util_max"] = util.max(axis=1)
    out["util_latest"] = out["util_1"]
    out["util_trend"] = out["util_1"] - out["util_6"]
    out["util_volatility"] = util.std(axis=1)
    out["months_over_90pct_util"] = (util > 0.90).sum(axis=1)

    # --- Repayment behaviour ----------------------------------------------------------
    # PAY_AMT{i} is the amount paid in month i, settling the bill raised in month i+1.
    for i in range(1, 6):
        prior_bill = out[f"BILL_AMT{i + 1}"].clip(lower=0)
        out[f"pay_ratio_{i}"] = out[f"PAY_AMT{i}"] / (prior_bill + eps)
        out[f"pay_ratio_{i}"] = out[f"pay_ratio_{i}"].clip(upper=2.0)
    ratios = out[[f"pay_ratio_{i}" for i in range(1, 6)]]
    out["pay_ratio_mean"] = ratios.mean(axis=1)
    out["pay_ratio_min"] = ratios.min(axis=1)
    out["pay_ratio_latest"] = out["pay_ratio_1"]
    out["months_paid_nothing"] = (out[PAY_AMT_COLS] == 0).sum(axis=1)
    out["total_paid_6m"] = out[PAY_AMT_COLS].sum(axis=1)
    out["total_billed_6m"] = out[BILL_COLS].clip(lower=0).sum(axis=1)
    out["paid_to_billed_6m"] = out["total_paid_6m"] / (out["total_billed_6m"] + eps)
    out["paid_to_billed_6m"] = out["paid_to_billed_6m"].clip(upper=2.0)

    # --- Delinquency ------------------------------------------------------------------
    dpd = out[PAY_COLS].clip(lower=0)  # months past due, floored at 0
    dpd.columns = [f"dpd_{i}" for i in range(1, 7)]
    out = pd.concat([out, dpd], axis=1)
    out["dpd_max"] = dpd.max(axis=1)
    out["dpd_latest"] = out["dpd_1"]
    out["months_delinquent"] = (dpd > 0).sum(axis=1)
    out["months_2plus_delinquent"] = (dpd >= 2).sum(axis=1)
    out["dpd_trend"] = out["dpd_1"] - out["dpd_6"]
    out["ever_delinquent"] = (out["dpd_max"] > 0).astype(int)
    out["revolver_months"] = (out[PAY_COLS] == 0).sum(axis=1)  # paid the minimum only
    out["full_payer_months"] = (out[PAY_COLS] == -1).sum(axis=1)
    out["inactive_months"] = (out[PAY_COLS] == -2).sum(axis=1)

    # --- Balance dynamics -------------------------------------------------------------
    out["bill_latest"] = out["BILL_AMT1"]
    out["bill_mean"] = out[BILL_COLS].mean(axis=1)
    out["bill_growth_6m"] = (out["BILL_AMT1"] - out["BILL_AMT6"]) / (
        out["BILL_AMT6"].abs() + eps
    )
    out["bill_growth_6m"] = out["bill_growth_6m"].clip(-5, 5)
    out["available_credit"] = (out["LIMIT_BAL"] - out["BILL_AMT1"]).clip(lower=0)

    # --- Demographics -----------------------------------------------------------------
    out["age_band"] = pd.cut(
        out["AGE"],
        bins=[20, 25, 30, 35, 40, 50, 60, 100],
        labels=["21-25", "26-30", "31-35", "36-40", "41-50", "51-60", "60+"],
        right=True,
    )
    out["limit_band"] = pd.qcut(
        out["LIMIT_BAL"], q=5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]
    )
    out["sex_label"] = out["SEX"].map(SEX_MAP)
    out["education_label"] = out["EDUCATION"].map(EDUCATION_MAP)
    out["marriage_label"] = out["MARRIAGE"].map(MARRIAGE_MAP)
    return out


# --------------------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------------------
@dataclass
class FeatureSet:
    """A named group of model inputs plus the reasoning behind the selection."""

    name: str
    numeric: list[str]
    categorical: list[str]
    rationale: str
    excluded: dict[str, str] = field(default_factory=dict)

    @property
    def all(self) -> list[str]:
        return self.numeric + self.categorical


BEHAVIOURAL = FeatureSet(
    name="behavioural",
    numeric=[
        "log_limit_bal",
        "AGE",
        "util_latest",
        "util_mean",
        "util_max",
        "util_trend",
        "util_volatility",
        "months_over_90pct_util",
        "pay_ratio_latest",
        "pay_ratio_mean",
        "pay_ratio_min",
        "paid_to_billed_6m",
        "months_paid_nothing",
        "dpd_latest",
        "dpd_max",
        "months_delinquent",
        "months_2plus_delinquent",
        "dpd_trend",
        "revolver_months",
        "full_payer_months",
        "inactive_months",
        "bill_latest",
        "bill_mean",
        "bill_growth_6m",
        "available_credit",
    ],
    categorical=["sex_label", "education_label", "marriage_label"],
    rationale=(
        "Everything a card issuer observes on an existing account at the September 2005 "
        "cut-off: the credit line, demographics captured at opening, and six months of "
        "billing, payment and repayment-status history. This is a behavioural scorecard."
    ),
    excluded={
        "ID": "Row identifier, carries no risk information.",
        "raw PAY_1..PAY_6": (
            "Replaced by non-negative days-past-due versions plus explicit counts of "
            "revolving, full-payment and inactive months. The raw -2/-1/0 codes are "
            "categorical states, not a monotone severity scale."
        ),
        "any October 2005 information": (
            "The target is the October 2005 default flag. Nothing dated on or after the "
            "outcome month is used, which is the leakage boundary for this model."
        ),
    },
)

ORIGINATION = FeatureSet(
    name="origination",
    numeric=["log_limit_bal", "AGE"],
    categorical=["sex_label", "education_label", "marriage_label"],
    rationale=(
        "Only what is known when the account is opened: the assigned credit line and "
        "application demographics. No repayment history exists at that point, so this "
        "is the fair comparison for an application scorecard and shows how much of the "
        "behavioural model's power comes from history the issuer would not yet have."
    ),
    excluded={
        "all repayment, billing and delinquency history": (
            "Not observable at origination. Including it in an application model would "
            "be data leakage relative to the decision being made."
        )
    },
)

FEATURE_SETS = {fs.name: fs for fs in (BEHAVIOURAL, ORIGINATION)}

# Columns kept alongside the features for exposure and segment reporting.
CARRY_COLS = [
    C.ID_COL,
    "LIMIT_BAL",
    "BILL_AMT1",
    "AGE",
    "age_band",
    "limit_band",
    "sex_label",
    "education_label",
    "marriage_label",
    "dpd_max",
    "months_delinquent",
    "util_latest",
]


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------
def split_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Stratified train / validation / test split, 60 / 20 / 20.

    The dataset carries no origination or observation dates - every account is observed
    over the same six-month window - so a time-based split is not possible. A stratified
    random split is used instead and the limitation is stated in the model card. The
    validation fold is used for threshold selection and probability calibration; the
    test fold is touched once, for final reporting.
    """
    train_val, test = train_test_split(
        df,
        test_size=C.TEST_SIZE,
        stratify=df[C.TARGET],
        random_state=C.RANDOM_STATE,
    )
    val_fraction = C.VAL_SIZE / (1.0 - C.TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val[C.TARGET],
        random_state=C.RANDOM_STATE,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def build_dataset(save: bool = True) -> tuple[dict[str, pd.DataFrame], dict]:
    """Run the full preparation pipeline and persist the splits."""
    raw = load_raw()
    integrity = validate_raw(raw)
    cleaned, clean_log = clean(raw)
    featured = engineer_features(cleaned)
    splits = split_data(featured)

    report = {
        "source_file": str(C.RAW_FILE.name),
        "integrity_checks": integrity,
        "cleaning": clean_log,
        "n_features_behavioural": len(BEHAVIOURAL.all),
        "n_features_origination": len(ORIGINATION.all),
        "splits": {
            k: {
                "rows": int(len(v)),
                "defaults": int(v[C.TARGET].sum()),
                "default_rate": float(v[C.TARGET].mean()),
            }
            for k, v in splits.items()
        },
    }

    if save:
        for name, frame in splits.items():
            frame.to_parquet(C.DATA_PROCESSED / f"{name}.parquet", index=False)
        (C.TABLES / "data_preparation_report.json").write_text(json.dumps(report, indent=2))
    return splits, report


def load_splits() -> dict[str, pd.DataFrame]:
    """Read the persisted splits, building them first if they do not exist."""
    paths = {n: C.DATA_PROCESSED / f"{n}.parquet" for n in ("train", "val", "test")}
    if not all(p.exists() for p in paths.values()):
        splits, _ = build_dataset(save=True)
        return splits
    return {n: pd.read_parquet(p) for n, p in paths.items()}


if __name__ == "__main__":
    _, rep = build_dataset()
    print(json.dumps(rep, indent=2))
