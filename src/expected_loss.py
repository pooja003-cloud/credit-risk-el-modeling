"""Expected-loss framework:  EL = PD x LGD x EAD.

Only one of the three components can be estimated from this dataset.

* **PD** comes from the calibrated model. It is a real estimate.
* **EAD** is constructed from observed balances and credit lines using a credit
  conversion factor for the undrawn portion. The balance and the line are observed; the
  conversion factor is an assumption.
* **LGD** cannot be estimated at all - the dataset contains no recovery, collection or
  charge-off information. It is a stated assumption and is labelled as such in every
  output this module produces.

Presenting an assumed LGD as an observed loss rate would be the single most damaging
thing to do with this analysis, so the assumption is carried alongside the numbers
rather than buried in the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------------------
# Exposure at default
# --------------------------------------------------------------------------------------
def exposure_at_default(
    limit: np.ndarray | pd.Series,
    balance: np.ndarray | pd.Series,
    ccf: float = C.CCF_BASELINE,
) -> np.ndarray:
    """EAD = drawn balance + CCF x undrawn line.

    A credit card is a revolving commitment, so the exposure at the moment of default is
    not today's balance: distressed borrowers typically draw further on the line before
    they stop paying. The credit conversion factor is the assumed share of the undrawn
    line that is drawn between today and default. Using the balance alone (CCF = 0)
    understates exposure; using the full line (CCF = 1) overstates it for accounts that
    are nowhere near their limit.
    """
    limit = np.asarray(limit, dtype=float)
    balance = np.clip(np.asarray(balance, dtype=float), 0, None)
    drawn = np.minimum(balance, limit)
    undrawn = np.clip(limit - drawn, 0, None)
    return drawn + ccf * undrawn


# --------------------------------------------------------------------------------------
# Expected loss
# --------------------------------------------------------------------------------------
def expected_loss(pd_: np.ndarray, lgd: float | np.ndarray, ead: np.ndarray) -> np.ndarray:
    """Element-wise EL = PD x LGD x EAD."""
    return np.asarray(pd_, dtype=float) * np.asarray(lgd, dtype=float) * np.asarray(ead, dtype=float)


def account_level_el(
    frame: pd.DataFrame,
    pd_values: np.ndarray,
    lgd: float = C.LGD_BASELINE,
    ccf: float = C.CCF_BASELINE,
) -> pd.DataFrame:
    """Build the per-account exposure and expected-loss table."""
    out = frame.copy()
    out["pd"] = pd_values
    out["lgd_assumed"] = lgd
    out["ccf_assumed"] = ccf
    out["ead"] = exposure_at_default(out["LIMIT_BAL"], out["BILL_AMT1"], ccf=ccf)
    out["expected_loss"] = expected_loss(out["pd"], lgd, out["ead"])
    out["el_rate_on_ead"] = out["expected_loss"] / out["ead"].replace(0, np.nan)
    return out


def portfolio_summary(el_frame: pd.DataFrame) -> dict:
    """Portfolio-level exposure, expected loss and realised default statistics."""
    total_ead = float(el_frame["ead"].sum())
    total_el = float(el_frame["expected_loss"].sum())
    summary = {
        "accounts": int(len(el_frame)),
        "total_limit": float(el_frame["LIMIT_BAL"].sum()),
        "total_drawn_balance": float(el_frame["BILL_AMT1"].clip(lower=0).sum()),
        "total_ead": total_ead,
        "total_expected_loss": total_el,
        "el_rate_on_ead": total_el / total_ead if total_ead else 0.0,
        "average_pd": float(el_frame["pd"].mean()),
        "exposure_weighted_pd": float(
            np.average(el_frame["pd"], weights=el_frame["ead"]) if total_ead else 0.0
        ),
        "lgd_assumed": float(el_frame["lgd_assumed"].iloc[0]),
        "ccf_assumed": float(el_frame["ccf_assumed"].iloc[0]),
    }
    if C.TARGET in el_frame.columns:
        summary["observed_default_rate"] = float(el_frame[C.TARGET].mean())
        defaulted = el_frame[el_frame[C.TARGET] == 1]
        summary["ead_of_defaulted_accounts"] = float(defaulted["ead"].sum())
        summary["realised_loss_at_assumed_lgd"] = float(
            defaulted["ead"].sum() * summary["lgd_assumed"]
        )
    return summary


def el_by_group(el_frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Expected loss and exposure concentration by any segment column."""
    grp = el_frame.groupby(group_col, observed=True)
    out = grp.agg(
        accounts=("pd", "size"),
        mean_pd=("pd", "mean"),
        total_ead=("ead", "sum"),
        expected_loss=("expected_loss", "sum"),
    ).reset_index()
    if C.TARGET in el_frame.columns:
        out["observed_default_rate"] = grp[C.TARGET].mean().to_numpy()
    out["el_rate_on_ead"] = out["expected_loss"] / out["total_ead"]
    out["share_of_ead"] = out["total_ead"] / out["total_ead"].sum()
    out["share_of_el"] = out["expected_loss"] / out["expected_loss"].sum()
    out["el_concentration"] = out["share_of_el"] / out["share_of_ead"]
    return out


def el_by_risk_band(el_frame: pd.DataFrame, n_bands: int = C.N_RISK_BANDS) -> pd.DataFrame:
    """Expected loss by predicted-risk decile, with exposure concentration."""
    tmp = el_frame.copy()
    tmp["risk_decile"] = pd.qcut(
        tmp["pd"].rank(method="first"), n_bands, labels=range(1, n_bands + 1)
    )
    out = el_by_group(tmp, "risk_decile")
    return out.sort_values("risk_decile").reset_index(drop=True)


def decision_rule_evaluation(
    el_frame: pd.DataFrame,
    threshold: float,
    margin: float = C.MARGIN_ON_GOOD_ACCOUNT,
) -> dict:
    """Apply a decline cutoff to the test fold and measure what it would have done.

    This is a *retrospective* evaluation on held-out data, not a claim about live lending
    decisions. It answers a narrow, checkable question: on this test population, applying
    this cutoff would have avoided this much loss, at the cost of declining this many
    accounts that in fact paid.
    """
    declined = el_frame["pd"] >= threshold
    approved = ~declined
    y = el_frame[C.TARGET].to_numpy().astype(bool)
    lgd = float(el_frame["lgd_assumed"].iloc[0])
    # Margin is earned on the balance the customer carries, not on the committed line.
    revenue_base = el_frame["BILL_AMT1"].clip(lower=0)

    loss_avoided = float((el_frame.loc[declined & y, "ead"] * lgd).sum())
    loss_retained = float((el_frame.loc[approved & y, "ead"] * lgd).sum())
    forgone_margin = float((revenue_base.loc[declined & ~y] * margin).sum())

    return {
        "threshold": float(threshold),
        "approval_rate": float(approved.mean()),
        "decline_rate": float(declined.mean()),
        "accounts_declined": int(declined.sum()),
        "bad_rate_of_approved": float(y[approved].mean()) if approved.any() else 0.0,
        "bad_rate_of_declined": float(y[declined].mean()) if declined.any() else 0.0,
        "bad_rate_no_cutoff": float(y.mean()),
        "defaults_avoided": int((declined & y).sum()),
        "defaults_retained": int((approved & y).sum()),
        "good_accounts_declined": int((declined & ~y).sum()),
        "loss_avoided_at_assumed_lgd": loss_avoided,
        "loss_retained_at_assumed_lgd": loss_retained,
        "forgone_margin_assumed": forgone_margin,
        "net_benefit_assumed": loss_avoided - forgone_margin,
        "lgd_assumed": lgd,
        "margin_assumed": float(margin),
    }
