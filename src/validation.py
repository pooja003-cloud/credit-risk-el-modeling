"""Model-validation tests for the PD model.

These follow the outcome-analysis, implementation and sensitivity checks a model-risk
validation would run on a PD model:

* Hosmer-Lemeshow goodness-of-fit across risk deciles
* a binomial test of observed against predicted defaults in each decile, at 99%
* champion against challenger output divergence, with a tolerance
* input sensitivity: each main driver shocked by +/-10%, 20% and 50%
* a 25-record data-lineage trace from the raw file to the model inputs

The lineage trace recomputes features with plain arithmetic from the raw CSV rather than
calling ``data_prep``, so it checks the transformation code instead of agreeing with it.

One caveat that belongs on every output: these tests were run by whoever built the model,
so they are a self-assessment against a validation framework, not an independent
validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

# ---------------------------------------------------------------------------
# Outcome analysis
# ---------------------------------------------------------------------------


def _deciles(p: np.ndarray, groups: int = 10) -> np.ndarray:
    ranks = pd.Series(p).rank(method="first")
    return pd.qcut(ranks, groups, labels=range(1, groups + 1)).astype(int).to_numpy()


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10) -> dict:
    """Hosmer-Lemeshow chi-squared statistic over equal-count risk groups.

    Degrees of freedom are reported both ways: g - 2 is the textbook in-sample
    convention, g is the usual choice when the model is scored on data it was not
    fitted on, which is the case here. On a sample this size the test has a lot of
    power, so small absolute miscalibration can still produce a low p-value - the
    per-group table is the thing to read, not just the verdict.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    g = _deciles(p, groups)
    rows, stat = [], 0.0
    for k in range(1, groups + 1):
        m = g == k
        n = int(m.sum())
        obs = float(y[m].sum())
        exp = float(p[m].sum())
        var = exp * (1 - exp / n)
        contrib = (obs - exp) ** 2 / var if var > 0 else 0.0
        stat += contrib
        rows.append({"group": k, "accounts": n, "observed_defaults": obs,
                     "expected_defaults": exp, "contribution": contrib})
    df_in, df_out = groups - 2, groups
    return {
        "statistic": float(stat),
        "df_in_sample": df_in,
        "p_value_in_sample_df": float(stats.chi2.sf(stat, df_in)),
        "df_out_of_sample": df_out,
        "p_value_out_of_sample_df": float(stats.chi2.sf(stat, df_out)),
        "groups": pd.DataFrame(rows),
    }


def binomial_by_decile(y: np.ndarray, p: np.ndarray, confidence: float = 0.99) -> pd.DataFrame:
    """Binomial test of observed defaults against the band's mean predicted PD.

    Treats each decile as a rating grade with a single PD equal to its mean prediction,
    which is the standard grade-level backtest. A band passes if its observed default
    count falls inside the two-sided interval the predicted PD implies.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    g = _deciles(p)
    tail = (1 - confidence) / 2
    rows = []
    for k in range(1, 11):
        m = g == k
        n = int(m.sum())
        defaults = int(y[m].sum())
        pd_band = float(p[m].mean())
        lo = int(stats.binom.ppf(tail, n, pd_band))
        hi = int(stats.binom.ppf(1 - tail, n, pd_band))
        rows.append({
            "decile": k,
            "accounts": n,
            "observed_defaults": defaults,
            "expected_defaults": n * pd_band,
            "mean_predicted_pd": pd_band,
            "observed_default_rate": defaults / n,
            "interval_low": lo,
            "interval_high": hi,
            "p_value": float(stats.binomtest(defaults, n, pd_band).pvalue),
            "passes": lo <= defaults <= hi,
        })
    return pd.DataFrame(rows)


def calibration_diagnostics(y: np.ndarray, p: np.ndarray, ead: np.ndarray) -> dict:
    """Calibration slope, and actual-to-expected weighted two ways.

    A slope below 1 means predicted PDs are more spread out than outcomes justify: the
    model is over-confident, too low at the safe end and too high at the risky end.

    Weighting matters for expected loss. If the under-predicted accounts are also the
    large ones, a model that is conservative by account count can still be optimistic by
    exposure, and expected loss follows the exposure-weighted figure.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    e = np.asarray(ead, dtype=float)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    return {
        "calibration_slope": float(fit.coef_[0][0]),
        "calibration_intercept": float(fit.intercept_[0]),
        "predicted_rate_count_weighted": float(p.mean()),
        "observed_rate_count_weighted": float(y.mean()),
        "ae_ratio_count_weighted": float(y.mean() / p.mean()),
        "predicted_rate_exposure_weighted": float(np.average(p, weights=e)),
        "observed_rate_exposure_weighted": float(np.average(y, weights=e)),
        "ae_ratio_exposure_weighted": float(np.average(y, weights=e) / np.average(p, weights=e)),
    }


# ---------------------------------------------------------------------------
# Implementation: champion against challenger
# ---------------------------------------------------------------------------


def champion_vs_challenger(p_champ, p_chall, ead, lgd: float, tolerance: float = 0.05) -> dict:
    """How far the champion's outputs sit from the challenger's on identical inputs.

    The portfolio-level figures are the ones the tolerance applies to. The account-level
    figures show the two models can agree in aggregate while disagreeing a lot on
    individual accounts, which matters if the score is used for decisions.
    """
    p1, p2, e = (np.asarray(a, dtype=float) for a in (p_champ, p_chall, ead))
    el1, el2 = float((p1 * lgd * e).sum()), float((p2 * lgd * e).sum())
    diff = np.abs(p1 - p2)
    mean_pd_gap = float(p1.mean() / p2.mean() - 1)
    el_gap = el1 / el2 - 1
    return {
        "champion_mean_pd": float(p1.mean()),
        "challenger_mean_pd": float(p2.mean()),
        "mean_pd_relative_gap": mean_pd_gap,
        "champion_el": el1,
        "challenger_el": el2,
        "el_relative_gap": el_gap,
        "tolerance": tolerance,
        "within_tolerance": abs(el_gap) <= tolerance and abs(mean_pd_gap) <= tolerance,
        "account_mean_abs_gap_pp": float(diff.mean() * 100),
        "account_p90_abs_gap_pp": float(np.quantile(diff, 0.9) * 100),
        "share_within_5pp": float((diff <= 0.05).mean()),
        "rank_correlation": float(stats.spearmanr(p1, p2).statistic),
    }


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

# Continuous drivers take proportional shocks. Delinquency counts are whole months, so a
# 10% shock to "2 months late" is meaningless; they take +/-1 month instead.
PROPORTIONAL_DRIVERS = {
    "available_credit": "Available credit",
    "bill_mean": "Average bill (6 months)",
    "bill_latest": "Latest bill",
    "util_mean": "Average utilisation",
    "pay_ratio_min": "Lowest payment ratio",
    "log_limit_bal": "Credit limit",
}
COUNT_DRIVERS = {
    "dpd_latest": ("Latest months past due", 8),
    "dpd_max": ("Worst months past due", 8),
    "months_2plus_delinquent": ("Months 2+ cycles late", 6),
}
SHOCKS = (-0.50, -0.20, -0.10, 0.10, 0.20, 0.50)


def input_sensitivity(model, frame: pd.DataFrame, features: list[str], ead: np.ndarray) -> pd.DataFrame:
    """Shock one input at a time and measure the change in portfolio PD.

    Each shock moves a single model input and leaves the rest alone, which is how a
    local sensitivity test is defined. It does produce rows that are not internally
    consistent - a higher latest bill without a matching higher utilisation - so read the
    results as "how much does the model lean on this input", not as a scenario.

    The credit limit enters the model as log(1 + limit), so it is shocked on the limit
    itself and transformed back.
    """
    X = frame[features].copy()
    base = model.predict_proba(X)[:, 1]
    base_mean = base.mean()
    base_ew = np.average(base, weights=ead)
    rows = []

    def record(name, label, kind, shock, shocked):
        p = model.predict_proba(shocked)[:, 1]
        rows.append({
            "input": name, "label": label, "shock_type": kind, "shock": shock,
            "mean_pd": float(p.mean()),
            "mean_pd_change": float(p.mean() / base_mean - 1),
            "exposure_weighted_pd_change": float(np.average(p, weights=ead) / base_ew - 1),
        })

    for name, label in PROPORTIONAL_DRIVERS.items():
        if name not in features:
            continue
        for s in SHOCKS:
            Xs = X.copy()
            if name == "log_limit_bal":
                Xs[name] = np.log1p(np.expm1(X[name]) * (1 + s))
            else:
                Xs[name] = X[name] * (1 + s)
            record(name, label, "proportional", s, Xs)

    for name, (label, cap) in COUNT_DRIVERS.items():
        if name not in features:
            continue
        for s in (-1, 1):
            Xs = X.copy()
            Xs[name] = (X[name] + s).clip(0, cap)
            record(name, label, "months", s, Xs)

    return pd.DataFrame(rows)


def sensitivity_ranking(table: pd.DataFrame) -> pd.DataFrame:
    """Largest absolute PD response per input, for a one-line-per-driver summary."""
    t = table.assign(abs_change=table["mean_pd_change"].abs())
    idx = t.groupby("input")["abs_change"].idxmax()
    return (t.loc[idx, ["input", "label", "shock_type", "shock", "mean_pd_change"]]
            .sort_values("mean_pd_change", key=np.abs, ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Data lineage
# ---------------------------------------------------------------------------


def _recompute_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a handful of model inputs directly from raw columns.

    Deliberately independent of ``data_prep.engineer_features``: if that code has a
    bug, this will disagree with it.
    """
    pay = raw[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].clip(lower=0)
    out = pd.DataFrame({"ID": raw["ID"]})
    out["log_limit_bal"] = np.log1p(raw["LIMIT_BAL"])
    out["util_latest"] = raw["BILL_AMT1"].clip(lower=0) / raw["LIMIT_BAL"]
    out["available_credit"] = (raw["LIMIT_BAL"] - raw["BILL_AMT1"]).clip(lower=0)
    out["dpd_latest"] = pay["PAY_0"]
    out["dpd_max"] = pay.max(axis=1)
    out["months_delinquent"] = (pay > 0).sum(axis=1)
    out["pay_ratio_latest"] = (raw["PAY_AMT1"] / (raw["BILL_AMT2"].clip(lower=0) + 1)).clip(upper=2)
    out["bill_latest"] = raw["BILL_AMT1"]
    return out


def data_lineage_trace(test: pd.DataFrame, n: int = 25, tolerance: float = 0.005) -> dict:
    """Trace a random sample of test accounts from the raw file to the model inputs."""
    raw = pd.read_csv(C.RAW_FILE)
    sample_ids = test[C.ID_COL].sample(n=n, random_state=C.RANDOM_STATE).to_numpy()
    rebuilt = _recompute_from_raw(raw[raw["ID"].isin(sample_ids)]).set_index("ID")
    stored = test.set_index(C.ID_COL).loc[sample_ids]

    rows = []
    for col in rebuilt.columns:
        a = rebuilt.loc[sample_ids, col].to_numpy(dtype=float)
        b = stored[col].to_numpy(dtype=float)
        denom = np.maximum(np.abs(a), 1e-9)
        rel = np.where(np.abs(a) < 1e-9, np.abs(b - a), np.abs(b - a) / denom)
        rows.append({
            "feature": col,
            "records_checked": n,
            "max_abs_difference": float(np.max(np.abs(b - a))),
            "max_relative_discrepancy": float(np.max(rel)),
            "passes": bool(np.max(rel) < tolerance),
        })
    table = pd.DataFrame(rows)
    return {
        "records": n,
        "features_checked": len(table),
        "tolerance": tolerance,
        "all_pass": bool(table["passes"].all()),
        "worst_relative_discrepancy": float(table["max_relative_discrepancy"].max()),
        "table": table,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all(champion_model, challenger_model, test: pd.DataFrame, features: list[str],
            ead: np.ndarray, lgd: float = C.LGD_BASELINE) -> dict:
    """Run every test and return plain results plus the tables to save."""
    y = test[C.TARGET].to_numpy()
    p_champ = champion_model.predict_proba(test[features])[:, 1]
    p_chall = challenger_model.predict_proba(test[features])[:, 1]

    hl = hosmer_lemeshow(y, p_champ)
    binom = binomial_by_decile(y, p_champ)
    calib = calibration_diagnostics(y, p_champ, ead)
    cc = champion_vs_challenger(p_champ, p_chall, ead, lgd)
    sens = input_sensitivity(champion_model, test, features, ead)
    lineage = data_lineage_trace(test)

    ranking = sensitivity_ranking(sens)
    summary = {
        "hosmer_lemeshow": {k: v for k, v in hl.items() if k != "groups"},
        "calibration": calib,
        "binomial": {
            "confidence": 0.99,
            "bands_passing": int(binom["passes"].sum()),
            "bands_total": int(len(binom)),
            "failing_deciles": binom.loc[~binom["passes"], "decile"].astype(int).tolist(),
        },
        "champion_vs_challenger": cc,
        "sensitivity_most_sensitive": ranking.iloc[0].to_dict(),
        "sensitivity_ranking": ranking.to_dict(orient="records"),
        "lineage": {k: v for k, v in lineage.items() if k != "table"},
        "auc_threshold": 0.70,
        "independent": False,
    }
    tables = {
        "validation_hosmer_lemeshow": hl["groups"],
        "validation_binomial_by_decile": binom,
        "validation_input_sensitivity": sens,
        "validation_sensitivity_ranking": ranking,
        "validation_data_lineage": lineage["table"],
    }
    return {"summary": summary, "tables": tables}
