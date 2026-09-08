"""Model evaluation: discrimination, calibration, thresholds, deciles and segments.

Accuracy is not reported as a headline anywhere. On a portfolio with a 22% default rate,
a model that approves everyone is 78% accurate and completely useless, so the metrics
here are the ones a credit-risk review actually asks for: rank-ordering (ROC-AUC, Gini,
KS), performance on the minority class (precision, recall, F1, PR-AUC), and whether the
predicted probabilities are *levels* that can be multiplied into an expected loss
(calibration curve, Brier score, expected calibration error).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config as C

EPS = 1e-12


# --------------------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------------------
def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between the good and bad score distributions.

    Still the headline separation measure in most retail scorecard shops.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Average absolute gap between predicted and observed default rate, bin-weighted."""
    bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf
    idx = np.digitize(y_prob, bins[1:-1], right=True)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.sum() * abs(y_prob[m].mean() - y_true[m].mean())
    return float(total / len(y_true))


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Confusion-matrix-derived metrics at one decision cutoff."""
    y_hat = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "flagged_rate": float(y_hat.mean()),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "accuracy": float((tp + tn) / len(y_true)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def core_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Threshold-free discrimination and calibration metrics."""
    constant = float(np.std(y_prob)) < 1e-12
    auc = 0.5 if constant else float(roc_auc_score(y_true, y_prob))
    return {
        "roc_auc": auc,
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ks": 0.0 if constant else ks_statistic(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, EPS, 1 - EPS))),
        "ece": expected_calibration_error(y_true, y_prob),
        "mean_predicted_pd": float(y_prob.mean()),
        "observed_default_rate": float(y_true.mean()),
        "prevalence_ratio": float(y_prob.mean() / max(y_true.mean(), EPS)),
    }


# --------------------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------------------
def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    grid = np.quantile(y_prob, np.linspace(0.01, 0.99, 99))
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    exposure: np.ndarray,
    revenue_base: np.ndarray | None = None,
    lgd: float = C.LGD_BASELINE,
    margin: float = C.MARGIN_ON_GOOD_ACCOUNT,
) -> tuple[float, pd.DataFrame]:
    """Find the decline cutoff that minimises net cost on the *validation* fold.

    Two errors have different prices, and in credit they are not close:

    * a false negative (approving an account that defaults) costs ``LGD x EAD``;
    * a false positive (declining an account that would have paid) costs the forgone
      margin, earned on the balance the customer actually carries rather than on the
      whole committed line.

    Both the LGD and the margin are stated assumptions (see ``src/config.py``). The
    result is very sensitive to their ratio, which is exactly why
    :func:`margin_sensitivity` exists: a single "optimal" cutoff quoted without that
    sensitivity would be a false precision.
    """
    revenue_base = exposure if revenue_base is None else revenue_base
    grid = np.unique(np.quantile(y_prob, np.linspace(0.05, 0.999, 200)))
    rows = []
    for t in grid:
        declined = y_prob >= t
        approved = ~declined
        loss_on_approved = float((exposure[approved & (y_true == 1)] * lgd).sum())
        forgone = float((revenue_base[declined & (y_true == 0)] * margin).sum())
        rows.append(
            {
                "threshold": float(t),
                "approval_rate": float(approved.mean()),
                "loss_on_approved": loss_on_approved,
                "forgone_margin": forgone,
                "net_cost": loss_on_approved + forgone,
                "bad_rate_of_approved": float(y_true[approved].mean()) if approved.any() else 0.0,
            }
        )
    curve = pd.DataFrame(rows)
    return float(curve.loc[curve["net_cost"].idxmin(), "threshold"]), curve


def margin_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    exposure: np.ndarray,
    revenue_base: np.ndarray,
    lgd: float = C.LGD_BASELINE,
    margins: tuple[float, ...] = (0.06, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60),
) -> pd.DataFrame:
    """How the loss-minimising cutoff moves with the assumed margin on a good account.

    A one-period loss of ``LGD x EAD`` is being weighed against a margin earned over the
    life of the relationship. If the margin is stated as a single year's spread, the
    arithmetic will always recommend an implausibly tight credit policy. This table makes
    that dependence explicit instead of hiding it inside one headline cutoff.
    """
    rows = []
    for m in margins:
        t, curve = cost_optimal_threshold(
            y_true, y_prob, exposure, revenue_base=revenue_base, lgd=lgd, margin=m
        )
        best = curve.loc[curve["net_cost"].idxmin()]
        rows.append(
            {
                "assumed_margin_on_good_account": m,
                "implied_cutoff": float(t),
                "approval_rate": float(best["approval_rate"]),
                "bad_rate_of_approved": float(best["bad_rate_of_approved"]),
                "net_cost": float(best["net_cost"]),
            }
        )
    return pd.DataFrame(rows)


def threshold_for_target_bad_rate(
    y_true: np.ndarray, y_prob: np.ndarray, target_bad_rate: float = 0.15
) -> float:
    """The loosest cutoff that still keeps the approved population's bad rate on target.

    This is how a credit policy is set in practice: risk appetite is expressed as a
    tolerable default rate on the book, and the cutoff is whatever delivers it. It is
    reported alongside the cost-minimising cutoff because the two answer different
    questions - "what does the business accept?" versus "what do the economics imply?".
    """
    # Walk cutoffs from loosest to tightest and stop at the first one that meets appetite.
    grid = np.unique(np.quantile(y_prob, np.linspace(0.02, 0.999, 400)))[::-1]
    for t in grid:
        approved = y_prob < t
        if approved.sum() < 50:
            continue
        if float(y_true[approved].mean()) <= target_bad_rate:
            return float(t)
    return float(grid[-1])


# --------------------------------------------------------------------------------------
# Risk bands and segments
# --------------------------------------------------------------------------------------
def decile_table(
    y_true: np.ndarray, y_prob: np.ndarray, exposure: np.ndarray | None = None
) -> pd.DataFrame:
    """Observed default rate, lift and capture by predicted-risk decile.

    Decile 1 is the safest 10% of accounts, decile 10 the riskiest. A monotone climb
    down this table is the single most persuasive evidence that a PD model rank-orders,
    and it is the form a credit committee is used to reading.
    """
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    if exposure is not None:
        df["exposure"] = exposure
    df["decile"] = pd.qcut(df["p"].rank(method="first"), C.N_RISK_BANDS, labels=range(1, 11))
    grp = df.groupby("decile", observed=True)
    out = grp.agg(
        accounts=("y", "size"),
        defaults=("y", "sum"),
        observed_default_rate=("y", "mean"),
        mean_predicted_pd=("p", "mean"),
        min_predicted_pd=("p", "min"),
        max_predicted_pd=("p", "max"),
    ).reset_index()
    if exposure is not None:
        out["exposure"] = grp["exposure"].sum().to_numpy()
        out["share_of_exposure"] = out["exposure"] / out["exposure"].sum()
    overall = float(np.mean(y_true))
    out["lift"] = out["observed_default_rate"] / overall
    out = out.sort_values("decile", ascending=False)
    out["cumulative_defaults_captured"] = out["defaults"].cumsum() / out["defaults"].sum()
    out["cumulative_accounts"] = out["accounts"].cumsum() / out["accounts"].sum()
    return out.sort_values("decile").reset_index(drop=True)


def segment_performance(
    frame: pd.DataFrame, y_prob: np.ndarray, segment_cols: list[str]
) -> pd.DataFrame:
    """Discrimination and calibration inside each borrower segment.

    A portfolio-level AUC can hide a model that works well on one population and fails on
    another. Segment-level performance is what a fair-lending or model-validation review
    asks for first.
    """
    rows = []
    y = frame[C.TARGET].to_numpy()
    for col in segment_cols:
        for value, idx in frame.groupby(col, observed=True).groups.items():
            m = frame.index.isin(idx)
            yt, yp = y[m], y_prob[m]
            if len(yt) < 50 or yt.sum() < 5 or yt.sum() == len(yt):
                continue
            rows.append(
                {
                    "segment_type": col,
                    "segment": str(value),
                    "accounts": int(len(yt)),
                    "defaults": int(yt.sum()),
                    "observed_default_rate": float(yt.mean()),
                    "mean_predicted_pd": float(yp.mean()),
                    "roc_auc": float(roc_auc_score(yt, yp)),
                    "brier": float(brier_score_loss(yt, yp)),
                    "calibration_gap": float(yp.mean() - yt.mean()),
                }
            )
    return pd.DataFrame(rows)


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted versus observed default rate, by equal-count probability bin."""
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
    out = (
        df.groupby("bin", observed=True)
        .agg(accounts=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    out["gap"] = out["predicted"] - out["observed"]
    return out


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    row = {"model": name}
    row.update(core_metrics(y_true, y_prob))
    row.update({f"at_cutoff_{k}": v for k, v in threshold_metrics(y_true, y_prob, threshold).items()})
    return row


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=float))
