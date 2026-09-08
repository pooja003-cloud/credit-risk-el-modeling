"""Independent verification of every headline number.

This script deliberately does **not** import the project's own risk modules. It reads the
scored account file and recomputes each published figure from first principles with plain
numpy and scikit-learn, then compares against ``outputs/tables/results.json``. If a bug in
``src/`` silently changed a definition, the two would disagree here.

    python tools/verify_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

LGD, CCF, MARGIN = 0.65, 0.35, 0.06
SCENARIOS = [("Baseline", 1.00, 0.65, 0.35),
             ("Moderate deterioration", 1.50, 0.75, 0.45),
             ("Severe deterioration", 2.25, 0.85, 0.55)]

checks: list[tuple[str, float, float, bool]] = []


def check(name: str, recomputed: float, published: float, tol: float = 1e-6) -> None:
    ok = abs(recomputed - published) <= tol * max(1.0, abs(published))
    checks.append((name, recomputed, published, ok))


def main() -> int:
    r = json.loads((TABLES / "results.json").read_text())
    h, el_pub, dec_pub, th = (r["headline"], r["expected_loss_baseline"],
                              r["decision_rule"], r["thresholds"])
    acc = pd.read_csv(TABLES / "scored_test_accounts.csv")

    y = acc["default"].to_numpy()
    p = acc["pd"].to_numpy()

    # --- discrimination and calibration, from scratch --------------------------------
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    check("test accounts", len(acc), h["test_accounts"])
    check("test default rate", y.mean(), h["test_default_rate"])
    check("ROC-AUC", auc, h["champion_roc_auc"])
    check("Gini", 2 * auc - 1, h["champion_gini"])
    check("PR-AUC", average_precision_score(y, p), h["champion_pr_auc"])
    check("KS", float(np.max(tpr - fpr)), h["champion_ks"])
    check("Brier", brier_score_loss(y, p), h["champion_brier"])

    # --- exposure and expected loss ---------------------------------------------------
    drawn = np.minimum(np.clip(acc["BILL_AMT1"].to_numpy(), 0, None), acc["LIMIT_BAL"].to_numpy())
    undrawn = np.clip(acc["LIMIT_BAL"].to_numpy() - drawn, 0, None)
    ead = drawn + CCF * undrawn
    el = p * LGD * ead
    check("portfolio EAD", ead.sum(), h["portfolio_ead"], tol=1e-9)
    check("portfolio EL", el.sum(), h["portfolio_el"], tol=1e-9)
    check("EL rate on EAD", el.sum() / ead.sum(), h["portfolio_el_rate"])
    check("average PD", p.mean(), el_pub["average_pd"])
    check("exposure-weighted PD", float(np.average(p, weights=ead)), el_pub["exposure_weighted_pd"])
    check("EAD of defaulted accounts", ead[y == 1].sum(), el_pub["ead_of_defaulted_accounts"], 1e-9)

    # --- deciles ----------------------------------------------------------------------
    order = pd.Series(p).rank(method="first")
    decile = pd.qcut(order, 10, labels=range(1, 11)).astype(int).to_numpy()
    top, bottom = y[decile == 10].mean(), y[decile == 1].mean()
    check("top decile default rate", top, h["top_decile_default_rate"])
    check("bottom decile default rate", bottom, h["bottom_decile_default_rate"])
    check("top decile lift", top / y.mean(), h["top_decile_lift"])
    check("top two deciles capture", y[decile >= 9].sum() / y.sum(), h["top_two_deciles_capture"])

    # --- decision rule ----------------------------------------------------------------
    cutoff = th["policy_cutoff_from_validation"]
    declined = p >= cutoff
    approved = ~declined
    check("policy approval rate", approved.mean(), dec_pub["approval_rate"])
    check("accounts declined", int(declined.sum()), dec_pub["accounts_declined"])
    check("bad rate of approved", y[approved].mean(), dec_pub["bad_rate_of_approved"])
    check("bad rate of declined", y[declined].mean(), dec_pub["bad_rate_of_declined"])
    check("defaults avoided", int((declined & (y == 1)).sum()), dec_pub["defaults_avoided"])
    check("good accounts declined", int((declined & (y == 0)).sum()),
          dec_pub["good_accounts_declined"])
    check("loss avoided", float(ead[declined & (y == 1)].sum() * LGD),
          dec_pub["loss_avoided_at_assumed_lgd"], 1e-9)
    revenue_base = np.clip(acc["BILL_AMT1"].to_numpy(), 0, None)
    check("forgone margin", float(revenue_base[declined & (y == 0)].sum() * MARGIN),
          dec_pub["forgone_margin_assumed"], 1e-9)

    # --- scenarios --------------------------------------------------------------------
    published_scen = {row["scenario"]: row for row in r["scenarios"]}
    baseline_el = None
    p_safe = np.clip(p, 1e-9, 1 - 1e-9)
    for name, mult, lgd_s, ccf_s in SCENARIOS:
        odds = p_safe / (1 - p_safe) * mult
        p_s = odds / (1 + odds)
        ead_s = drawn + ccf_s * undrawn
        el_s = float((p_s * lgd_s * ead_s).sum())
        baseline_el = el_s if name == "Baseline" else baseline_el
        check(f"scenario EL - {name}", el_s, published_scen[name]["total_expected_loss"], 1e-9)
        check(f"scenario EL uplift - {name}", el_s / baseline_el - 1,
              published_scen[name]["el_vs_baseline_pct"], 1e-6)
    check("severe EL (headline)", published_scen["Severe deterioration"]["total_expected_loss"],
          h["severe_el"], 1e-9)

    # --- PD floor and cap actually applied -------------------------------------------
    check("minimum PD >= floor", float(p.min() >= 0.0003), 1.0)
    check("maximum PD <= cap", float(p.max() <= 0.9997), 1.0)

    # --- report -----------------------------------------------------------------------
    width = max(len(n) for n, *_ in checks)
    failed = 0
    for name, got, want, ok in checks:
        flag = "PASS" if ok else "FAIL"
        failed += 0 if ok else 1
        print(f"[{flag}] {name:<{width}}  recomputed {got:>18,.6f}   published {want:>18,.6f}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
