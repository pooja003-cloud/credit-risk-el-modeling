"""End-to-end pipeline: data -> models -> evaluation -> expected loss -> figures -> results.

Running ``python -m src.run_pipeline`` reproduces every number quoted in the README, the
executive summary, the risk memo and both dashboards. Nothing in the written deliverables
is typed by hand; they read from ``outputs/tables/results.json``, which this script
writes.

Order of operations matters for a defensible result:

1. models are selected on the **validation** fold;
2. probabilities are calibrated on the **validation** fold;
3. the decision cutoff is chosen on the **validation** fold;
4. the **test** fold is scored once, at the end, and is what every reported figure uses.
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from . import config as C
from . import evaluate as E
from . import explainability as X
from . import plots as P
from . import scenarios as S
from .data_prep import BEHAVIOURAL, FEATURE_SETS, build_dataset, load_splits
from .expected_loss import (
    account_level_el,
    decision_rule_evaluation,
    el_by_group,
    el_by_risk_band,
    portfolio_summary,
)
from .models import load_model, train_all

CHAMPION_CANDIDATES = [
    "Gradient boosting (XGBoost)",
    "Gradient boosting (LightGBM)",
    "Random forest",
    "Logistic regression",
]

TARGET_BAD_RATE = 0.15

SEGMENT_COLS = ["age_band", "limit_band", "education_label", "marriage_label", "sex_label"]


def _save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(C.TABLES / f"{name}.csv", index=False)


def _predict(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(frame[features])[:, 1]


def main(retrain: bool = True) -> dict:
    results: dict = {"generated_on": date.today().isoformat()}

    # ---------------------------------------------------------------- data -------------
    print("[1/8] Preparing data")
    splits, prep_report = build_dataset(save=True)
    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    results["data"] = prep_report

    # ---------------------------------------------------------------- models ----------
    print("[2/8] Training models")
    manifests = {}
    for fs_name in ("behavioural", "origination"):
        manifests[fs_name] = (
            train_all(fs_name) if retrain
            else json.loads((C.MODELS / f"manifest_{fs_name}.json").read_text())
        )
    results["training"] = manifests

    fs = BEHAVIOURAL
    feats = fs.all
    model_names = [m for m in C.MODEL_ORDER if m in manifests["behavioural"]["models"]]

    # ---------------------------------------------------------------- evaluation ------
    print("[3/8] Evaluating on the test fold")
    y_val = val[C.TARGET].to_numpy()
    y_test = test[C.TARGET].to_numpy()

    val_probs, test_probs, test_probs_uncal = {}, {}, {}
    for name in model_names:
        cal = load_model("behavioural", name, calibrated=True)
        raw = load_model("behavioural", name, calibrated=False)
        val_probs[name] = _predict(cal, val, feats)
        test_probs[name] = _predict(cal, test, feats)
        test_probs_uncal[name] = _predict(raw, test, feats)

    # Champion selected on validation ROC-AUC among the non-trivial models.
    champion = max(
        (m for m in CHAMPION_CANDIDATES if m in val_probs),
        key=lambda m: E.core_metrics(y_val, val_probs[m])["roc_auc"],
    )
    results["champion_model"] = champion
    print(f"      champion: {champion}")

    # Cutoffs are chosen on validation, never on test.
    val_exposure = account_level_el(val, val_probs[champion])["ead"].to_numpy()
    val_revenue_base = val["BILL_AMT1"].clip(lower=0).to_numpy()
    cost_cutoff, cutoff_curve_df = E.cost_optimal_threshold(
        y_val, val_probs[champion], val_exposure, revenue_base=val_revenue_base
    )
    f1_cutoff = E.best_f1_threshold(y_val, val_probs[champion])
    policy_cutoff = E.threshold_for_target_bad_rate(
        y_val, val_probs[champion], target_bad_rate=TARGET_BAD_RATE
    )
    margin_sens = E.margin_sensitivity(
        y_val, val_probs[champion], val_exposure, val_revenue_base
    )
    _save_table(margin_sens, "cutoff_margin_sensitivity")
    results["thresholds"] = {
        "policy_target_bad_rate": TARGET_BAD_RATE,
        "policy_cutoff_from_validation": policy_cutoff,
        "cost_optimal_from_validation": cost_cutoff,
        "f1_optimal_from_validation": f1_cutoff,
        "naive_default": 0.5,
        "margin_sensitivity": margin_sens.to_dict(orient="records"),
    }
    _save_table(cutoff_curve_df, "cutoff_cost_curve_validation")

    comparison = pd.DataFrame(
        [E.evaluate_model(n, y_test, test_probs[n], policy_cutoff) for n in model_names]
    )
    comparison.insert(
        1, "val_auc", [E.core_metrics(y_val, val_probs[n])["roc_auc"] for n in model_names]
    )
    comparison["brier_uncalibrated"] = [
        E.core_metrics(y_test, test_probs_uncal[n])["brier"] for n in model_names
    ]
    comparison["ece_uncalibrated"] = [
        E.core_metrics(y_test, test_probs_uncal[n])["ece"] for n in model_names
    ]
    _save_table(comparison, "model_comparison")
    results["model_comparison"] = comparison.to_dict(orient="records")

    # Behavioural versus origination-only: what the repayment history is worth.
    leakage_rows = []
    for fs_name in ("behavioural", "origination"):
        f = FEATURE_SETS[fs_name].all
        for name in ("Logistic regression", "Gradient boosting (XGBoost)"):
            if name not in manifests[fs_name]["models"]:
                continue
            m = load_model(fs_name, name, calibrated=True)
            p = _predict(m, test, f)
            met = E.core_metrics(y_test, p)
            leakage_rows.append(
                {
                    "feature_set": fs_name,
                    "model": name,
                    "n_features": len(f),
                    "roc_auc": met["roc_auc"],
                    "gini": met["gini"],
                    "pr_auc": met["pr_auc"],
                    "ks": met["ks"],
                    "brier": met["brier"],
                }
            )
    leakage = pd.DataFrame(leakage_rows)
    _save_table(leakage, "feature_set_comparison")
    results["feature_set_comparison"] = leakage.to_dict(orient="records")

    p_test = test_probs[champion]

    at_cutoffs = {
        f"risk_appetite_{TARGET_BAD_RATE:.0%}_bad_rate": E.threshold_metrics(y_test, p_test, policy_cutoff),
        "f1_optimal": E.threshold_metrics(y_test, p_test, f1_cutoff),
        "cost_minimising": E.threshold_metrics(y_test, p_test, cost_cutoff),
        "naive_0.5": E.threshold_metrics(y_test, p_test, 0.5),
    }
    results["champion_at_cutoffs"] = at_cutoffs
    _save_table(pd.DataFrame(at_cutoffs).T.reset_index(names="cutoff_rule"), "cutoff_comparison")

    deciles = E.decile_table(y_test, p_test, exposure=account_level_el(test, p_test)["ead"].to_numpy())
    _save_table(deciles, "risk_deciles")
    results["deciles"] = deciles.to_dict(orient="records")

    cal_champ = E.calibration_table(y_test, p_test)
    cal_uncal = E.calibration_table(y_test, test_probs_uncal[champion])
    _save_table(cal_champ, "calibration_champion")
    _save_table(cal_uncal, "calibration_champion_uncalibrated")

    segperf = E.segment_performance(test, p_test, SEGMENT_COLS)
    _save_table(segperf, "segment_performance")
    results["segment_performance"] = segperf.to_dict(orient="records")

    stability = X.stability_check(
        load_model("behavioural", champion, calibrated=True), splits, feats
    )
    psi = X.population_stability_index(
        _predict(load_model("behavioural", champion, calibrated=True), train, feats), p_test
    )
    stability["psi_train_vs_test"] = psi
    _save_table(stability, "stability")
    results["stability"] = {"by_split": stability.to_dict(orient="records"), "psi_train_vs_test": psi}

    # ---------------------------------------------------------------- expected loss ---
    print("[4/8] Expected loss and scenarios")
    el_base = account_level_el(test, p_test, lgd=C.LGD_BASELINE, ccf=C.CCF_BASELINE)
    base_summary = portfolio_summary(el_base)
    results["expected_loss_baseline"] = base_summary
    _save_table(el_by_risk_band(el_base), "el_by_risk_decile")

    scen_table, scen_frames = S.run_all_scenarios(test, p_test)
    _save_table(scen_table, "scenario_summary")
    results["scenarios"] = scen_table.to_dict(orient="records")

    sens = S.sensitivity_grid(test, p_test)
    _save_table(sens, "sensitivity_grid")

    seg_el_tables = {}
    for col in SEGMENT_COLS:
        t = el_by_group(el_base, col)
        seg_el_tables[col] = t
        _save_table(t, f"el_by_{col}")
    results["el_by_segment"] = {k: v.to_dict(orient="records") for k, v in seg_el_tables.items()}

    scen_seg = S.scenario_by_segment(scen_frames, "limit_band")
    _save_table(scen_seg, "scenario_by_limit_band")

    # Account-level scored file: the input the dashboards read.
    scored_cols = [
        C.ID_COL, C.TARGET, "pd", "ead", "expected_loss", "LIMIT_BAL", "BILL_AMT1",
        "AGE", "age_band", "limit_band", "sex_label", "education_label", "marriage_label",
        "dpd_max", "months_delinquent", "util_latest",
    ]
    scored = el_base[[c for c in scored_cols if c in el_base.columns]].copy()
    scored["risk_decile"] = pd.qcut(
        scored["pd"].rank(method="first"), C.N_RISK_BANDS, labels=range(1, C.N_RISK_BANDS + 1)
    ).astype(int)
    scored.to_csv(C.TABLES / "scored_test_accounts.csv", index=False)

    decision = decision_rule_evaluation(el_base, policy_cutoff)
    decision_cost_rule = decision_rule_evaluation(el_base, cost_cutoff)
    results["decision_rule_cost_minimising"] = decision_cost_rule
    _save_table(pd.DataFrame([decision_cost_rule]), "decision_rule_cost_minimising_test")
    results["decision_rule"] = decision
    _save_table(pd.DataFrame([decision]), "decision_rule_test")

    # ---------------------------------------------------------------- explainability --
    print("[5/8] Explainability")
    logit = load_model("behavioural", "Logistic regression", calibrated=False)
    odds = X.logistic_odds_ratios(logit)
    _save_table(odds, "logistic_odds_ratios")
    results["odds_ratios_top"] = odds.head(12).to_dict(orient="records")

    champ_raw = load_model("behavioural", champion, calibrated=False)
    perm = X.permutation_importances(champ_raw, test, feats, n_repeats=8)
    _save_table(perm, "permutation_importance")
    results["permutation_importance_top"] = perm.head(12).to_dict(orient="records")

    shap_table = None
    if X.HAS_SHAP:
        try:
            vals, X_df, sub, _ = X.shap_values_for_tree_model(champ_raw, test, feats, sample=2000)
            shap_table = X.shap_global_importance(vals, X_df)
            _save_table(shap_table, "shap_global_importance")
            results["shap_top"] = shap_table.head(12).to_dict(orient="records")

            order = np.argsort(sub.index.map(lambda i: p_test[test.index.get_loc(i)]))
            riskiest, safest = int(order[-1]), int(order[0])
            _save_table(X.reason_codes(vals, X_df, riskiest), "reason_codes_high_risk_account")
            _save_table(X.reason_codes(vals, X_df, safest), "reason_codes_low_risk_account")
        except Exception as exc:  # pragma: no cover
            print(f"      SHAP skipped: {exc}")

    # ---------------------------------------------------------------- figures ---------
    print("[6/8] Figures")
    full = pd.concat([train, val, test], ignore_index=True)
    figs = {}
    figs["roc"] = P.roc_curves(
        {n: (y_test, test_probs[n]) for n in model_names if n != "Baseline (prior)"},
        {n: E.core_metrics(y_test, test_probs[n])["roc_auc"] for n in model_names},
    )
    figs["pr"] = P.pr_curves(
        {n: (y_test, test_probs[n]) for n in model_names if n != "Baseline (prior)"},
        {n: E.core_metrics(y_test, test_probs[n])["pr_auc"] for n in model_names},
        prevalence=float(y_test.mean()),
    )
    figs["model_comparison"] = P.model_comparison_bars(
        comparison[comparison["model"] != "Baseline (prior)"]
    )
    figs["scores"] = P.score_distributions(y_test, p_test)
    figs["calibration"] = P.calibration_plot(
        {f"{champion} - calibrated": cal_champ, f"{champion} - raw output": cal_uncal}
    )
    figs["deciles"] = P.decile_default_rates(deciles)
    figs["confusion"] = P.confusion_matrix_plot(at_cutoffs[f"risk_appetite_{TARGET_BAD_RATE:.0%}_bad_rate"])
    figs["cutoff"] = P.cutoff_curve(cutoff_curve_df, cost_cutoff, policy_cutoff,
                                   base_margin=C.MARGIN_ON_GOOD_ACCOUNT,
                                   portfolio_bad_rate=float(y_val.mean()))
    figs["el_decile"] = P.el_by_decile(el_by_risk_band(el_base))
    figs["scenarios"] = P.scenario_comparison(scen_table)
    figs["sensitivity"] = P.sensitivity_heatmap(sens)
    figs["el_limit_band"] = P.el_by_segment(seg_el_tables["limit_band"], "credit-limit quintile", "el_by_limit_band")
    figs["el_age"] = P.el_by_segment(seg_el_tables["age_band"], "age band", "el_by_age_band")
    figs["odds"] = P.odds_ratio_plot(odds)
    figs["permutation"] = P.importance_bars(
        perm, "auc_drop_mean", "What the champion model relies on",
        "Drop in test ROC-AUC when a single input is randomly permuted.",
        "Mean ROC-AUC drop", "permutation_importance",
    )
    if shap_table is not None:
        figs["shap"] = P.importance_bars(
            shap_table, "mean_abs_shap", "Global SHAP importance",
            "Mean absolute SHAP value across a 2,000-account sample of the test fold.",
            "Mean |SHAP| (log-odds)", "shap_importance",
        )
    figs["delinquency"] = P.delinquency_profile(full)
    figs["by_limit"] = P.default_rate_by_category(full, "limit_band", "Credit-limit quintile", "default_rate_by_limit")
    figs["by_age"] = P.default_rate_by_category(full, "age_band", "Age band", "default_rate_by_age")
    figs["by_education"] = P.default_rate_by_category(full, "education_label", "Education", "default_rate_by_education")
    results["figures"] = {k: v.split("outputs/")[-1] for k, v in figs.items()}

    # ---------------------------------------------------------------- headline --------
    print("[7/8] Headline numbers")
    champ_metrics = E.core_metrics(y_test, p_test)
    logreg_metrics = E.core_metrics(y_test, test_probs["Logistic regression"])
    results["headline"] = {
        "champion": champion,
        "test_accounts": int(len(test)),
        "test_default_rate": float(y_test.mean()),
        "champion_roc_auc": champ_metrics["roc_auc"],
        "champion_gini": champ_metrics["gini"],
        "champion_pr_auc": champ_metrics["pr_auc"],
        "champion_ks": champ_metrics["ks"],
        "champion_brier": champ_metrics["brier"],
        "champion_ece": champ_metrics["ece"],
        "logreg_roc_auc": logreg_metrics["roc_auc"],
        "logreg_pr_auc": logreg_metrics["pr_auc"],
        "logreg_brier": logreg_metrics["brier"],
        "auc_gain_over_logreg": champ_metrics["roc_auc"] - logreg_metrics["roc_auc"],
        "top_decile_default_rate": float(deciles.iloc[-1]["observed_default_rate"]),
        "bottom_decile_default_rate": float(deciles.iloc[0]["observed_default_rate"]),
        "top_decile_lift": float(deciles.iloc[-1]["lift"]),
        "top_two_deciles_capture": float(deciles.iloc[-2:]["defaults"].sum() / deciles["defaults"].sum()),
        "policy_cutoff": policy_cutoff,
        "policy_approval_rate": decision["approval_rate"],
        "policy_bad_rate_approved": decision["bad_rate_of_approved"],
        "policy_bad_rate_no_cutoff": decision["bad_rate_no_cutoff"],
        "policy_recall": at_cutoffs[f"risk_appetite_{TARGET_BAD_RATE:.0%}_bad_rate"]["recall"],
        "policy_precision": at_cutoffs[f"risk_appetite_{TARGET_BAD_RATE:.0%}_bad_rate"]["precision"],
        "portfolio_ead": base_summary["total_ead"],
        "portfolio_el": base_summary["total_expected_loss"],
        "portfolio_el_rate": base_summary["el_rate_on_ead"],
        "severe_el": float(scen_table.iloc[-1]["total_expected_loss"]),
        "severe_el_uplift_pct": float(scen_table.iloc[-1]["el_vs_baseline_pct"]),
        "moderate_el": float(scen_table.iloc[1]["total_expected_loss"]),
        "moderate_el_uplift_pct": float(scen_table.iloc[1]["el_vs_baseline_pct"]),
    }

    print("[8/8] Writing results.json")
    (C.TABLES / "results.json").write_text(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-retrain", action="store_true", help="reuse the models already on disk")
    args = ap.parse_args()
    r = main(retrain=not args.no_retrain)
    print("\nHeadline results")
    for k, v in r["headline"].items():
        print(f"  {k:<32} {v}")
