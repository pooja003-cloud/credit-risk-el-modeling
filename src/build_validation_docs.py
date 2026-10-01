"""Generate the model validation report and the one-page model-risk appendix.

Structure follows two published review frameworks:

* the validation report uses the Model Risk Validator layout - five review domains,
  findings with an ID and a severity, outcome analysis, limitations and compensating
  controls, and an overall risk rating;
* the appendix uses the Model Risk Governance Validator layout, built on SR 11-7 - model
  identification and tier, four pass/fail ratings, an overall opinion, conditions, and
  ongoing monitoring.

Every figure is read from ``outputs/tables/results.json``. The findings themselves are
written here, because they are judgements about the model rather than outputs of it.
"""

from __future__ import annotations

import pandas as pd

from . import config as C


def _money(x: float) -> str:
    return f"{C.CURRENCY}{x / 1e6:,.1f}m"


# Findings are kept in one place so the report and the appendix cannot disagree.
# Severity follows the Model Risk Validator scale in the report and the Governance
# Validator scale in the appendix; the mapping is fixed here.
def findings(r: dict) -> list[dict]:
    h = r["headline"]
    v = r["validation"]
    cal = v["calibration"]
    cc = v["champion_vs_challenger"]
    top = v["sensitivity_ranking"][0]
    sens = pd.read_csv(C.TABLES / "validation_input_sensitivity.csv")
    max_prop = sens.loc[sens["shock_type"] == "proportional", "mean_pd_change"].abs().max()
    fs = pd.read_csv(C.TABLES / "feature_set_comparison.csv")
    orig_auc = fs[fs["feature_set"] == "origination"]["roc_auc"].max()
    el = r["expected_loss_baseline"]

    return [
        {
            "id": "F-01", "domain": "Conceptual soundness", "severity": "High", "gov": "Major",
            "status": "Remediated (documentation)",
            "title": "Expected-loss horizon was not stated",
            "detail": (
                "The target is default in October 2005 given data to September, so every PD "
                f"is a one-month PD and the {_money(el['total_expected_loss'])} baseline is a "
                "one-month expected loss. It was presented with no horizon, inviting comparison "
                "with 12-month IFRS 9 Stage 1 or lifetime CECL figures, which it is not."
            ),
            "action": (
                "State the horizon wherever a PD or loss figure appears. A 12-month figure would "
                "need a 12-month default label or a documented term-structure assumption."
            ),
        },
        {
            "id": "F-02", "domain": "Conceptual soundness", "severity": "High", "gov": "Major",
            "status": "Open",
            "title": "Default definition and LGD assumption may not match",
            "detail": (
                "The source defines the label only as 'default payment next month' and gives no "
                f"days-past-due threshold. The {C.LGD_BASELINE:.0%} LGD is a charge-off-scale "
                "loss. If the label behaves more like a missed payment than a terminal default, "
                "some flagged accounts will cure, and a charge-off LGD overstates loss on them."
            ),
            "action": (
                "Treat the expected-loss figures as an upper bound under that reading. With "
                "account-level cure or roll-rate data, either redefine default at 90+ days past "
                "due or apply a cure-adjusted LGD."
            ),
        },
        {
            "id": "F-03", "domain": "Outcome analysis", "severity": "Medium", "gov": "Major",
            "status": "Open",
            "title": "Model is slightly over-confident, and optimistic by exposure",
            "detail": (
                f"Calibration slope is {cal['calibration_slope']:.2f} (1.0 is ideal): PDs are too "
                "low in the safest bands and too high in the riskiest. By account count the model "
                f"is conservative (actual-to-expected {cal['ae_ratio_count_weighted']:.3f}). By "
                f"exposure it is optimistic ({cal['ae_ratio_exposure_weighted']:.3f}), because the "
                "under-predicted low-risk accounts carry the largest balances. Portfolio expected "
                f"loss is therefore understated by about "
                f"{1 - 1 / cal['ae_ratio_exposure_weighted']:.1%} of the realised figure."
            ),
            "action": (
                "Refit calibration with a slope term, or hold a documented margin of "
                "conservatism. Monitor the exposure-weighted actual-to-expected ratio, not only "
                "the count-weighted one."
            ),
        },
        {
            "id": "F-04", "domain": "Outcome analysis", "severity": "Low", "gov": "Minor",
            "status": "Remediated (documentation)",
            "title": "Expected-loss back-test claimed more than it tests",
            "detail": (
                "Documentation said the back-test checked PD and EAD against outcomes. The same "
                "EAD formula and LGD appear on both sides of the comparison and cancel, so it "
                "checks only exposure-weighted PD calibration. Its "
                f"{1 - 1 / cal['ae_ratio_exposure_weighted']:.1%} gap is the F-03 effect."
            ),
            "action": "Wording corrected in the README and risk memo.",
        },
        {
            "id": "F-05", "domain": "Sensitivity and stress testing", "severity": "Medium",
            "gov": "Minor", "status": "Open",
            "title": "PD is concentrated on one input",
            "detail": (
                f"Adding one month to latest arrears status moves mean PD by "
                f"{top['mean_pd_change']:+.0%}. No proportional shock to a continuous driver, "
                f"even at 50%, moves it by more than {max_prop:.1%}. Errors in the latest "
                "repayment-status field would pass almost directly into PD and loss."
            ),
            "action": (
                "Put data-quality controls on the repayment-status feed and monitor that "
                "field's distribution separately from the overall score."
            ),
        },
        {
            "id": "F-06", "domain": "Implementation verification", "severity": "Low",
            "gov": "Observation", "status": "Open",
            "title": "Champion and challenger agree in aggregate, not account by account",
            "detail": (
                f"Portfolio expected loss differs by {cc['el_relative_gap']:.1%} and mean PD by "
                f"{cc['mean_pd_relative_gap']:.1%}, inside the 5% tolerance. But the average "
                f"account-level gap is {cc['account_mean_abs_gap_pp']:.1f} percentage points and "
                f"{1 - cc['share_within_5pp']:.0%} of accounts differ by more than 5 points."
            ),
            "action": (
                "Model choice matters for account-level decisions even though it barely moves "
                "portfolio loss. Document the champion choice on account-level grounds."
            ),
        },
        {
            "id": "F-07", "domain": "Sensitivity and stress testing", "severity": "Low",
            "gov": "Minor", "status": "Remediated (documentation)",
            "title": "Scenario narratives cited figures the model does not use",
            "detail": (
                "Narratives referred to specific unemployment moves. No macroeconomic variable "
                "is modelled and nothing links those figures to the multipliers."
            ),
            "action": "Narratives relabelled as hypothetical, with severity stated as judgemental.",
        },
        {
            "id": "F-08", "domain": "Data integrity", "severity": "High", "gov": "Major",
            "status": "Open",
            "title": "No out-of-time validation is possible",
            "detail": (
                "Every account shares one six-month window and one outcome month. The split is "
                f"random, so the PSI of {r['stability']['psi_train_vs_test']:.4f} says nothing "
                "about stability through time."
            ),
            "action": "Re-fit and test on at least one later cohort before any production use.",
        },
        {
            "id": "F-09", "domain": "Conceptual soundness", "severity": "High", "gov": "Major",
            "status": "Open",
            "title": "Protected characteristics are model inputs",
            "detail": (
                "Sex, education and marital status are inputs to the champion model, not only "
                "segment labels. Several are prohibited or restricted for credit decisions in "
                "many jurisdictions."
            ),
            "action": (
                "Drop them, re-fit, measure the discrimination lost, and test the remaining "
                "behavioural inputs for disparate impact."
            ),
        },
        {
            "id": "F-10", "domain": "Governance", "severity": "Critical", "gov": "Critical",
            "status": "Open",
            "title": "Validation is not independent of development",
            "detail": (
                "These tests were run by the model's developer. Both frameworks require "
                "independent validation, so this is a structured self-assessment."
            ),
            "action": "Restrict use to research and reporting until independently validated.",
        },
        {
            "id": "F-11", "domain": "Data integrity", "severity": "Low", "gov": "Observation",
            "status": "Open",
            "title": "Population conditioned on survival",
            "detail": (
                "Only accounts active across the whole window are present. Accounts closed or "
                "charged off earlier are missing, so PDs are conditional on surviving to "
                "September 2005."
            ),
            "action": "Appropriate for a behavioural score; state it as a scope limit.",
        },
        {
            "id": "F-12", "domain": "Conceptual soundness", "severity": "Informational",
            "gov": "Observation", "status": "Open",
            "title": "Simplifications in EAD and LGD",
            "detail": (
                "EAD uses the September statement balance as the drawn balance, so payments "
                "since the statement are not reflected. LGD is one figure for every segment. "
                f"Origination-only AUC of {orig_auc:.2f} confirms the behavioural scope."
            ),
            "action": "Acceptable as documented assumptions; revisit with recovery data.",
        },
    ]


def build_validation_report(r: dict) -> str:
    h = r["headline"]
    v = r["validation"]
    hl = v["hosmer_lemeshow"]
    cal = v["calibration"]
    b = v["binomial"]
    cc = v["champion_vs_challenger"]
    lin = v["lineage"]
    F = findings(r)

    binom = pd.read_csv(C.TABLES / "validation_binomial_by_decile.csv")
    binom_md = "\n".join(
        ["| Decile | Accounts | Observed | Expected | 99% interval | Result |",
         "| --- | --- | --- | --- | --- | --- |"]
        + [f"| {int(x.decile)} | {int(x.accounts)} | {int(x.observed_defaults)} | "
           f"{x.expected_defaults:.1f} | {int(x.interval_low)}-{int(x.interval_high)} | "
           f"{'Pass' if x.passes else 'Fail'} |" for x in binom.itertuples()]
    )

    rank = pd.DataFrame(v["sensitivity_ranking"])
    def _shock(row):
        if row["shock_type"] == "months":
            return f"{int(row['shock']):+d} month"
        return f"{row['shock']:+.0%}"
    sens_md = "\n".join(
        ["| Input | Largest-response shock | Change in mean PD |", "| --- | --- | --- |"]
        + [f"| {x['label']} | {_shock(x)} | {x['mean_pd_change']:+.1%} |"
           for _, x in rank.iterrows()]
    )

    def domain_block(domain: str) -> str:
        items = [f for f in F if f["domain"] == domain]
        if not items:
            return "No findings.\n"
        out = []
        for f in items:
            out.append(
                f"**{f['id']} · {f['severity']} · {f['status']}** - {f['title']}\n\n"
                f"{f['detail']}\n\n*Recommendation:* {f['action']}\n"
            )
        return "\n".join(out)

    sev_order = ["Critical", "High", "Medium", "Low", "Informational"]
    counts = {s: sum(f["severity"] == s for f in F) for s in sev_order}
    count_line = ", ".join(f"{n} {s.lower()}" for s, n in counts.items() if n)

    checklist = [
        ("Target definition", "Default clearly defined, appropriate horizon",
         "Partly. Label is 'default payment next month' (one-month horizon). No days-past-due "
         "threshold in the source.", "F-01, F-02"),
        ("Feature set", "No post-decision variables",
         "Met. All features dated September 2005 or earlier. Origination-only model built to "
         "measure the behavioural dependency.", "-"),
        ("Model selection", "Credible interpretable baseline",
         f"Met. Logistic regression benchmark at {h['logreg_roc_auc']:.3f} ROC-AUC; boosting adds "
         f"{h['auc_gain_over_logreg']:+.4f}.", "F-06"),
        ("Metrics", "Appropriate for an imbalanced target",
         "Met. ROC-AUC, Gini, KS, PR-AUC, precision, recall, F1, Brier, calibration error, "
         "decile table. Accuracy not used as a headline.", "-"),
        ("Calibration", "Predicted PDs match observed rates",
         f"Met with a caveat. 10/10 deciles pass binomial at 99%; slope {cal['calibration_slope']:.2f}; "
         "optimistic by exposure.", "F-03"),
        ("Expected loss", "PD, LGD, EAD consistent and documented",
         "Partly. Formulas and assumptions in code and docs. Horizon and default-definition "
         "consistency needed stating.", "F-01, F-02, F-04"),
        ("Scenarios", "Internally consistent, labelled hypothetical",
         "Met after review. All three components move together; narratives relabelled.", "F-07"),
        ("Model risk", "Limitations, bias, drift, monitoring",
         "Met. Model card, this report, and the one-page appendix.", "F-08 to F-11"),
        ("Business use", "Not overstated",
         "Met. Cutoff evaluated retrospectively; no claim of improved lending decisions.", "-"),
    ]
    checklist_md = "\n".join(
        ["| Area | Check | Finding | Ref |", "| --- | --- | --- | --- |"]
        + [f"| {a} | {b_} | {c} | {d} |" for a, b_, c, d in checklist]
    )

    return f"""# Model validation report

**Model:** Consumer card behavioural PD model, v1.0 ({h['champion']}, calibrated)
**Scope:** Conceptual soundness, data integrity, implementation, outcome analysis,
sensitivity and stress testing
**Basis:** {h['test_accounts']:,}-account held-out test fold
**Overall model risk:** Moderate for research and portfolio reporting. Would be a High-tier
model if used for credit decisions.

> **Independence.** These tests were run by the model's developer. Both review frameworks this
> report follows require validation to be independent of development, so read it as a
> structured self-assessment, not a validation opinion. See F-10.

---

## 1. Executive summary

The model discriminates well ({h['champion_roc_auc']:.3f} ROC-AUC against a 0.70 threshold),
passes the binomial backtest in all {b['bands_total']} risk deciles at 99% confidence, and its
inputs trace exactly to the raw file. The data pipeline is sound.

The material issues are in what the outputs mean rather than how they are computed. Expected
loss had no stated horizon and is a one-month figure (F-01). The default label is not defined
by days past due, so the charge-off LGD may overstate loss (F-02). And the model is slightly
over-confident, which makes it optimistic by exposure even though it is conservative by
account count (F-03).

{len(F)} findings: {count_line}. Three were documentation errors and are corrected in this
release (F-01, F-04, F-07).

## 2. Model overview

A gradient-boosted tree model predicts the probability that a credit-card account defaults in
the following month, from six months of billing, payment and repayment-status history plus
the credit limit and three demographic fields. Probabilities are calibrated on a validation
fold, floored at {C.PD_FLOOR:.2%} and capped at {C.PD_CAP:.2%}. A logistic regression on the
same inputs is the challenger. Full specification: [`model_card.md`](model_card.md).

### Post-completion audit checklist

{checklist_md}

## 3. Findings by domain

### 3.1 Conceptual soundness

{domain_block("Conceptual soundness")}
### 3.2 Data integrity

A 25-record trace rebuilt {lin['features_checked']} model inputs from the raw file with
independent arithmetic and compared them to what the model received. Worst discrepancy:
{lin['worst_relative_discrepancy']:.2%} against a 0.5% tolerance. The loader also checks the
file against the published row, default and default-rate counts before anything runs.

{domain_block("Data integrity")}
### 3.3 Implementation verification

The logistic-regression challenger was scored on identical inputs. Portfolio expected loss
differs by {cc['el_relative_gap']:.1%} and mean PD by {cc['mean_pd_relative_gap']:.1%}, inside
the 5% tolerance. Rank correlation between the two is {cc['rank_correlation']:.2f}. Unit tests
cover the exposure, expected-loss, stress and validation arithmetic, and a separate script
recomputes every headline figure from the scored file without using the project's own code.

{domain_block("Implementation verification")}
### 3.4 Outcome analysis

{domain_block("Outcome analysis")}
### 3.5 Sensitivity and stress testing

{domain_block("Sensitivity and stress testing")}
### 3.6 Governance

{domain_block("Governance")}
## 4. Outcome analysis results

**Discrimination.** ROC-AUC {h['champion_roc_auc']:.3f}, Gini {h['champion_gini']:.3f}, KS
{h['champion_ks']:.3f}. Above the 0.70 AUC threshold.

**Binomial backtest by decile, 99% confidence.** {b['bands_passing']} of {b['bands_total']} pass.

{binom_md}

**Hosmer-Lemeshow.** Statistic {hl['statistic']:.2f}. p = {hl['p_value_out_of_sample_df']:.3f}
with {hl['df_out_of_sample']} degrees of freedom, the usual choice for a model scored out of
sample, and p = {hl['p_value_in_sample_df']:.3f} with the in-sample convention of
{hl['df_in_sample']}. Borderline: it passes at 5% under the appropriate convention and fails
under the stricter one. With about 600 accounts per group the test has high power, so the
group table matters more than the verdict, and it shows the pattern in F-03.

**Calibration diagnostics.**

| Measure | Value |
| --- | --- |
| Calibration slope (ideal 1.0) | {cal['calibration_slope']:.3f} |
| Calibration intercept (ideal 0.0) | {cal['calibration_intercept']:.3f} |
| Predicted / observed rate, by account | {cal['predicted_rate_count_weighted']:.2%} / {cal['observed_rate_count_weighted']:.2%} |
| Predicted / observed rate, by exposure | {cal['predicted_rate_exposure_weighted']:.2%} / {cal['observed_rate_exposure_weighted']:.2%} |
| Actual-to-expected, by account | {cal['ae_ratio_count_weighted']:.3f} |
| Actual-to-expected, by exposure | {cal['ae_ratio_exposure_weighted']:.3f} |

**Input sensitivity.** Each driver shocked alone, by +/-10%, 20% and 50% for continuous inputs
and +/-1 month for delinquency counts. Largest response per input:

{sens_md}

Single-input shocks produce rows that are not internally consistent, so these show how much
the model leans on each input rather than describing a plausible scenario.

## 5. Limitations and compensating controls

| Limitation | Compensating control in place |
| --- | --- |
| No out-of-time test (F-08) | Stratified split with validation-only model selection; test fold scored once |
| LGD not estimable (F-02, F-12) | LGD stated as an assumption everywhere; full PD-stress by LGD grid published |
| Calibration slope below 1 (F-03) | Binomial pass in every decile; exposure-weighted A/E reported alongside |
| Input concentration (F-05) | Input traced to raw in the lineage check; sensitivity published |
| Protected inputs (F-09) | Not approved for decisioning; disparate-impact testing named as a condition |
| Not independent (F-10) | Use restricted to research and reporting |

## 6. Conclusion and risk rating

Model risk is **Moderate** for its intended use: research and portfolio-level reporting of
one-month expected loss. The model is **not fit for credit decisioning** in its current form,
because of F-09 and F-10, and would be a High-tier model if it were used that way.

The one-page summary of conditions and monitoring is in
[`model_risk_appendix.md`](model_risk_appendix.md).
"""


def build_model_risk_appendix(r: dict) -> str:
    h = r["headline"]
    v = r["validation"]
    cal = v["calibration"]
    b = v["binomial"]
    lin = v["lineage"]
    F = findings(r)
    th = r["thresholds"]

    window = {"Critical": "Before any use beyond research", "Major": "90 days",
              "Minor": "180 days", "Observation": "No action required"}
    conds = [f for f in F if f["gov"] in ("Critical", "Major", "Minor") and not f["status"].startswith("Remediated")]
    cond_md = "\n".join(
        ["| ID | Severity | Condition | Remediation window |", "| --- | --- | --- | --- |"]
        + [f"| {f['id']} | {f['gov']} | {f['title']} | {window[f['gov']]} |" for f in conds]
    )

    return f"""# Model risk and limitations appendix

**Model:** Consumer card behavioural PD model, v1.0 · **Use:** one-month expected loss for
research and portfolio reporting · **Tier:** would be High if used for credit decisions ·
**Basis:** {h['test_accounts']:,}-account held-out test fold

## Ratings

| Area | Rating | Basis |
| --- | --- | --- |
| Conceptual soundness | Pass with conditions | Sound method; horizon now stated, default definition open (F-01, F-02) |
| Data quality | Pass | {lin['records']}-record lineage trace, {lin['worst_relative_discrepancy']:.1%} discrepancy; integrity checks on load |
| Performance testing | Pass | ROC-AUC {h['champion_roc_auc']:.3f} vs 0.70; {b['bands_passing']}/{b['bands_total']} deciles pass binomial at 99%; calibration slope {cal['calibration_slope']:.2f} |
| Governance | Not met | Self-assessed, not independent; no inventory, owner or approval (F-10) |

**Opinion: Approved with conditions** for research and portfolio-level reporting. **Not
approved** for credit decisioning.

## Open conditions

{cond_md}

Remediation windows follow the framework's severity scale and would run from the date of any
production approval. F-01, F-04 and F-07 were documentation issues, corrected in this release.

## What the outputs are, and are not

- **One-month figures.** Every PD and the {_money(r['expected_loss_baseline']['total_expected_loss'])}
  expected loss cover the month after September 2005. They are not 12-month or lifetime numbers.
- **Probably an upper bound on loss.** Default is not defined by days past due, and LGD is an
  assumed {C.LGD_BASELINE:.0%}. If some defaults cure, true loss is lower, possibly by a lot.
- **Slightly optimistic by exposure,** which pushes the other way but is small: actual-to-expected
  is {cal['ae_ratio_exposure_weighted']:.3f} by exposure against {cal['ae_ratio_count_weighted']:.3f}
  by account, worth about {1 - 1 / cal['ae_ratio_exposure_weighted']:.1%} of loss.
- **Behavioural only.** Needs six months of repayment history; cannot score new applicants.
- **One cohort, Taiwan 2005**, after a domestic card-debt crisis. Not transferable to another book.

## Ongoing monitoring

| Indicator | Frequency | Trigger |
| --- | --- | --- |
| Population stability index, score | Monthly | Investigate above 0.10, revalidate above 0.25 |
| Stability of latest repayment status (F-05) | Monthly | Shift in distribution above 0.10 PSI |
| Actual-to-expected by exposure (F-03) | Quarterly | Outside 0.95-1.05 |
| Calibration slope | Quarterly | Below 0.85 or above 1.15 |
| Observed vs predicted by decile | Quarterly | Any decile outside its 99% binomial interval |
| Approved-book default rate | Monthly | Above the {th['policy_target_bad_rate']:.0%} appetite target |
| Champion vs challenger expected loss | Quarterly | Gap above 5% |

**Revalidation:** annually if deployed at High tier, and on any change to inputs, default
definition, calibration or cutoff.

Full findings: [`model_validation_report.md`](model_validation_report.md).
"""
