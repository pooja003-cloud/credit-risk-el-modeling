"""Generate and execute the technical notebook.

The notebook is written from this script rather than by hand so that it stays in step with
``src/`` - it calls the same modules the pipeline does instead of re-implementing anything,
which is the only way a notebook and a codebase can be trusted to agree.

    python tools/make_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_credit_risk_modelling.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELLS = [
    md("""# Consumer Loan Credit Risk Modeling and Expected Loss Analysis

**Technical walkthrough.** Probability-of-default modelling, expected credit loss and scenario
analysis on the UCI *Default of Credit Card Clients* dataset (Taiwan, 2005).

This notebook is the readable version of the pipeline. It calls the same modules in `src/`
that `python -m src.run_pipeline` calls, so nothing here can drift away from the code that
produces the repository's published numbers.

**Contents**

1. The question, and what would count as an answer
2. Data integrity and cleaning
3. What the data says before any model
4. The leakage boundary: behavioural versus origination
5. The model ladder
6. Evaluation: discrimination, calibration, and the two errors
7. Expected loss: `EL = PD x LGD x EAD`
8. Scenario analysis
9. What drives the score
10. Limitations"""),

    md("""## 1. The question, and what would count as an answer

**Business question.** Can borrower and account characteristics be used to estimate the
probability of default, rank accounts by risk, and estimate expected credit losses under
different scenarios?

Three things have to be true for the answer to be useful:

1. The model must **rank** accounts (ROC-AUC, Gini, KS, and a monotone decile table).
2. The probabilities must be **calibrated**, because expected loss multiplies PD *levels* -
   a model that ranks perfectly but sits 30% low produces a loss number that is 30% low.
3. The result must survive being **turned into a decision**, with the two errors priced
   differently: declining a good borrower costs forgone margin, approving a bad one costs
   `LGD x EAD`.

Accuracy is not on that list. With a 22% default rate, approving everyone is 78% accurate."""),

    code("""import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as C
from src import data_prep, evaluate as E, expected_loss as EL, scenarios as S, explainability as X
from src.models import load_model

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)
print("Working from:", ROOT)"""),

    md("""## 2. Data integrity and cleaning

Before anything else, the loaded file is checked against the statistics published for the UCI
dataset. If a mirror has been truncated or altered, the pipeline stops here rather than
quietly training on the wrong data."""),

    code("""raw = data_prep.load_raw()
integrity = data_prep.validate_raw(raw)
integrity"""),

    md("""30,000 accounts, 6,636 defaults, a 22.12% default rate, no missing values - matching the
published figures exactly.

Two things the raw file does contain: 35 rows that are exact duplicates once `ID` is removed,
and several undocumented category codes."""),

    code("""cleaned, clean_log = data_prep.clean(raw)
clean_log"""),

    md("""**The decisions, and why:**

- **Duplicates dropped.** Identical accounts landing in both the training and test folds would
  inflate measured performance.
- **Undocumented `EDUCATION` (0, 5, 6) and `MARRIAGE` (0) codes collapsed into "Other."**
  Treating an undocumented code as its own risk level invents signal that is not there.
- **Negative bill amounts kept.** These are credit balances from overpayments, not errors.
  Utilisation features floor them at zero.
- **`PAY_0` renamed `PAY_1`** so repayment status lines up with the bill and payment columns.

One subtlety that matters more than it looks. The `PAY_*` scale runs -2 (no usage), -1 (paid in
full), 0 (revolving, minimum paid), then 1-8 months past due. It is **not monotone in risk**:
-2 and -1 are qualitatively different states, not smaller amounts of delinquency. So the raw
codes are never fed to a model as a continuous variable - they are split into non-negative
months-past-due plus explicit counts of revolving, full-payment and inactive months."""),

    code("""featured = data_prep.engineer_features(cleaned)
splits = data_prep.split_data(featured)
train, val, test = splits["train"], splits["val"], splits["test"]

pd.DataFrame({
    "rows": {k: len(v) for k, v in splits.items()},
    "defaults": {k: int(v[C.TARGET].sum()) for k, v in splits.items()},
    "default rate": {k: v[C.TARGET].mean() for k, v in splits.items()},
}).style.format({"default rate": "{:.2%}", "rows": "{:,}", "defaults": "{:,}"})"""),

    md("""A stratified 60 / 20 / 20 split. **A time-based split is not possible here** - every account
is observed over the same April-September 2005 window with the same October outcome month, so
there is no time dimension to split on. That is a real limitation, recorded in the model card
rather than papered over.

The folds have distinct jobs:

- **train** - fit the models;
- **validation** - choose hyper-parameters, calibrate probabilities, set the decision cutoff;
- **test** - scored once, at the end. Every number reported below comes from it."""),

    md("""## 3. What the data says before any model

Worth looking at the portfolio directly before fitting anything, if only to know what the model
should be finding."""),

    code("""full = pd.concat(splits.values(), ignore_index=True)

delinq = full.copy()
delinq["worst_arrears"] = delinq["dpd_max"].clip(upper=5)
summary = delinq.groupby("worst_arrears").agg(
    accounts=(C.TARGET, "size"), default_rate=(C.TARGET, "mean")
)
summary["share_of_portfolio"] = summary["accounts"] / summary["accounts"].sum()
summary.style.format({"accounts": "{:,}", "default_rate": "{:.1%}", "share_of_portfolio": "{:.1%}"})"""),

    md("""This single table is most of the story. Accounts that were never late in six months default
at roughly one in eight; accounts two or more cycles down default at over two in three. Any
model that does not find this is broken.

The demographic splits are much weaker by comparison."""),

    code("""for col in ["limit_band", "age_band", "education_label"]:
    g = full.groupby(col, observed=True)[C.TARGET].agg(["size", "mean"])
    g.columns = ["accounts", "default_rate"]
    print(f"\\n{col}")
    print(g.assign(default_rate=lambda d: d.default_rate.map("{:.1%}".format)).to_string())"""),

    md("""Credit-limit quintile separates risk (low-limit accounts default far more), but that is
largely the issuer's own past risk assessment showing through - the limit was set by an earlier
underwriting decision. Age and education move the rate far less."""),

    md("""## 4. The leakage boundary: behavioural versus origination

The obvious leakage check passes trivially: the target is October 2005 and every feature is
dated September 2005 or earlier.

The harder question is **which decision** the model supports. A model with six months of
repayment history is a *behavioural* scorecard - the tool for re-underwriting, limit management
and collections on an existing account. It is not an *application* scorecard, because at
origination none of that history exists.

So both are built, and both are reported. Quoting the behavioural AUC as though it applied at
account opening would be leakage relative to the decision being made."""),

    code("""for name, fs in data_prep.FEATURE_SETS.items():
    print(f"=== {name} ({len(fs.all)} features) ===")
    print(fs.rationale)
    print("features:", ", ".join(fs.all))
    print()"""),

    code("""fs_comparison = pd.read_csv(C.TABLES / "feature_set_comparison.csv")
fs_comparison.style.format({c: "{:.4f}" for c in ["roc_auc", "gini", "pr_auc", "ks", "brier"]})"""),

    md("""Removing the repayment history costs roughly 0.15 of ROC-AUC. The origination-only model is
only modestly better than chance, and that gap is the honest measure of how much of this
model's power an application-time user would **not** get."""),

    md("""## 5. The model ladder

Five families, each fitted with a small explicit grid and selected on the validation fold:

1. **Baseline** - predicts the training default rate for everyone. No discriminatory power by
   construction; it exists to define zero.
2. **Logistic regression** - the scorecard reference. Monotone in each input, coefficients
   readable as log-odds, defensible in a credit committee.
3. **Decision tree** - splits read as policy rules.
4. **Random forest** - shows how much of the gain is ensembling rather than boosting.
5. **Gradient boosting** (LightGBM and XGBoost) - captures interactions the scorecard cannot.

Every model above the logistic regression has to justify its complexity on discrimination
*and* calibration."""),

    code("""# Models are fitted by `python -m src.models` / `python -m src.run_pipeline`; loaded here.
import json
manifest = json.loads((C.MODELS / "manifest_behavioural.json").read_text())
pd.DataFrame([
    {"model": k, "validation ROC-AUC": v["val_auc_uncalibrated"],
     "grid points searched": v["n_grid_points"], "chosen parameters": v["params"]}
    for k, v in manifest["models"].items()
])"""),

    md("""### Why calibration is a separate step

Discrimination and calibration are different properties. A model can rank every account
correctly and still be systematically wrong about the *level* of risk - and expected loss
multiplies levels, not ranks.

Probabilities are therefore calibrated on the validation fold, with the base model frozen so
calibration never refits on training data. Isotonic and Platt scaling are compared by five-fold
cross-validation *inside* the validation fold - judging a calibrator on the rows it was fitted
on would flatter the more flexible method - and the winner is refitted on the whole fold.

The calibrated probability is then floored and capped. Isotonic calibration is a step function
and will hand out a PD of exactly 0 or exactly 1 in its outermost steps; a PD of zero asserts
that no loss is possible and would zero out that account's expected loss. Supervisory
frameworks use an explicit floor for exactly this reason - Basel sets 0.03% for retail
exposures - and the same device is applied here."""),

    code("""fs = data_prep.BEHAVIOURAL
feats = fs.all
champion = json.loads((C.TABLES / "results.json").read_text())["champion_model"]

cal_model = load_model("behavioural", champion, calibrated=True)
raw_model = load_model("behavioural", champion, calibrated=False)
logit_model = load_model("behavioural", "Logistic regression", calibrated=True)

y_test = test[C.TARGET].to_numpy()
p_test = cal_model.predict_proba(test[feats])[:, 1]
p_test_raw = raw_model.predict_proba(test[feats])[:, 1]
p_logit = logit_model.predict_proba(test[feats])[:, 1]

print(f"Champion: {champion}")
print(f"mean predicted PD (calibrated):   {p_test.mean():.4f}")
print(f"mean predicted PD (uncalibrated): {p_test_raw.mean():.4f}")
print(f"observed default rate:            {y_test.mean():.4f}")"""),

    md("""## 6. Evaluation

### 6.1 Discrimination and calibration"""),

    code("""comparison = pd.read_csv(C.TABLES / "model_comparison.csv")
comparison[["model", "val_auc", "roc_auc", "pr_auc", "ks", "brier", "ece"]].style.format(
    {c: "{:.4f}" for c in ["val_auc", "roc_auc", "pr_auc", "ks", "brier", "ece"]}
)"""),

    md("""Reading this table:

- **ROC-AUC / Gini / KS** - can the model tell good from bad? Boosting wins, but not by much.
- **PR-AUC** - performance on the minority class, against a 22% base rate. This is the honest
  view for a rare event, and the gap between models is wider here than on ROC-AUC.
- **Brier / calibration error** - are the probabilities usable as levels?

**The gradient-boosting gain over the scorecard is real but small.** On this dataset a fully
interpretable logistic regression gets most of the way, and that is worth stating plainly
rather than burying: the complexity of a boosted model has to be justified by more than a
third decimal place."""),

    code("""stability = X.stability_check(cal_model, splits, feats)
psi = X.population_stability_index(cal_model.predict_proba(train[feats])[:, 1], p_test)
print(stability.to_string(index=False))
print(f"\\nPSI (train vs test score distribution): {psi:.4f}")"""),

    md("""Train ROC-AUC exceeds test by about 0.03 - mild optimism, not collapse. The PSI is
essentially zero, which is expected for a random split of a single cohort and is *not* evidence
of stability through time."""),

    code("""cal_table = E.calibration_table(y_test, p_test)
cal_table.style.format({"predicted": "{:.2%}", "observed": "{:.2%}", "gap": "{:+.2%}"})"""),

    md("""### 6.2 Rank-ordering

The decile table is the form a credit committee reads."""),

    code("""ead_test = EL.account_level_el(test, p_test)["ead"].to_numpy()
deciles = E.decile_table(y_test, p_test, exposure=ead_test)
deciles.style.format({
    "observed_default_rate": "{:.1%}", "mean_predicted_pd": "{:.1%}",
    "min_predicted_pd": "{:.1%}", "max_predicted_pd": "{:.1%}",
    "lift": "{:.2f}x", "exposure": "{:,.0f}", "share_of_exposure": "{:.1%}",
    "cumulative_defaults_captured": "{:.0%}", "cumulative_accounts": "{:.0%}",
})"""),

    md("""### 6.3 The two errors, and where to stand

`predict()` returns a 0.5 cutoff. That is a statistical convention, not a credit policy, and on
this portfolio it is the wrong place to stand:

- a **false negative** (approve, then default) costs `LGD x EAD`;
- a **false positive** (decline someone who would have paid) costs the forgone margin on the
  balance that customer would have carried.

Under the working assumptions those differ by roughly an order of magnitude."""),

    code("""results = json.loads((C.TABLES / "results.json").read_text())
th = results["thresholds"]
cutoffs = pd.read_csv(C.TABLES / "cutoff_comparison.csv")
cutoffs.style.format({
    "threshold": "{:.1%}", "flagged_rate": "{:.1%}", "precision": "{:.3f}",
    "recall": "{:.3f}", "f1": "{:.3f}", "specificity": "{:.3f}", "accuracy": "{:.3f}",
})"""),

    md("""Four candidate rules, and they disagree - which is the point.

The **cost-minimising** cutoff comes out extremely tight. That is not a discovery about credit;
it is an artefact of weighing a one-period loss against one period of margin. The sensitivity
table below shows the implied cutoff moving a long way as the assumed margin changes, which is
precisely why a cutoff should not be read off a cost calculation alone."""),

    code("""pd.read_csv(C.TABLES / "cutoff_margin_sensitivity.csv").style.format({
    "assumed_margin_on_good_account": "{:.0%}", "implied_cutoff": "{:.1%}",
    "approval_rate": "{:.1%}", "bad_rate_of_approved": "{:.2%}", "net_cost": "{:,.0f}",
})"""),

    md("""So the deployed cutoff is set the way credit policy is actually set: risk appetite is stated
as a tolerable default rate on the approved book, and the cutoff is whatever delivers it."""),

    code("""policy_cutoff = th["policy_cutoff_from_validation"]
el_base = EL.account_level_el(test, p_test, lgd=C.LGD_BASELINE, ccf=C.CCF_BASELINE)
decision = EL.decision_rule_evaluation(el_base, policy_cutoff)
pd.Series(decision).to_frame("value")"""),

    md("""**What this is, and is not.** It is a retrospective evaluation on held-out accounts: applying
this cutoff to this population would have cut the approved book's realised default rate from
22.1% to about 15%, while declining roughly 350 accounts that in fact paid.

It is **not** a claim that lending decisions were improved. No live decisions were made, and
the accounts the rule declines were extended credit in reality - so their behaviour under a
decline is unobservable."""),

    md("""## 7. Expected loss

$$EL = PD \\times LGD \\times EAD$$

| Component | Source | Status |
| --- | --- | --- |
| **PD** | Calibrated model | **Estimated** |
| **EAD** | Drawn balance + CCF x undrawn line | Balance observed, **CCF assumed** |
| **LGD** | Config constant | **Assumed** - the dataset has no recovery data |

**EAD is not just today's balance.** A card is a revolving commitment; distressed borrowers
typically draw further before they stop paying. The credit conversion factor is the assumed
share of the undrawn line drawn between now and default. Using the balance alone understates
exposure, using the full line overstates it."""),

    code("""summary = EL.portfolio_summary(el_base)
pd.Series(summary).to_frame("value")"""),

    md("""### A partial check on the loss number

The PD and EAD components can be tested against outcomes, even though LGD cannot. Holding the
LGD assumption fixed, the loss implied by the accounts that actually defaulted should be close
to the modelled expected loss."""),

    code("""modelled = summary["total_expected_loss"]
realised = summary["realised_loss_at_assumed_lgd"]
print(f"Modelled expected loss:                 {C.CURRENCY}{modelled:,.0f}")
print(f"Loss implied by actual defaults (same LGD): {C.CURRENCY}{realised:,.0f}")
print(f"Gap: {modelled / realised - 1:+.2%}")"""),

    md("""Close. That validates PD and EAD together. It says **nothing** about whether the 65% LGD
assumption is right - only that if it were right, the loss estimate would be about right."""),

    code("""by_band = EL.el_by_risk_band(el_base)
by_band.style.format({
    "mean_pd": "{:.2%}", "observed_default_rate": "{:.2%}", "total_ead": "{:,.0f}",
    "expected_loss": "{:,.0f}", "el_rate_on_ead": "{:.2%}", "share_of_ead": "{:.1%}",
    "share_of_el": "{:.1%}", "el_concentration": "{:.2f}x",
})"""),

    code("""EL.el_by_group(el_base, "limit_band").style.format({
    "mean_pd": "{:.2%}", "observed_default_rate": "{:.2%}", "total_ead": "{:,.0f}",
    "expected_loss": "{:,.0f}", "el_rate_on_ead": "{:.2%}", "share_of_ead": "{:.1%}",
    "share_of_el": "{:.1%}", "el_concentration": "{:.2f}x",
})"""),

    md("""## 8. Scenario analysis

Three scenarios, each moving PD, LGD and the credit conversion factor together - in a downturn
they do not move independently.

**How the PD stress works.** The multiplier acts on the *odds* of default, not the probability:

$$\\text{odds}' = m \\times \\text{odds}, \\qquad PD' = \\frac{\\text{odds}'}{1 + \\text{odds}'}$$

This is equivalent to shifting every score by a constant $\\log(m)$ on the log-odds scale. A
flat multiplier on the probability itself would push high-risk accounts above 1.0 and would
stress a 1% account and a 40% account by wildly different amounts in score terms."""),

    code("""for p0 in (0.01, 0.10, 0.40, 0.80):
    print(f"PD {p0:>5.0%} -> x1.5 odds -> {S.stress_pd(np.array([p0]), 1.5)[0]:.3f}"
          f"   |  x2.25 odds -> {S.stress_pd(np.array([p0]), 2.25)[0]:.3f}")"""),

    code("""scen_table, scen_frames = S.run_all_scenarios(test, p_test)
scen_table[["scenario", "pd_odds_multiplier", "lgd_assumed", "ccf_assumed", "average_pd",
            "total_ead", "total_expected_loss", "el_rate_on_ead", "el_vs_baseline_pct"]].style.format({
    "pd_odds_multiplier": "{:.2f}x", "lgd_assumed": "{:.0%}", "ccf_assumed": "{:.0%}",
    "average_pd": "{:.2%}", "total_ead": "{:,.0f}", "total_expected_loss": "{:,.0f}",
    "el_rate_on_ead": "{:.2%}", "el_vs_baseline_pct": "{:+.0%}",
})"""),

    md("""### How much of the answer is the assumption?

Because LGD is assumed rather than estimated, the honest presentation of expected loss is a
surface, not a point."""),

    code("""grid = S.sensitivity_grid(test, p_test)
pivot = grid.pivot(index="lgd", columns="pd_odds_multiplier", values="total_expected_loss") / 1e6
pivot.style.format("{:,.0f}").background_gradient(cmap="Blues")"""),

    md("""Read across a row and the movement is the model's stress. Read down a column and it is the
LGD assumption alone. They are comparable in size - which is exactly why the assumption is
labelled everywhere it appears, and why sourcing real recovery data is the single
highest-value next step for this analysis."""),

    md("""## 9. What drives the score

Three views, because one importance ranking is not an explanation."""),

    code("""odds = X.logistic_odds_ratios(load_model("behavioural", "Logistic regression", calibrated=False))
odds.head(12).style.format({"coefficient_log_odds": "{:+.3f}", "odds_ratio": "{:.3f}"})"""),

    md("""Coefficients are on standardised inputs, so each is the change in log-odds per one standard
deviation of that feature (or versus the dropped reference level for the one-hot columns). An
odds ratio above 1 raises the odds of default.

Note the correlated-feature effect: several delinquency measures carry overlapping information,
so individual coefficients trade off against each other and should be read as a group rather
than one at a time. This is a normal scorecard problem, and it is one reason the permutation
and SHAP views below are worth having."""),

    code("""perm = pd.read_csv(C.TABLES / "permutation_importance.csv")
shap_imp = pd.read_csv(C.TABLES / "shap_global_importance.csv")
pd.concat([
    perm.head(10)[["feature", "auc_drop_mean"]].reset_index(drop=True),
    shap_imp.head(10)[["feature", "mean_abs_shap"]].reset_index(drop=True),
], axis=1)"""),

    md("""All three views agree: recent arrears status dominates, then the depth and persistence of
delinquency, then available credit and balance level. Demographics sit far down the list.

The clean read for a risk audience: **how the customer has been paying matters far more than
who the customer is.**"""),

    code("""# Account-level reason codes - the adverse-action-style view of one account's score.
pd.read_csv(C.TABLES / "reason_codes_high_risk_account.csv").style.format({
    "feature_value": "{:.3f}", "shap_value": "{:+.3f}"
})"""),

    md("""## 10. Limitations

1. **One cohort, one outcome month.** No out-of-time validation; the split is
   stratified-random, not time-based.
2. **Taiwan, 2005.** The portfolio follows a domestic card-debt crisis. A 22% default rate is
   not a through-the-cycle rate and none of these levels should be read across to another book.
3. **Behavioural, not application.** The champion model needs six months of repayment history.
4. **LGD is assumed.** Every currency figure inherits that assumption one-for-one.
5. **No macroeconomic variables.** Unemployment and rates enter only as scenario multipliers.
6. **Demographic inputs.** Sex, education and marital status are present in the data and are
   used here for segment monitoring. Several are prohibited or restricted inputs for credit
   decisions in many jurisdictions; a deployable model would exclude them and be tested for
   disparate impact.

### Next steps

- Source recovery and collection data to replace the assumed LGD with an estimated one.
- Re-fit on multiple cohorts to enable out-of-time validation and a real PSI baseline.
- Rebuild without demographic inputs and quantify what discrimination that costs.
- Extend to a lifetime PD term structure if the goal is IFRS 9 / CECL-style reporting.

---

*Portfolio project built on public data. Not investment, credit or financial advice.
Reproduce end to end with `python -m src.run_pipeline`.*"""),
]


def build(execute: bool = True) -> Path:
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}

    if execute:
        client = NotebookClient(
            nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
        )
        client.execute()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    print(build(execute="--no-exec" not in sys.argv))
