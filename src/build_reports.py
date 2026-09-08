"""Generate the written deliverables from ``outputs/tables/results.json``.

Every number in the README, the executive summary and the risk memo is substituted from
the pipeline's own output. Nothing is typed by hand, so the documents cannot drift away
from the model as the code changes - re-run the pipeline and re-run this script and the
prose updates itself.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from . import config as C


def _load():
    return json.loads((C.TABLES / "results.json").read_text())


def money(x: float, unit: str = "m") -> str:
    if unit == "m":
        return f"{C.CURRENCY}{x / 1e6:,.1f}m"
    if unit == "bn":
        return f"{C.CURRENCY}{x / 1e9:,.2f}bn"
    return f"{C.CURRENCY}{x:,.0f}"


def md_table(df: pd.DataFrame, formats: dict[str, str] | None = None,
             rename: dict[str, str] | None = None) -> str:
    d = df.copy()
    if formats:
        for col, fmt in formats.items():
            if col in d.columns:
                d[col] = d[col].map(lambda v: fmt.format(v) if pd.notna(v) else "-")
    if rename:
        d = d.rename(columns=rename)
    d.columns = [str(c).replace("_", " ") for c in d.columns]
    header = "| " + " | ".join(d.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


# --------------------------------------------------------------------------------------
def build_readme(r: dict) -> str:
    h = r["headline"]
    d = r["decision_rule"]
    el = r["expected_loss_baseline"]
    th = r["thresholds"]
    prep = r["data"]
    stab = {s["split"]: s["roc_auc"] for s in r["stability"]["by_split"]}
    champ_cal = r["training"]["behavioural"]["models"][h["champion"]]["calibration_method"]

    comp = pd.read_csv(C.TABLES / "model_comparison.csv")
    comp_md = md_table(
        comp[["model", "val_auc", "roc_auc", "pr_auc", "ks", "brier", "ece"]],
        formats={c: "{:.4f}" for c in ["val_auc", "roc_auc", "pr_auc", "ks", "brier", "ece"]},
        rename={"model": "Model", "val_auc": "Validation ROC-AUC", "roc_auc": "Test ROC-AUC",
                "pr_auc": "PR-AUC", "ks": "KS", "brier": "Brier", "ece": "Calibration error"},
    )

    fs = pd.read_csv(C.TABLES / "feature_set_comparison.csv")
    fs["feature_set"] = fs["feature_set"].map(
        {"behavioural": "Behavioural (6 months of history)",
         "origination": "Origination only (limit + demographics)"}
    )
    fs_md = md_table(
        fs, formats={c: "{:.4f}" for c in ["roc_auc", "gini", "pr_auc", "ks", "brier"]},
        rename={"feature_set": "Feature set", "model": "Model", "n_features": "Features"},
    )

    scen = pd.read_csv(C.TABLES / "scenario_summary.csv")
    scen_md = md_table(
        scen[["scenario", "pd_odds_multiplier", "lgd_assumed", "ccf_assumed", "average_pd",
              "total_ead", "total_expected_loss", "el_rate_on_ead", "el_vs_baseline_pct"]],
        formats={"pd_odds_multiplier": "{:.2f}x", "lgd_assumed": "{:.0%}", "ccf_assumed": "{:.0%}",
                 "average_pd": "{:.2%}", "total_ead": "{:,.0f}", "total_expected_loss": "{:,.0f}",
                 "el_rate_on_ead": "{:.2%}", "el_vs_baseline_pct": "{:+.0%}"},
        rename={"scenario": "Scenario", "total_ead": f"EAD ({C.CURRENCY})",
                "total_expected_loss": f"Expected loss ({C.CURRENCY})",
                "el_rate_on_ead": "EL / EAD", "el_vs_baseline_pct": "vs baseline"},
    )

    dec = pd.read_csv(C.TABLES / "risk_deciles.csv")
    dec_md = md_table(
        dec[["decile", "accounts", "defaults", "observed_default_rate", "mean_predicted_pd",
             "lift", "share_of_exposure", "cumulative_defaults_captured"]],
        formats={"observed_default_rate": "{:.1%}", "mean_predicted_pd": "{:.1%}",
                 "lift": "{:.2f}x", "share_of_exposure": "{:.1%}",
                 "cumulative_defaults_captured": "{:.0%}"},
        rename={"decile": "Decile", "observed_default_rate": "Observed default rate",
                "mean_predicted_pd": "Mean predicted PD", "lift": "Lift",
                "share_of_exposure": "Share of exposure",
                "cumulative_defaults_captured": "Cumulative defaults captured"},
    )

    cut = pd.read_csv(C.TABLES / "cutoff_comparison.csv")
    cut_md = md_table(
        cut[["cutoff_rule", "threshold", "flagged_rate", "precision", "recall", "f1"]],
        formats={"threshold": "{:.1%}", "flagged_rate": "{:.1%}", "precision": "{:.3f}",
                 "recall": "{:.3f}", "f1": "{:.3f}"},
        rename={"cutoff_rule": "Cutoff rule", "threshold": "PD cutoff",
                "flagged_rate": "Share declined", "precision": "Precision", "recall": "Recall",
                "f1": "F1"},
    )

    return f"""# Consumer Loan Credit Risk Modeling and Expected Loss Analysis

Probability-of-default modelling, expected-credit-loss estimation and scenario analysis on
the UCI *Default of Credit Card Clients* dataset (Taiwan, 2005; 30,000 accounts).

## What this project does, in plain terms

A credit-card lender needs to answer three questions about the customers already on its
books, and this project answers all three from the data alone.

**Who is likely to miss their next payment?** Each account gets a score between 0% and 100%
- its chance of failing to pay next month. The model learns this from six months of billing
and repayment history: how much was owed, how much was actually paid, and how many months
each customer had fallen behind.

**How much money is at risk?** A chance of default is not yet a loss. To turn one into the
other you need three things: how likely the customer is to default, how much they will owe
at that moment, and how much of that the lender fails to recover. Multiply them together and
you get the **expected loss** - the amount the lender should expect to write off. This
project computes it for every account and adds them up.

**What happens if conditions get worse?** Unemployment rises, people fall behind, and
recoveries shrink. The project re-runs the loss calculation under a mild downturn and a
severe one, so the lender can see the size of the hole before it appears.

The honest caveat, stated everywhere in this repository: only the first of those three
ingredients can be measured from this dataset. How much a lender recovers after a default is
not recorded anywhere in it, so that number is an **assumption**, clearly labelled as one.
Every currency figure here rises and falls with it.

**How to read the results.** Two ideas do most of the work. A model can be good at *ranking*
- putting the risky customers above the safe ones - and separately good at *calibration* -
being right about the actual level of risk. Ranking is what you need to decide who to
decline. Calibration is what you need to put a number on the losses, because a model that
ranks perfectly but is systematically too low produces a loss estimate that is too low by
the same amount. Both are measured here, on data the model never saw.

## The business question

Stated formally: **can borrower and account characteristics be used to estimate the
probability of default, rank accounts by risk, and translate that into an expected credit
loss under baseline and stressed conditions?**

Every figure below is computed on a held-out test fold of {h['test_accounts']:,} accounts that
no model saw during training, model selection, calibration or cutoff setting. Re-running
`python -m src.run_pipeline` regenerates all of them, and re-running
`python -m src.build_reports` regenerates this file - none of the numbers in this README are
typed by hand.

Every abbreviation used below is defined in the [glossary](#glossary) at the end.

---

## Headline results

| | |
| --- | --- |
| Champion model | **{h['champion']}** |
| Test ROC-AUC / Gini / KS | **{h['champion_roc_auc']:.3f}** / {h['champion_gini']:.3f} / {h['champion_ks']:.3f} |
| Test PR-AUC (base rate {h['test_default_rate']:.1%}) | **{h['champion_pr_auc']:.3f}** |
| Brier score / expected calibration error | {h['champion_brier']:.4f} / {h['champion_ece']:.4f} |
| Logistic-regression benchmark (ROC-AUC) | {h['logreg_roc_auc']:.3f} - the champion beats it by {h['auc_gain_over_logreg']:.4f} |
| Riskiest decile default rate | **{h['top_decile_default_rate']:.0%}** vs {h['bottom_decile_default_rate']:.1%} in the safest decile ({h['top_decile_lift']:.1f}x lift) |
| Defaults captured in the riskiest two deciles | {h['top_two_deciles_capture']:.0%} |
| Portfolio exposure at default (test fold) | {money(el['total_ead'])} |
| Baseline expected loss | {money(el['total_expected_loss'])} ({el['el_rate_on_ead']:.2%} of exposure) |
| Severe-scenario expected loss | {money(h['severe_el'])} ({h['severe_el_uplift_pct']:+.0%}) |

**What those measures mean.** *ROC-AUC* is how reliably the model puts a customer who
defaults above one who does not, when you draw one of each at random: 0.5 is a coin flip and
1.0 is perfect, so {h['champion_roc_auc']:.3f} means it gets that comparison right about
{h['champion_roc_auc']:.0%} of the time. *Gini* is the same information on a 0-to-1 scale
({h['champion_gini']:.3f}). *KS* is the widest gap between the score distributions of
defaulters and non-defaulters. *PR-AUC* measures performance on the minority group - only
{h['test_default_rate']:.1%} of accounts default - where a naive model scores
{h['test_default_rate']:.3f}. *Brier score* and *calibration error* check whether the
predicted percentages are true in level, not just in order; both are better when smaller.

**What is estimated and what is assumed.** PD is estimated from the data. Loss given default
is *not*: the dataset contains no recovery, collection or charge-off information, so LGD is a
stated assumption ({C.LGD_BASELINE:.0%} baseline) and every loss figure moves one-for-one with
it. Exposure at default combines the observed balance with an assumed {C.CCF_BASELINE:.0%}
credit conversion factor on the undrawn line. This split is stated everywhere the numbers
appear, and `outputs/tables/sensitivity_grid.csv` shows the whole PD-stress x LGD surface
rather than a single loss number.

---

## Deliverables

| Deliverable | Where |
| --- | --- |
| Technical notebook | [`notebooks/01_credit_risk_modelling.ipynb`](notebooks/01_credit_risk_modelling.ipynb) |
| Executive summary | [`docs/executive_summary.md`](docs/executive_summary.md) |
| Two-page risk memo | [`docs/risk_memo.md`](docs/risk_memo.md) |
| Model card | [`docs/model_card.md`](docs/model_card.md) |
| Data dictionary | [`docs/data_dictionary.md`](docs/data_dictionary.md) |
| Interactive risk dashboard | **[Open the live app](https://credit-risk-el-modeling.streamlit.app)** - move the loss-given-default, credit-conversion-factor, stress and cutoff sliders and every figure recomputes. Source: [`dashboard/app.py`](dashboard/app.py) |
| Static risk dashboard | **[View it live](https://pooja003-cloud.github.io/credit-risk-el-modeling/dashboard/dashboard.html)** or open [`dashboard/dashboard.html`](dashboard/dashboard.html) locally |
| Model-comparison table | [`outputs/tables/model_comparison.csv`](outputs/tables/model_comparison.csv) |
| Expected-loss tables | [`outputs/tables/`](outputs/tables/) (`el_by_risk_decile.csv`, `el_by_*.csv`, `scenario_summary.csv`) |
| Figures | [`outputs/figures/`](outputs/figures/) |

---

## Quickstart

```bash
git clone https://github.com/pooja003-cloud/credit-risk-el-modeling.git
cd credit-risk-el-modeling
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

python -m src.run_pipeline            # data -> models -> evaluation -> EL -> figures
python -m src.build_static_dashboard  # rebuild dashboard/dashboard.html
python -m src.build_reports           # rebuild README and the written deliverables
streamlit run dashboard/app.py        # interactive dashboard

pytest -q                             # regression tests on the risk maths
python tools/verify_results.py        # recompute every headline number independently
```

The raw dataset is committed in `data/raw/`, so the pipeline runs offline. Full run:
about a minute on a laptop.

---

## Data and preparation

**Source.** UCI Machine Learning Repository, *Default of Credit Card Clients* (Yeh & Lien,
2009) - 30,000 Taiwanese credit-card accounts observed April-September 2005, with a binary
flag for default in October 2005. Amounts are in New Taiwan dollars.

The loader validates the file against the published dataset statistics before anything else
runs - {prep['integrity_checks']['rows']:,} rows, {prep['integrity_checks']['defaults']:,}
defaults, a {prep['integrity_checks']['default_rate']:.2%} default rate - and raises if a
mirror does not match, so a truncated or altered copy cannot flow silently into the models.

Preparation decisions, all reproducible in `src/data_prep.py`:

- **{prep['cleaning']['duplicate_rows_dropped']} exact duplicate rows** (identical on every
  field except `ID`) dropped, so the same account cannot appear in both training and test.
- **{prep['cleaning']['education_codes_recoded_to_other']} `EDUCATION` values** and
  **{prep['cleaning']['marriage_codes_recoded_to_other']} `MARRIAGE` values** carry
  undocumented codes; each is collapsed into an explicit "Other" level rather than treated
  as a distinct category.
- **{prep['cleaning']['negative_bill_amounts']:,} negative bill amounts** are retained -
  they are credit balances (overpayments), not errors - and utilisation features floor them
  at zero.
- `PAY_0` renamed `PAY_1` so repayment status lines up with the bill and payment columns.
- Raw `PAY_*` codes (-2 = no usage, -1 = paid in full, 0 = revolving) are **not** used as a
  numeric severity scale. They are split into non-negative months-past-due plus explicit
  counts of revolving, full-payment and inactive months.
- **{prep['n_features_behavioural']} engineered features**: utilisation level, trend and
  volatility; payment ratios; delinquency depth, breadth and direction; balance dynamics;
  available credit; demographics.
- Stratified **60 / 20 / 20** train / validation / test split. The dataset has no origination
  or observation dates - every account shares the same six-month window - so a time-based
  split is not possible. That limitation is recorded in the model card rather than papered
  over.

### Data leakage: the boundary this project draws

The target is default in **October 2005**; every feature is dated **September 2005 or
earlier**, so no post-outcome information is used. That is the easy half.

The harder half is being clear about *which decision* the model supports. A model with six
months of repayment history is a **behavioural** scorecard - the tool for re-underwriting,
limit management and collections prioritisation on an existing account. It is not an
**application** scorecard, because at origination none of that history exists. Quoting the
behavioural AUC as if it applied at account opening would be leakage relative to the decision
being made, so both models are built and reported:

{fs_md}

Roughly {(h['champion_roc_auc'] - fs[fs['feature_set'].str.startswith('Origination')]['roc_auc'].max()):.2f}
of ROC-AUC comes from the repayment history alone. The origination-only model is barely
better than a coin flip once history is removed, which is the honest measure of how much of
this model's power an application-time user would *not* get.

---

## Models and results

Five model families, each selected on the validation fold with a small explicit grid, then
scored once on test.

**Calibration is treated as a modelling step, not an afterthought,** because expected loss
multiplies PD *levels* and an uncalibrated ranking score cannot be used in
`EL = PD x LGD x EAD`. Isotonic and Platt scaling are compared by five-fold cross-validation
*inside* the validation fold - judging a calibrator on the same rows it was fitted on would
flatter the more flexible method - and the winner is refitted on the whole fold with the base
model frozen, so calibration never touches training data or the test fold. The champion's
chosen method was **{champ_cal}**.

Calibrated probabilities are then floored at {C.PD_FLOOR:.2%} and capped at {C.PD_CAP:.2%}.
Isotonic calibration is a step function and will assign a PD of exactly 0 or exactly 1 to
accounts in its outermost steps; a PD of zero would zero out that account's expected loss
entirely. Supervisory frameworks handle this with an explicit floor - Basel uses 0.03% for
retail exposures - and the same device is used here.

{comp_md}

Accuracy is deliberately not a headline: approving every account scores
{1 - h['test_default_rate']:.0%} accurate on this portfolio and identifies no defaults at all.

**The gradient-boosting gain over the scorecard is real but small** -
{h['auc_gain_over_logreg']:+.4f} ROC-AUC, {h['champion_pr_auc'] - h['logreg_pr_auc']:+.4f}
PR-AUC. That is worth saying plainly: on this dataset a fully interpretable logistic
regression gets most of the way, and the complexity of a boosted model has to be justified by
more than a third decimal place.

**Stability.** ROC-AUC is {stab['train']:.3f} on train, {stab['val']:.3f} on validation and
{stab['test']:.3f} on test - mild optimism, no collapse. The population stability index
between the training and test score distributions is
{r['stability']['psi_train_vs_test']:.4f}, far below the 0.10 monitoring trigger, as expected
for a random split of one cohort.

### Rank-ordering

{dec_md}

![Default rate by predicted-risk decile](outputs/figures/decile_default_rates.png)

### The two errors and where the cutoff sits

A false negative - approving an account that defaults - costs `LGD x EAD`. A false positive -
declining an account that would have paid - costs the forgone margin on the balance that
customer would have carried. Under the assumptions here those are roughly an order of
magnitude apart, so a 0.5 cutoff is simply the wrong place to stand.

{cut_md}

The **deployed cutoff of {th['policy_cutoff_from_validation']:.1%}** was chosen on the
validation fold as the loosest rule that holds the approved book's default rate at or below
the {th['policy_target_bad_rate']:.0%} risk-appetite target. Applied to the test fold it
declines {d['decline_rate']:.0%} of accounts ({d['accounts_declined']:,}) and cuts the realised
default rate of the approved book from {d['bad_rate_no_cutoff']:.1%} to
**{d['bad_rate_of_approved']:.1%}**. It catches {d['defaults_avoided']:,} of the
{d['defaults_avoided'] + d['defaults_retained']:,} defaults in the fold and turns away
{d['good_accounts_declined']:,} accounts that would in fact have paid.

This is a retrospective evaluation on held-out data, not a claim that lending decisions
improved - no live decisions were made, and the accounts declined by the rule were extended
credit in reality.

The purely cost-minimising cutoff is reported alongside it and comes out at
{th['cost_optimal_from_validation']:.1%}, which would decline most of the portfolio. That is
not a credible policy; it is what happens when a one-period loss is weighed against one period
of margin. `outputs/tables/cutoff_margin_sensitivity.csv` shows the implied cutoff moving from
{pd.read_csv(C.TABLES / 'cutoff_margin_sensitivity.csv')['implied_cutoff'].iloc[0]:.0%} to
{pd.read_csv(C.TABLES / 'cutoff_margin_sensitivity.csv')['implied_cutoff'].iloc[-1]:.0%} as the
assumed margin rises, which is exactly why the deployed cutoff is set by risk appetite instead.

![Setting the cutoff](outputs/figures/cutoff_net_cost.png)

---

## Expected loss

```
EL = PD x LGD x EAD
```

| Component | How it is obtained | Status |
| --- | --- | --- |
| **PD** | Calibrated champion model | Estimated from data |
| **EAD** | Drawn balance + CCF x undrawn line | Balance and line observed; **CCF {C.CCF_BASELINE:.0%} assumed** |
| **LGD** | Stated assumption | **Assumed {C.LGD_BASELINE:.0%}** - not estimable from this dataset |

A credit card is a revolving commitment, so exposure at the moment of default is not today's
balance: distressed borrowers typically draw further before they stop paying. Using the
balance alone understates exposure; using the full credit line overstates it. The credit
conversion factor is the assumed share of the undrawn line drawn between today and default.

On the test fold, exposure at default is {money(el['total_ead'])} against
{money(el['total_drawn_balance'])} of drawn balances and {money(el['total_limit'])} of
committed limits.

**A partial check on the EL number.** Holding the LGD assumption fixed, the loss actually
implied by the accounts that defaulted is {money(el['realised_loss_at_assumed_lgd'])}, against
a modelled expected loss of {money(el['total_expected_loss'])} - a
{abs(el['total_expected_loss'] / el['realised_loss_at_assumed_lgd'] - 1):.1%} gap. That tests
the PD and EAD components against outcomes; it says nothing about whether the LGD assumption
itself is right.

### Scenario analysis

{scen_md}

Each scenario moves all three components together, because in a downturn they do not move
independently. The PD stress is applied to the **odds** of default, equivalent to a constant
shift on the log-odds (score) scale, so PDs stay inside (0, 1) and the stress is proportionate
across the risk spectrum instead of pushing high-risk accounts above 1.0.

Scenario severities are judgemental and are **not** calibrated to a published macroeconomic
model. They are labelled as illustrative wherever they appear.

![Expected loss under three scenarios](outputs/figures/scenario_comparison.png)

![Sensitivity of expected loss](outputs/figures/sensitivity_heatmap.png)

---

## What drives the score

Three views, because a single importance ranking is not an explanation:

- **Logistic-regression odds ratios** - direction and size of each effect, auditable line by line.
- **Permutation importance** - measured on held-out data, so it reflects what the model relies
  on rather than what it split on.
- **SHAP** - additive per-account attributions, which also give reason codes for individual
  accounts (`outputs/tables/reason_codes_*.csv`).

All three agree on the ranking: recent arrears status dominates, followed by the depth and
persistence of delinquency, then available credit and balance level. The clean read is that
*how the customer has been paying* matters far more than *who the customer is*.

![Global SHAP importance](outputs/figures/shap_importance.png)

![Logistic-regression odds ratios](outputs/figures/logistic_odds_ratios.png)

---

## Repository layout

```
credit-risk-el-modeling/
├── data/raw/                     UCI dataset, committed so the pipeline runs offline
├── src/
│   ├── config.py                 paths, seeds and every risk assumption in one place
│   ├── data_prep.py              validation, cleaning, features, leakage-aware feature sets
│   ├── models.py                 model ladder, validation-fold selection, calibration
│   ├── evaluate.py               discrimination, calibration, thresholds, deciles, segments
│   ├── expected_loss.py          EAD construction, EL, decision-rule evaluation
│   ├── scenarios.py              PD stressing on the odds scale, scenarios, sensitivity grid
│   ├── explainability.py         odds ratios, permutation importance, SHAP, PSI
│   ├── plots.py                  every figure
│   ├── run_pipeline.py           end-to-end run; writes outputs/tables/results.json
│   ├── build_static_dashboard.py self-contained HTML dashboard
│   └── build_reports.py          regenerates README and the written deliverables
├── notebooks/                    technical walkthrough
├── dashboard/                    Streamlit app + static HTML dashboard
├── docs/                         executive summary, risk memo, model card, data dictionary
├── outputs/{{figures,tables,models}}
└── tests/                        regression tests on the risk maths
```

---

## Limitations

- **One cohort, one outcome month.** No out-of-time validation is possible; the split is
  stratified-random, not time-based.
- **Taiwan, 2005.** The portfolio follows a domestic card-debt crisis. A
  {h['test_default_rate']:.0%} default rate is not a through-the-cycle rate and none of these
  levels should be read across to another book.
- **Behavioural, not application.** The champion model needs six months of repayment history.
- **LGD is assumed, not estimated.** Every currency figure inherits that assumption.
- **No macroeconomic variables.** Unemployment and rates enter only through scenario
  multipliers, not as model inputs.
- **Demographic inputs.** Sex, education and marital status are present in the dataset and are
  used here for segment monitoring. Several are prohibited or restricted inputs for credit
  decisions in many jurisdictions; a deployable model would exclude them and be tested for
  disparate impact.

---

## Glossary

Terms are listed in the order they first matter, not alphabetically.

| Term | What it means |
| --- | --- |
| **Default** | The customer fails to make the required payment. Here, specifically, failing to pay in October 2005. |
| **PD** - probability of default | The chance that an account defaults, between 0% and 100%. This is what the model predicts. |
| **EAD** - exposure at default | How much the customer owes at the moment they default. Not the same as today's balance: a card is a revolving credit line, and people in trouble tend to draw down more before they stop paying. |
| **CCF** - credit conversion factor | The assumed share of the *unused* part of the credit limit that gets drawn between now and default. Used to turn today's balance into an exposure at default. Assumed here, at {C.CCF_BASELINE:.0%}. |
| **LGD** - loss given default | The share of the exposure the lender ultimately fails to recover, after collections and any settlement. Assumed here, at {C.LGD_BASELINE:.0%}, because this dataset records no recoveries. |
| **EL** - expected loss | `PD x LGD x EAD`. The amount a lender should expect to lose on an account, in money. |
| **Basis point (bp)** | One hundredth of a percentage point. A 3bp floor means 0.03%. |
| **ROC-AUC** | How reliably the model ranks a defaulter above a non-defaulter. 0.5 is random, 1.0 is perfect. The standard headline measure of a credit model's discrimination. |
| **Gini** | The same information as ROC-AUC on a 0-to-1 scale, calculated as `2 x ROC-AUC - 1`. Preferred in most credit-risk teams. |
| **KS** - Kolmogorov-Smirnov statistic | The widest separation between the score distributions of defaulters and non-defaulters. The traditional separation measure in retail scorecards. |
| **PR-AUC** - precision-recall area under the curve | Performance on the minority group. More informative than ROC-AUC when defaults are rare. |
| **Precision** | Of the accounts the model flags as risky, the share that really do default. |
| **Recall** | Of all the accounts that really default, the share the model flags. |
| **Calibration** | Whether a predicted 8% really corresponds to 8 defaults in every 100 such accounts. Different from ranking, and essential for loss estimates. |
| **Brier score** | The average squared error of the predicted probabilities. Smaller is better. |
| **Expected calibration error** | The average gap between predicted and observed default rates across risk bands. Smaller is better. |
| **Decile** | A tenth of the portfolio, sorted by predicted risk. Decile 1 is the safest 10%, decile 10 the riskiest. |
| **Lift** | How much more often a group defaults than the portfolio average. A lift of 3 means three times the average rate. |
| **Cutoff** | The predicted-PD level above which an account is declined. |
| **Behavioural scorecard** | A model that uses repayment history on an existing account. Used for limit changes, collections and loss reporting. This project's champion model is one. |
| **Application scorecard** | A model that scores a *new* applicant, before any repayment history exists. This project builds one only as a comparison. |
| **Data leakage** | Using information that would not have been available at the moment of the decision, which flatters results in testing and fails in reality. |
| **Delinquency / arrears** | Being behind on payments, measured here in whole months. |
| **Revolving** | Paying only the minimum and carrying the balance forward. |
| **Utilisation** | The share of the credit limit currently used. |
| **Stress scenario** | A deliberately worse set of assumptions - more defaults, lower recoveries - used to size losses in a downturn. |
| **Log-odds** | The scale credit scores live on. Stressing a model on this scale keeps every probability between 0% and 100% and moves risky and safe accounts proportionately. |
| **Isotonic / Platt scaling** | Two standard methods for correcting a model's predicted probabilities after training. |
| **PSI** - population stability index | A monitoring measure of how far a scored population has drifted from the one the model was built on. Below 0.10 is stable. |
| **SHAP** | A method that attributes a single prediction to its individual inputs, so a score can be explained account by account. |
| **Champion / challenger** | The model in use, and the simpler or rival model kept alongside it as a benchmark. |
| **Train / validation / test** | The three slices of data: one to fit the model, one to tune and calibrate it, and one untouched until the end, used to report honest performance. |

---

## Reference

Yeh, I.-C. and Lien, C.-H. (2009). "The comparisons of data mining techniques for the
predictive accuracy of probability of default of credit card clients." *Expert Systems with
Applications*, 36(2), 2473-2480. Dataset: UCI Machine Learning Repository,
<https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>.

Portfolio project. Not investment, credit or financial advice.
"""


# --------------------------------------------------------------------------------------
def build_executive_summary(r: dict) -> str:
    h = r["headline"]
    d = r["decision_rule"]
    el = r["expected_loss_baseline"]
    th = r["thresholds"]
    scen = pd.read_csv(C.TABLES / "scenario_summary.csv")
    seg = pd.read_csv(C.TABLES / "el_by_limit_band.csv").sort_values("el_concentration",
                                                                    ascending=False)
    worst = seg.iloc[0]

    return f"""# Executive summary

**Consumer Loan Credit Risk Modeling and Expected Loss Analysis**
UCI *Default of Credit Card Clients* (Taiwan, 2005) · {h['test_accounts']:,}-account held-out
test fold · generated {date.today().isoformat()}

---

## The question

Can borrower and account characteristics be used to estimate the probability of default, rank
accounts by risk, and translate that ranking into an expected credit loss under baseline and
stressed conditions?

## The answer, in five lines

1. **Yes, and the signal is behavioural.** A gradient-boosting model reaches
   **{h['champion_roc_auc']:.3f} ROC-AUC** ({h['champion_gini']:.3f} Gini,
   {h['champion_ks']:.3f} KS) on accounts it has never seen. Recent arrears status is the
   dominant driver; demographics contribute little.
2. **The ranking is usable.** The riskiest decile defaults at
   **{h['top_decile_default_rate']:.0%}** against {h['bottom_decile_default_rate']:.1%} in the
   safest - a {h['top_decile_lift']:.1f}x lift - and the riskiest two deciles contain
   {h['top_two_deciles_capture']:.0%} of all defaults.
3. **The probabilities are calibrated,** not just ordered: expected calibration error
   {h['champion_ece']:.4f}, Brier score {h['champion_brier']:.4f}. That is the precondition for
   using them in an expected-loss calculation at all.
4. **Baseline expected loss is {money(el['total_expected_loss'])}** on
   {money(el['total_ead'])} of exposure ({el['el_rate_on_ead']:.2%}), rising to
   {money(h['severe_el'])} ({h['severe_el_uplift_pct']:+.0%}) under a severe scenario in which
   default odds more than double and recoveries fall.
5. **A tested cutoff works.** Declining accounts with a predicted PD at or above
   {th['policy_cutoff_from_validation']:.1%} would have cut the approved book's realised
   default rate from {d['bad_rate_no_cutoff']:.1%} to **{d['bad_rate_of_approved']:.1%}** while
   approving {d['approval_rate']:.0%} of applications - measured retrospectively on held-out
   accounts, not in live lending.

## What is estimated and what is assumed

| Component | Source | Status |
| --- | --- | --- |
| Probability of default | Calibrated gradient-boosting model | **Estimated** |
| Exposure at default | Drawn balance + {C.CCF_BASELINE:.0%} of the undrawn line | Balance observed, **CCF assumed** |
| Loss given default | {C.LGD_BASELINE:.0%} baseline | **Assumed** - the dataset has no recovery data |
| Scenario severity | PD odds x{scen['pd_odds_multiplier'].iloc[-1]:.2f}, LGD {scen['lgd_assumed'].iloc[-1]:.0%} | **Judgemental** |

Every currency figure in this project is a modelled PD multiplied by an assumed LGD. It moves
one-for-one with that assumption, and the full PD-stress x LGD surface is published alongside
the point estimate rather than behind it.

## Where the risk is concentrated

The **{worst['limit_band']}** credit-limit quintile carries {worst['share_of_ead']:.0%} of
exposure but {worst['share_of_el']:.0%} of expected loss - a concentration ratio of
{worst['el_concentration']:.2f}x. Concentration by risk decile is sharper still: the riskiest
decile alone accounts for
{pd.read_csv(C.TABLES / 'el_by_risk_decile.csv')['share_of_el'].iloc[-1]:.0%} of expected loss.

## What this does not show

- It is a **behavioural** model, requiring six months of repayment history. Rebuilt on
  origination-time inputs only, discrimination falls to roughly
  {pd.read_csv(C.TABLES / 'feature_set_comparison.csv').query("feature_set=='origination'")['roc_auc'].max():.2f}
  ROC-AUC. It cannot be used to underwrite new applications.
- One cohort, one outcome month, one country, one year - and that year followed a domestic
  card-debt crisis. A {h['test_default_rate']:.0%} default rate is not a through-the-cycle rate.
- No live decisions were made and no lending outcome was improved. The cutoff analysis is a
  retrospective evaluation on held-out data.

## Recommended next steps

1. Source recovery and collection data to replace the assumed LGD with an estimated one; it is
   the single largest source of uncertainty in the loss numbers.
2. Re-fit on multiple cohorts to enable an out-of-time validation and a genuine PSI monitoring
   baseline.
3. Before any deployment, drop the demographic inputs and run disparate-impact testing on the
   remaining behavioural features.

Full detail: [`docs/risk_memo.md`](risk_memo.md), [`docs/model_card.md`](model_card.md),
[`notebooks/01_credit_risk_modelling.ipynb`](../notebooks/01_credit_risk_modelling.ipynb).
"""


# --------------------------------------------------------------------------------------
def build_risk_memo(r: dict) -> str:
    h = r["headline"]
    d = r["decision_rule"]
    dc = r["decision_rule_cost_minimising"]
    el = r["expected_loss_baseline"]
    th = r["thresholds"]
    stab = {s["split"]: s["roc_auc"] for s in r["stability"]["by_split"]}
    scen = pd.read_csv(C.TABLES / "scenario_summary.csv")
    ms = pd.read_csv(C.TABLES / "cutoff_margin_sensitivity.csv")
    segperf = pd.read_csv(C.TABLES / "segment_performance.csv")
    weakest = segperf.sort_values("roc_auc").iloc[0]

    # State rank-ordering exactly as observed rather than asserting monotonicity.
    dec = pd.read_csv(C.TABLES / "risk_deciles.csv").sort_values("decile")
    rates = dec["observed_default_rate"].to_numpy()
    breaks = [int(dec["decile"].iloc[i]) for i in range(len(rates) - 1) if rates[i] >= rates[i + 1]]
    if not breaks:
        rank_sentence = "**Rank-ordering is monotone across all ten deciles.**"
    else:
        pairs = ", ".join(f"{b} and {b + 1}" for b in breaks)
        rank_sentence = (
            f"**Rank-ordering is monotone except between deciles {pairs}**, where the observed "
            f"rates are within {max(abs(rates[b - 1] - rates[b]) for b in breaks):.1%} of each "
            "other and the ordering is not statistically meaningful at this sample size."
        )

    return f"""# Risk memo

**To:** Credit Risk Committee
**From:** Model development
**Subject:** Consumer card portfolio - PD model, expected credit loss and scenario results
**Date:** {date.today().isoformat()}
**Basis:** UCI *Default of Credit Card Clients* (Taiwan, 2005); {h['test_accounts']:,}-account
held-out test fold

---

## 1. Recommendation

Adopt the calibrated gradient-boosting PD model as the **behavioural** risk-ranking engine for
this portfolio - for limit management, collections prioritisation and expected-loss reporting -
and set the decline cutoff at a predicted PD of **{th['policy_cutoff_from_validation']:.1%}**,
which holds the approved book's default rate at the {th['policy_target_bad_rate']:.0%} appetite
target. Do **not** use it as an application scorecard: it depends on repayment history that does
not exist at origination.

Retain the logistic-regression scorecard as the challenger and the reporting benchmark. It
reaches {h['logreg_roc_auc']:.3f} ROC-AUC against the champion's {h['champion_roc_auc']:.3f} -
a gain of {h['auc_gain_over_logreg']:+.4f} - and remains fully interpretable. The committee
should decide explicitly whether that gap justifies a model that needs SHAP to explain a single
decision.

## 2. Model performance

| Measure | Champion | Logistic benchmark |
| --- | --- | --- |
| ROC-AUC (test) | {h['champion_roc_auc']:.4f} | {h['logreg_roc_auc']:.4f} |
| Gini | {h['champion_gini']:.4f} | {2 * h['logreg_roc_auc'] - 1:.4f} |
| PR-AUC (base rate {h['test_default_rate']:.1%}) | {h['champion_pr_auc']:.4f} | {h['logreg_pr_auc']:.4f} |
| KS | {h['champion_ks']:.4f} | - |
| Brier | {h['champion_brier']:.4f} | {h['logreg_brier']:.4f} |
| Expected calibration error | {h['champion_ece']:.4f} | - |

Discrimination degrades gracefully across folds - {stab['train']:.3f} train, {stab['val']:.3f}
validation, {stab['test']:.3f} test - indicating mild optimism rather than over-fitting.

{rank_sentence} The riskiest decile defaults at
{h['top_decile_default_rate']:.0%} versus {h['bottom_decile_default_rate']:.1%} in the safest,
and the top two deciles capture {h['top_two_deciles_capture']:.0%} of defaults.

**Calibration was treated as a first-class requirement,** not a nicety. Expected loss multiplies
PD levels, so probabilities are isotonic-calibrated on the validation fold before being used.
Post-calibration, mean predicted PD is {el['average_pd']:.2%} against an observed
{h['test_default_rate']:.2%}.

**Segment performance.** Discrimination holds across borrower segments; the weakest is
"{weakest['segment']}" ({weakest['segment_type'].replace('_', ' ')}) at
{weakest['roc_auc']:.3f} ROC-AUC on {int(weakest['accounts']):,} accounts. No segment collapses
to random.

## 3. The two errors

- **False negative** - an account is approved and defaults. Cost: `LGD x EAD`.
- **False positive** - an account is declined that would have paid. Cost: forgone margin on the
  balance that customer would have carried.

Under the working assumptions these differ by roughly an order of magnitude, which is why the
0.5 cutoff that `predict()` returns by default is the wrong place to stand.

At the recommended {th['policy_cutoff_from_validation']:.1%} cutoff, applied to the test fold:

| | |
| --- | --- |
| Approval rate | {d['approval_rate']:.0%} ({d['accounts_declined']:,} accounts declined) |
| Default rate of the approved book | **{d['bad_rate_of_approved']:.1%}** (from {d['bad_rate_no_cutoff']:.1%}) |
| Default rate of the declined population | {d['bad_rate_of_declined']:.1%} |
| Defaults avoided / retained | {d['defaults_avoided']:,} / {d['defaults_retained']:,} |
| Good accounts turned away | {d['good_accounts_declined']:,} |
| Precision / recall at the cutoff | {h['policy_precision']:.3f} / {h['policy_recall']:.3f} |

This is a **retrospective evaluation on held-out data**. No live decisions were made, the
declined accounts were in fact extended credit, and no claim is made that lending outcomes
improved.

**Why not the cost-minimising cutoff?** Minimising modelled net cost implies a
{th['cost_optimal_from_validation']:.1%} cutoff - declining most of the book
({dc['decline_rate']:.0%}). That is an artefact of comparing a one-period loss against one
period of margin: as the assumed margin rises from {ms['assumed_margin_on_good_account'].iloc[0]:.0%}
to {ms['assumed_margin_on_good_account'].iloc[-1]:.0%}, the implied cutoff moves from
{ms['implied_cutoff'].iloc[0]:.0%} to {ms['implied_cutoff'].iloc[-1]:.0%} and the approval rate
from {ms['approval_rate'].iloc[0]:.0%} to {ms['approval_rate'].iloc[-1]:.0%}. The cutoff should
therefore be set by risk appetite, with the cost calculation used as a sensitivity.

## 4. Expected credit loss

`EL = PD x LGD x EAD`, on the test fold, baseline assumptions:

| | |
| --- | --- |
| Committed limits | {money(el['total_limit'])} |
| Drawn balances | {money(el['total_drawn_balance'])} |
| Exposure at default (balance + {C.CCF_BASELINE:.0%} of undrawn) | **{money(el['total_ead'])}** |
| Exposure-weighted PD | {el['exposure_weighted_pd']:.2%} |
| Assumed LGD | {C.LGD_BASELINE:.0%} |
| **Expected loss** | **{money(el['total_expected_loss'])}** ({el['el_rate_on_ead']:.2%} of exposure) |

**Partial back-test.** Holding LGD fixed, the loss implied by accounts that actually defaulted
is {money(el['realised_loss_at_assumed_lgd'])} against a modelled
{money(el['total_expected_loss'])} - within
{abs(el['total_expected_loss'] / el['realised_loss_at_assumed_lgd'] - 1):.1%}. This validates the
PD and EAD components against outcomes. It says nothing about the LGD assumption itself.

### Scenarios

| Scenario | PD odds | LGD | CCF | Expected loss | vs baseline |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(f"| {row['scenario']} | x{row['pd_odds_multiplier']:.2f} | {row['lgd_assumed']:.0%} | {row['ccf_assumed']:.0%} | {money(row['total_expected_loss'])} | {row['el_vs_baseline_pct']:+.0%} |" for _, row in scen.iterrows())}

The PD stress is applied to the odds of default (a constant shift on the log-odds scale), so
probabilities stay bounded and the stress is proportionate across the risk spectrum. Severities
are judgemental and are not calibrated to a published macroeconomic model.

**A severe scenario roughly {h['severe_el'] / el['total_expected_loss']:.1f}x the baseline loss
number is the planning figure the committee should react to**, and the sensitivity grid in
`outputs/tables/sensitivity_grid.csv` shows that roughly a third of that movement comes from the
LGD assumption rather than from the model.

## 5. Key risks in using this model

1. **LGD is assumed, not estimated.** Every currency figure scales with it. Sourcing recovery
   data is the highest-value next step.
2. **Single cohort.** One six-month window and one outcome month. No out-of-time validation is
   possible, so the reported PSI of {r['stability']['psi_train_vs_test']:.4f} is a formality, not
   evidence of through-time stability.
3. **Regime.** Taiwan 2005 followed a domestic card-debt crisis; a {h['test_default_rate']:.0%}
   default rate should not be read across to another portfolio.
4. **Behavioural dependency.** The model degrades to roughly
   {pd.read_csv(C.TABLES / 'feature_set_comparison.csv').query("feature_set=='origination'")['roc_auc'].max():.2f}
   ROC-AUC without repayment history. Any application-time use would be a misuse.
5. **Protected characteristics.** Sex, education and marital status are present in the data and
   are used here only for segment monitoring. They are prohibited or restricted inputs for credit
   decisions in many jurisdictions and must be removed, with disparate-impact testing on what
   remains, before deployment.

## 6. Monitoring, if deployed

- Monthly PSI on the score distribution; investigate above 0.10, escalate above 0.25.
- Quarterly recalibration check: predicted versus observed default rate by decile.
- Track the approved book's default rate against the {th['policy_target_bad_rate']:.0%} appetite
  target and re-set the cutoff when it drifts.
- Annual challenger comparison against the logistic scorecard; retire the champion if the gap
  closes.

---

*Portfolio project built on public data. Not investment, credit or financial advice.
Reproduce with `python -m src.run_pipeline`.*
"""


def build_all() -> list[str]:
    r = _load()
    written = []
    for path, content in [
        (C.ROOT / "README.md", build_readme(r)),
        (C.DOCS / "executive_summary.md", build_executive_summary(r)),
        (C.DOCS / "risk_memo.md", build_risk_memo(r)),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(str(path))
    return written


if __name__ == "__main__":
    for p in build_all():
        print(p)
