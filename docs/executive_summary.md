# Executive summary

**Consumer Loan Credit Risk Modeling and Expected Loss Analysis**
UCI *Default of Credit Card Clients* (Taiwan, 2005) · 5,993-account held-out
test fold · generated 2026-09-08

---

## The question

Can borrower and account characteristics be used to estimate the probability of default, rank
accounts by risk, and translate that ranking into an expected credit loss under baseline and
stressed conditions?

## The answer, in five lines

1. **Yes, and the signal is behavioural.** A gradient-boosting model reaches
   **0.778 ROC-AUC** (0.556 Gini,
   0.427 KS) on accounts it has never seen. Recent arrears status is the
   dominant driver; demographics contribute little.
2. **The ranking is usable.** The riskiest decile defaults at
   **70%** against 4.2% in the
   safest - a 3.1x lift - and the riskiest two deciles contain
   50% of all defaults.
3. **The probabilities are calibrated,** not just ordered: expected calibration error
   0.0174, Brier score 0.1351. That is the precondition for
   using them in an expected-loss calculation at all.
4. **Baseline expected loss is NT$65.7m** on
   NT$549.0m of exposure (11.96%), rising to
   NT$168.9m (+157%) under a severe scenario in which
   default odds more than double and recoveries fall.
5. **A tested cutoff works.** Declining accounts with a predicted PD at or above
   44.7% would have cut the approved book's realised
   default rate from 22.1% to **14.9%** while
   approving 85% of applications - measured retrospectively on held-out
   accounts, not in live lending.

## What is estimated and what is assumed

| Component | Source | Status |
| --- | --- | --- |
| Probability of default | Calibrated gradient-boosting model | **Estimated** |
| Exposure at default | Drawn balance + 35% of the undrawn line | Balance observed, **CCF assumed** |
| Loss given default | 65% baseline | **Assumed** - the dataset has no recovery data |
| Scenario severity | PD odds x2.25, LGD 85% | **Judgemental** |

Every currency figure in this project is a modelled PD multiplied by an assumed LGD. It moves
one-for-one with that assumption, and the full PD-stress x LGD surface is published alongside
the point estimate rather than behind it.

## Where the risk is concentrated

The **Q1 lowest** credit-limit quintile carries 7% of
exposure but 12% of expected loss - a concentration ratio of
1.71x. Concentration by risk decile is sharper still: the riskiest
decile alone accounts for
31% of expected loss.

## What this does not show

- It is a **behavioural** model, requiring six months of repayment history. Rebuilt on
  origination-time inputs only, discrimination falls to roughly
  0.63
  ROC-AUC. It cannot be used to underwrite new applications.
- One cohort, one outcome month, one country, one year - and that year followed a domestic
  card-debt crisis. A 22% default rate is not a through-the-cycle rate.
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
