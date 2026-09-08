# Risk memo

**To:** Credit Risk Committee
**From:** Model development
**Subject:** Consumer card portfolio - PD model, expected credit loss and scenario results
**Date:** 2026-09-08
**Basis:** UCI *Default of Credit Card Clients* (Taiwan, 2005); 5,993-account
held-out test fold

---

## 1. Recommendation

Adopt the calibrated gradient-boosting PD model as the **behavioural** risk-ranking engine for
this portfolio - for limit management, collections prioritisation and expected-loss reporting -
and set the decline cutoff at a predicted PD of **44.7%**,
which holds the approved book's default rate at the 15% appetite
target. Do **not** use it as an application scorecard: it depends on repayment history that does
not exist at origination.

Retain the logistic-regression scorecard as the challenger and the reporting benchmark. It
reaches 0.764 ROC-AUC against the champion's 0.778 -
a gain of +0.0136 - and remains fully interpretable. The committee
should decide explicitly whether that gap justifies a model that needs SHAP to explain a single
decision.

## 2. Model performance

| Measure | Champion | Logistic benchmark |
| --- | --- | --- |
| ROC-AUC (test) | 0.7779 | 0.7643 |
| Gini | 0.5557 | 0.5286 |
| PR-AUC (base rate 22.1%) | 0.5516 | 0.5168 |
| KS | 0.4271 | - |
| Brier | 0.1351 | 0.1381 |
| Expected calibration error | 0.0174 | - |

Discrimination degrades gracefully across folds - 0.811 train, 0.796
validation, 0.778 test - indicating mild optimism rather than over-fitting.

**Rank-ordering is monotone except between deciles 5 and 6**, where the observed rates are within 0.6% of each other and the ordering is not statistically meaningful at this sample size. The riskiest decile defaults at
70% versus 4.2% in the safest,
and the top two deciles capture 50% of defaults.

**Calibration was treated as a first-class requirement,** not a nicety. Expected loss multiplies
PD levels, so probabilities are isotonic-calibrated on the validation fold before being used.
Post-calibration, mean predicted PD is 22.55% against an observed
22.13%.

**Segment performance.** Discrimination holds across borrower segments; the weakest is
"Other" (education label) at
0.580 ROC-AUC on 90 accounts. No segment collapses
to random.

## 3. The two errors

- **False negative** - an account is approved and defaults. Cost: `LGD x EAD`.
- **False positive** - an account is declined that would have paid. Cost: forgone margin on the
  balance that customer would have carried.

Under the working assumptions these differ by roughly an order of magnitude, which is why the
0.5 cutoff that `predict()` returns by default is the wrong place to stand.

At the recommended 44.7% cutoff, applied to the test fold:

| | |
| --- | --- |
| Approval rate | 85% (922 accounts declined) |
| Default rate of the approved book | **14.9%** (from 22.1%) |
| Default rate of the declined population | 61.8% |
| Defaults avoided / retained | 570 / 756 |
| Good accounts turned away | 352 |
| Precision / recall at the cutoff | 0.618 / 0.430 |

This is a **retrospective evaluation on held-out data**. No live decisions were made, the
declined accounts were in fact extended credit, and no claim is made that lending outcomes
improved.

**Why not the cost-minimising cutoff?** Minimising modelled net cost implies a
5.1% cutoff - declining most of the book
(92%). That is an artefact of comparing a one-period loss against one
period of margin: as the assumed margin rises from 6%
to 60%, the implied cutoff moves from
5% to 23% and the approval rate
from 8% to 72%. The cutoff should
therefore be set by risk appetite, with the cost calculation used as a sensitivity.

## 4. Expected credit loss

`EL = PD x LGD x EAD`, on the test fold, baseline assumptions:

| | |
| --- | --- |
| Committed limits | NT$1,007.5m |
| Drawn balances | NT$307.7m |
| Exposure at default (balance + 35% of undrawn) | **NT$549.0m** |
| Exposure-weighted PD | 18.40% |
| Assumed LGD | 65% |
| **Expected loss** | **NT$65.7m** (11.96% of exposure) |

**Partial back-test.** Holding LGD fixed, the loss implied by accounts that actually defaulted
is NT$67.5m against a modelled
NT$65.7m - within
2.7%. This validates the
PD and EAD components against outcomes. It says nothing about the LGD assumption itself.

### Scenarios

| Scenario | PD odds | LGD | CCF | Expected loss | vs baseline |
| --- | --- | --- | --- | --- | --- |
| Baseline | x1.00 | 65% | 35% | NT$65.7m | +0% |
| Moderate deterioration | x1.50 | 75% | 45% | NT$107.5m | +64% |
| Severe deterioration | x2.25 | 85% | 55% | NT$168.9m | +157% |

The PD stress is applied to the odds of default (a constant shift on the log-odds scale), so
probabilities stay bounded and the stress is proportionate across the risk spectrum. Severities
are judgemental and are not calibrated to a published macroeconomic model.

**A severe scenario roughly 2.6x the baseline loss
number is the planning figure the committee should react to**, and the sensitivity grid in
`outputs/tables/sensitivity_grid.csv` shows that roughly a third of that movement comes from the
LGD assumption rather than from the model.

## 5. Key risks in using this model

1. **LGD is assumed, not estimated.** Every currency figure scales with it. Sourcing recovery
   data is the highest-value next step.
2. **Single cohort.** One six-month window and one outcome month. No out-of-time validation is
   possible, so the reported PSI of 0.0007 is a formality, not
   evidence of through-time stability.
3. **Regime.** Taiwan 2005 followed a domestic card-debt crisis; a 22%
   default rate should not be read across to another portfolio.
4. **Behavioural dependency.** The model degrades to roughly
   0.63
   ROC-AUC without repayment history. Any application-time use would be a misuse.
5. **Protected characteristics.** Sex, education and marital status are present in the data and
   are used here only for segment monitoring. They are prohibited or restricted inputs for credit
   decisions in many jurisdictions and must be removed, with disparate-impact testing on what
   remains, before deployment.

## 6. Monitoring, if deployed

- Monthly PSI on the score distribution; investigate above 0.10, escalate above 0.25.
- Quarterly recalibration check: predicted versus observed default rate by decile.
- Track the approved book's default rate against the 15% appetite
  target and re-set the cutoff when it drifts.
- Annual challenger comparison against the logistic scorecard; retire the champion if the gap
  closes.

---

*Portfolio project built on public data. Not investment, credit or financial advice.
Reproduce with `python -m src.run_pipeline`.*
