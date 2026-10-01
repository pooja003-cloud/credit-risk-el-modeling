# Risk memo

**To:** Credit Risk Committee
**From:** Model development
**Subject:** Consumer card portfolio - PD model, expected credit loss and scenario results
**Date:** 2026-10-01
**Basis:** UCI *Default of Credit Card Clients* (Taiwan, 2005); 5,993-account
held-out test fold

---

## 1. Recommendation: APPROVE WITH CONDITIONS

Approve the calibrated gradient-boosting PD model for **one-month expected-loss reporting and
portfolio risk analysis** on this book. Do not use it for credit decisions yet. Three conditions
stand in the way, set out in the [model risk appendix](model_risk_appendix.md): sex, education
and marital status are model inputs; there has been no out-of-time test; and the validation was
not independent.

The 44.7% PD cutoff in section 3 is a **proposed** policy for
when those conditions are met. It is not a recommendation to start declining accounts.

The model is behavioural. It needs six months of repayment history, so it cannot score new
applicants. Keep the logistic-regression scorecard as challenger: 0.764
ROC-AUC against 0.778, a gap of +0.0136, and
interpretable without tooling.

**Rating rationale.** Positive: strong discrimination, a binomial pass in all ten risk
deciles at 99%, inputs traced exactly to source, and loss assumptions stated wherever they are
used. Negative: a one-month horizon only, a default label with no days-past-due definition,
slight optimism by exposure, and heavy reliance on a single input.

## 2. Model performance

| Measure | Champion | Logistic benchmark |
| --- | --- | --- |
| ROC-AUC (test) | 0.7779 | 0.7643 |
| Gini | 0.5557 | 0.5286 |
| PR-AUC (base rate 22.1%) | 0.5516 | 0.5168 |
| KS | 0.4271 | - |
| Brier | 0.1351 | 0.1381 |
| Expected calibration error | 0.0174 | - |

ROC-AUC across folds: 0.811 train, 0.796 validation, 0.778
test. Mild optimism, not over-fitting.

**Rank-ordering is monotone except between deciles 5 and 6**, where the observed rates are within 0.6% of each other and the ordering is not statistically meaningful at this sample size. The riskiest decile defaults at
70% versus 4.2% in the safest,
and the top two deciles capture 50% of defaults.

**Calibration.** Expected loss multiplies PD levels, so the probabilities are calibrated on the
validation fold before they are used anywhere. After calibration, mean predicted PD is
22.55% against an observed 22.13%.

**Segments.** The weakest is "Other" (education label),
0.580 ROC-AUC on 90 accounts. None falls to random.

## 3. The two errors

- **False negative** - an account is approved and defaults. Cost: `LGD x EAD`.
- **False positive** - an account is declined that would have paid. Cost: forgone margin on the
  balance that customer would have carried.

Under the working assumptions these differ by roughly a factor of ten, which is why the 0.5
cutoff that `predict()` returns by default is the wrong place to stand.

At the proposed 44.7% cutoff, applied retrospectively to the
test fold:

| | |
| --- | --- |
| Approval rate | 85% (922 accounts declined) |
| Default rate of the approved book | **14.9%** (from 22.1%) |
| Default rate of the declined population | 61.8% |
| Defaults avoided / retained | 570 / 756 |
| Good accounts turned away | 352 |
| Precision / recall at the cutoff | 0.618 / 0.430 |

No live decisions were made and every declined account was in fact given credit, so this
shows what the rule would have done, not that lending improved.

A purely cost-minimising cutoff would sit at 5.1% and decline
92% of the book. That comes from weighing one period of loss against one
period of margin, and it swings from 5% to
23% as the assumed margin changes. Risk appetite is the more
stable anchor.

## 4. Expected credit loss

`EL = PD x LGD x EAD`, on the test fold, baseline assumptions. **Horizon: one month.** Every
PD is the chance of default in October 2005, so this is one month's expected loss, not a
12-month or lifetime figure.

| | |
| --- | --- |
| Committed limits | NT$1,007.5m |
| Drawn balances | NT$307.7m |
| Exposure at default (balance + 35% of undrawn) | **NT$549.0m** |
| Exposure-weighted PD | 18.40% |
| Assumed LGD | 65% |
| **Expected loss** | **NT$65.7m** (11.96% of exposure) |

**Partial back-test.** Holding LGD fixed, the loss implied by the accounts that actually
defaulted is NT$67.5m against a modelled
NT$65.7m, 2.7%
lower. The same EAD and LGD sit on both sides and cancel, so this checks only exposure-weighted
PD calibration. The gap is real: low-risk accounts are slightly under-predicted and carry the
largest balances, so the model is optimistic by exposure even though it is conservative by
account count.

### Scenarios

| Scenario | PD odds | LGD | CCF | Expected loss | vs baseline |
| --- | --- | --- | --- | --- | --- |
| Baseline | x1.00 | 65% | 35% | NT$65.7m | +0% |
| Moderate deterioration | x1.50 | 75% | 45% | NT$107.5m | +64% |
| Severe deterioration | x2.25 | 85% | 55% | NT$168.9m | +157% |

The PD stress is applied to the odds of default (a constant shift on the log-odds scale), so
probabilities stay bounded and the stress is proportionate across the risk spectrum. Severities
are judgemental and are not calibrated to a published macroeconomic model.

The planning figure for the committee is the severe scenario, roughly
2.6x the baseline loss. The sensitivity grid in
`outputs/tables/sensitivity_grid.csv` shows that a substantial share of that movement comes from
the LGD assumption rather than from the model.

## 5. Risks and mitigants

1. **Risk:** default is defined only as "default payment next month", with no days-past-due
   threshold, while LGD is a charge-off-scale 65%. If some defaults cure, loss
   is overstated.
   **Mitigant:** treat the figures as an upper bound; the PD-stress by LGD grid shows the range.
2. **Risk:** LGD is assumed, not estimated, and every currency figure scales with it.
   **Mitigant:** stated wherever it is used; recovery data is the first thing to source.
3. **Risk:** calibration slope of 0.91; loss understated by about
   2.7% by exposure.
   **Mitigant:** all ten deciles pass the binomial backtest; monitor exposure-weighted
   actual-to-expected quarterly.
4. **Risk:** one more month of latest arrears moves mean PD +25%, so
   errors in that field flow straight into loss.
   **Mitigant:** lineage check traces it to source exactly; monitor its distribution monthly.
5. **Risk:** sex, education and marital status are **model inputs**, restricted or prohibited for
   credit decisions in many jurisdictions.
   **Mitigant:** not approved for decisioning; removal and disparate-impact testing are conditions.
6. **Risk:** one cohort, one outcome month, Taiwan 2005; PSI of
   0.0007 reflects a random split, not stability over time.
   Without repayment history the model falls to 0.63 ROC-AUC.
   **Mitigant:** use restricted to this book and to existing accounts; out-of-time test required
   before production.

## 6. Monitoring

Monitoring indicators, triggers and the revalidation cycle are in the
[model risk appendix](model_risk_appendix.md). Full findings are in the
[validation report](model_validation_report.md).

---

*Portfolio project built on public data. Not investment, credit or financial advice.
Reproduce with `python -m src.run_pipeline`.*
