# Model card

## Model details

| | |
| --- | --- |
| **Name** | Consumer card behavioural probability-of-default model |
| **Version** | 1.0 |
| **Type** | Binary classifier, gradient-boosted decision trees (XGBoost), post-hoc calibrated with a PD floor |
| **Benchmark / challenger** | Penalised logistic regression (scorecard-style), same feature set |
| **Output** | Calibrated probability that the account defaults in the following month |
| **Training data** | UCI *Default of Credit Card Clients* (Taiwan, 2005), 60% stratified sample |
| **Selection data** | 20% validation fold - hyper-parameters, probability calibration and decision cutoff |
| **Evaluation data** | 20% test fold, scored once |
| **Owner** | Portfolio project. Not a production model, not validated by an independent model-risk function |
| **Reproduce** | `python -m src.run_pipeline` (fixed seed 42) |

## Intended use

**In scope.** Ranking existing card accounts by default risk one month ahead; portfolio
expected-loss estimation and scenario analysis; collections and limit-management
prioritisation; teaching and demonstration of an end-to-end credit-risk workflow.

**Out of scope.**

- **Application / origination decisions.** The model consumes six months of repayment history
  that does not exist when an account is opened. The origination-only variant reported in the
  README is the honest benchmark for that decision, and it is much weaker.
- **Any real lending decision.** The model is trained on one 2005 Taiwanese cohort and has had
  no independent validation, no fair-lending testing and no production monitoring.
- **Regulatory capital or IFRS 9 / CECL reporting.** The loss-given-default input is an
  assumption, there is no lifetime PD term structure, and there is no macroeconomic model.
- **Individual adverse-action decisions.** SHAP reason codes are produced for illustration;
  they have not been reviewed against any jurisdiction's adverse-action requirements.

## Inputs

28 features derived from the credit line, six months of billing, payment and repayment-status
history, and three application demographics. Full definitions in
[`data_dictionary.md`](data_dictionary.md); construction in `src/data_prep.py`.

Feature families:

- **Exposure and utilisation** - credit line, utilisation level, trend, volatility, months above
  90% utilisation, available credit.
- **Repayment behaviour** - payment-to-bill ratios (latest, mean, minimum), six-month
  paid-to-billed ratio, months with no payment.
- **Delinquency** - latest and worst months past due, count of delinquent months, count of months
  two or more cycles down, direction of travel, and counts of revolving, full-payment and
  inactive months.
- **Balance dynamics** - latest and mean bill, six-month balance growth.
- **Demographics** - sex, education, marital status, age.

**Deliberately excluded.** The `ID` column; the raw `PAY_*` codes as a numeric severity scale
(-2 "no usage", -1 "paid in full" and 0 "revolving" are states, not a smaller amount of
delinquency); anything dated in or after the October 2005 outcome month.

## Target

`default.payment.next.month` - default on the October 2005 payment. Base rate 22.13% after
cleaning.

## Performance

Reported on the held-out test fold; the exact figures are regenerated into the
[README](../README.md) and [risk memo](risk_memo.md) by `python -m src.build_reports`, and the
underlying table is `outputs/tables/model_comparison.csv`.

Metrics reported: ROC-AUC, Gini, PR-AUC, KS, precision, recall, F1 at explicit cutoffs, Brier
score, expected calibration error, default rate by predicted-risk decile, and per-segment AUC
and calibration gap.

**Accuracy is not reported as a headline.** On a 22% base rate, approving every account is 78%
accurate and identifies no defaults.

## Calibration

Probabilities are calibrated on the validation fold using a frozen base estimator, so
calibration never refits on training data. This matters because expected loss multiplies PD
levels: a model that ranks perfectly but is systematically 30% too low produces a loss number
that is 30% too low.

**Method selection.** Isotonic regression and Platt scaling are compared by five-fold
cross-validation *within* the validation fold - fitting a calibrator and scoring it on the same
rows would flatter isotonic, which is the more flexible of the two - and the method with the
lower mean out-of-fold Brier score is refitted on the whole validation fold. The choice made
for each model is recorded in `outputs/models/manifest_behavioural.json`.

**PD floor and cap.** Calibrated probabilities are floored at 0.03% and capped at 99.97%.
Isotonic calibration is a step function and assigns a PD of exactly 0 or exactly 1 to accounts
falling in its outermost steps. Neither is a defensible estimate: a PD of zero asserts that no
loss is possible and removes that account from the expected-loss calculation entirely, and a PD
of one claims certainty from a handful of observations. Basel applies a 0.03% PD floor to retail
exposures for the same reason, and that value is used here.

## Ethical and fairness considerations

The dataset contains **sex, education and marital status**. These are used here for segment
monitoring and are present in the model's input set for the demonstration. In many
jurisdictions sex and marital status are prohibited inputs for credit decisions, and education
is restricted or contested.

Before any deployment: remove those inputs, re-fit, quantify the discrimination lost, and run
disparate-impact testing (approval-rate ratios and error-rate parity across protected groups) on
the remaining behavioural features. Behavioural features are not automatically neutral - they
can proxy for protected characteristics.

Per-segment performance is published in `outputs/tables/segment_performance.csv` so that a
reviewer can see where the model is weaker rather than only a portfolio-level average.

## Limitations

1. **Single cohort.** One six-month observation window and one outcome month. No out-of-time
   validation is possible; the split is stratified-random. The reported population stability
   index is therefore a formality, not evidence of stability through time.
2. **Regime specificity.** Taiwan, 2005, following a domestic card-debt crisis. The default rate
   is far above a normal through-the-cycle level for a card book.
3. **LGD is assumed.** No recovery, collection or charge-off data exists in this dataset. Every
   currency figure scales one-for-one with the assumption.
4. **EAD is partly assumed.** The credit conversion factor on the undrawn line is judgemental.
5. **No macroeconomic linkage.** Unemployment and interest rates enter only as scenario
   multipliers on the odds of default, not as model inputs, so the scenarios are illustrative
   rather than macro-conditioned forecasts.
6. **Small segments.** Some demographic segments have few hundred accounts; per-segment metrics
   there are noisy, and segments below 50 accounts or 5 defaults are suppressed entirely.
7. **Mild optimism.** Train ROC-AUC exceeds test ROC-AUC by roughly 0.03. Consistent with light
   over-fitting that the validation-fold selection did not fully remove.

## Monitoring plan (if it were deployed)

| Check | Frequency | Trigger |
| --- | --- | --- |
| Population stability index on the score distribution | Monthly | Investigate > 0.10, escalate > 0.25 |
| Predicted vs observed default rate by decile | Quarterly | Any decile off by more than 5 percentage points |
| Default rate of the approved book | Monthly | Above the risk-appetite target |
| Champion vs challenger ROC-AUC | Annually | Gap closes - retire the champion |
| Feature drift on the top ten drivers | Quarterly | Distribution shift on any dominant feature |

## Reference

Yeh, I.-C. and Lien, C.-H. (2009). "The comparisons of data mining techniques for the predictive
accuracy of probability of default of credit card clients." *Expert Systems with Applications*,
36(2), 2473-2480.
