# Model validation report

**Model:** Consumer card behavioural PD model, v1.0 (Gradient boosting (XGBoost), calibrated)
**Scope:** Conceptual soundness, data integrity, implementation, outcome analysis,
sensitivity and stress testing
**Basis:** 5,993-account held-out test fold
**Overall model risk:** Moderate for research and portfolio reporting. Would be a High-tier
model if used for credit decisions.

> **Independence.** These tests were run by the model's developer. Both review frameworks this
> report follows require validation to be independent of development, so read it as a
> structured self-assessment, not a validation opinion. See F-10.

---

## 1. Executive summary

The model discriminates well (0.778 ROC-AUC against a 0.70 threshold),
passes the binomial backtest in all 10 risk deciles at 99% confidence, and its
inputs trace exactly to the raw file. The data pipeline is sound.

The material issues are in what the outputs mean rather than how they are computed. Expected
loss had no stated horizon and is a one-month figure (F-01). The default label is not defined
by days past due, so the charge-off LGD may overstate loss (F-02). And the model is slightly
over-confident, which makes it optimistic by exposure even though it is conservative by
account count (F-03).

12 findings: 1 critical, 4 high, 2 medium, 4 low, 1 informational. Three were documentation errors and are corrected in this
release (F-01, F-04, F-07).

## 2. Model overview

A gradient-boosted tree model predicts the probability that a credit-card account defaults in
the following month, from six months of billing, payment and repayment-status history plus
the credit limit and three demographic fields. Probabilities are calibrated on a validation
fold, floored at 0.03% and capped at 99.97%. A logistic regression on the
same inputs is the challenger. Full specification: [`model_card.md`](model_card.md).

### Post-completion audit checklist

| Area | Check | Finding | Ref |
| --- | --- | --- | --- |
| Target definition | Default clearly defined, appropriate horizon | Partly. Label is 'default payment next month' (one-month horizon). No days-past-due threshold in the source. | F-01, F-02 |
| Feature set | No post-decision variables | Met. All features dated September 2005 or earlier. Origination-only model built to measure the behavioural dependency. | - |
| Model selection | Credible interpretable baseline | Met. Logistic regression benchmark at 0.764 ROC-AUC; boosting adds +0.0136. | F-06 |
| Metrics | Appropriate for an imbalanced target | Met. ROC-AUC, Gini, KS, PR-AUC, precision, recall, F1, Brier, calibration error, decile table. Accuracy not used as a headline. | - |
| Calibration | Predicted PDs match observed rates | Met with a caveat. 10/10 deciles pass binomial at 99%; slope 0.91; optimistic by exposure. | F-03 |
| Expected loss | PD, LGD, EAD consistent and documented | Partly. Formulas and assumptions in code and docs. Horizon and default-definition consistency needed stating. | F-01, F-02, F-04 |
| Scenarios | Internally consistent, labelled hypothetical | Met after review. All three components move together; narratives relabelled. | F-07 |
| Model risk | Limitations, bias, drift, monitoring | Met. Model card, this report, and the one-page appendix. | F-08 to F-11 |
| Business use | Not overstated | Met. Cutoff evaluated retrospectively; no claim of improved lending decisions. | - |

## 3. Findings by domain

### 3.1 Conceptual soundness

**F-01 · High · Remediated (documentation)** - Expected-loss horizon was not stated

The target is default in October 2005 given data to September, so every PD is a one-month PD and the NT$65.7m baseline is a one-month expected loss. It was presented with no horizon, inviting comparison with 12-month IFRS 9 Stage 1 or lifetime CECL figures, which it is not.

*Recommendation:* State the horizon wherever a PD or loss figure appears. A 12-month figure would need a 12-month default label or a documented term-structure assumption.

**F-02 · High · Open** - Default definition and LGD assumption may not match

The source defines the label only as 'default payment next month' and gives no days-past-due threshold. The 65% LGD is a charge-off-scale loss. If the label behaves more like a missed payment than a terminal default, some flagged accounts will cure, and a charge-off LGD overstates loss on them.

*Recommendation:* Treat the expected-loss figures as an upper bound under that reading. With account-level cure or roll-rate data, either redefine default at 90+ days past due or apply a cure-adjusted LGD.

**F-09 · High · Open** - Protected characteristics are model inputs

Sex, education and marital status are inputs to the champion model, not only segment labels. Several are prohibited or restricted for credit decisions in many jurisdictions.

*Recommendation:* Drop them, re-fit, measure the discrimination lost, and test the remaining behavioural inputs for disparate impact.

**F-12 · Informational · Open** - Simplifications in EAD and LGD

EAD uses the September statement balance as the drawn balance, so payments since the statement are not reflected. LGD is one figure for every segment. Origination-only AUC of 0.63 confirms the behavioural scope.

*Recommendation:* Acceptable as documented assumptions; revisit with recovery data.

### 3.2 Data integrity

A 25-record trace rebuilt 8 model inputs from the raw file with
independent arithmetic and compared them to what the model received. Worst discrepancy:
0.00% against a 0.5% tolerance. The loader also checks the
file against the published row, default and default-rate counts before anything runs.

**F-08 · High · Open** - No out-of-time validation is possible

Every account shares one six-month window and one outcome month. The split is random, so the PSI of 0.0007 says nothing about stability through time.

*Recommendation:* Re-fit and test on at least one later cohort before any production use.

**F-11 · Low · Open** - Population conditioned on survival

Only accounts active across the whole window are present. Accounts closed or charged off earlier are missing, so PDs are conditional on surviving to September 2005.

*Recommendation:* Appropriate for a behavioural score; state it as a scope limit.

### 3.3 Implementation verification

The logistic-regression challenger was scored on identical inputs. Portfolio expected loss
differs by 1.9% and mean PD by 0.8%, inside
the 5% tolerance. Rank correlation between the two is 0.89. Unit tests
cover the exposure, expected-loss, stress and validation arithmetic, and a separate script
recomputes every headline figure from the scored file without using the project's own code.

**F-06 · Low · Open** - Champion and challenger agree in aggregate, not account by account

Portfolio expected loss differs by 1.9% and mean PD by 0.8%, inside the 5% tolerance. But the average account-level gap is 4.9 percentage points and 34% of accounts differ by more than 5 points.

*Recommendation:* Model choice matters for account-level decisions even though it barely moves portfolio loss. Document the champion choice on account-level grounds.

### 3.4 Outcome analysis

**F-03 · Medium · Open** - Model is slightly over-confident, and optimistic by exposure

Calibration slope is 0.91 (1.0 is ideal): PDs are too low in the safest bands and too high in the riskiest. By account count the model is conservative (actual-to-expected 0.981). By exposure it is optimistic (1.028), because the under-predicted low-risk accounts carry the largest balances. Portfolio expected loss is therefore understated by about 2.7% of the realised figure.

*Recommendation:* Refit calibration with a slope term, or hold a documented margin of conservatism. Monitor the exposure-weighted actual-to-expected ratio, not only the count-weighted one.

**F-04 · Low · Remediated (documentation)** - Expected-loss back-test claimed more than it tests

Documentation said the back-test checked PD and EAD against outcomes. The same EAD formula and LGD appear on both sides of the comparison and cancel, so it checks only exposure-weighted PD calibration. Its 2.7% gap is the F-03 effect.

*Recommendation:* Wording corrected in the README and risk memo.

### 3.5 Sensitivity and stress testing

**F-05 · Medium · Open** - PD is concentrated on one input

Adding one month to latest arrears status moves mean PD by +25%. No proportional shock to a continuous driver, even at 50%, moves it by more than 3.9%. Errors in the latest repayment-status field would pass almost directly into PD and loss.

*Recommendation:* Put data-quality controls on the repayment-status feed and monitor that field's distribution separately from the overall score.

**F-07 · Low · Remediated (documentation)** - Scenario narratives cited figures the model does not use

Narratives referred to specific unemployment moves. No macroeconomic variable is modelled and nothing links those figures to the multipliers.

*Recommendation:* Narratives relabelled as hypothetical, with severity stated as judgemental.

### 3.6 Governance

**F-10 · Critical · Open** - Validation is not independent of development

These tests were run by the model's developer. Both frameworks require independent validation, so this is a structured self-assessment.

*Recommendation:* Restrict use to research and reporting until independently validated.

## 4. Outcome analysis results

**Discrimination.** ROC-AUC 0.778, Gini 0.556, KS
0.427. Above the 0.70 AUC threshold.

**Binomial backtest by decile, 99% confidence.** 10 of 10 pass.

| Decile | Accounts | Observed | Expected | 99% interval | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | 600 | 25 | 18.8 | 9-31 | Pass |
| 2 | 599 | 44 | 36.0 | 22-52 | Pass |
| 3 | 599 | 54 | 50.8 | 34-69 | Pass |
| 4 | 599 | 66 | 60.1 | 42-80 | Pass |
| 5 | 600 | 99 | 88.0 | 66-111 | Pass |
| 6 | 599 | 95 | 109.2 | 85-134 | Pass |
| 7 | 599 | 110 | 126.4 | 101-153 | Pass |
| 8 | 599 | 167 | 159.8 | 132-188 | Pass |
| 9 | 599 | 248 | 265.8 | 235-297 | Pass |
| 10 | 600 | 418 | 436.3 | 408-464 | Pass |

**Hosmer-Lemeshow.** Statistic 16.81. p = 0.079
with 10 degrees of freedom, the usual choice for a model scored out of
sample, and p = 0.032 with the in-sample convention of
8. Borderline: it passes at 5% under the appropriate convention and fails
under the stricter one. With about 600 accounts per group the test has high power, so the
group table matters more than the verdict, and it shows the pattern in F-03.

**Calibration diagnostics.**

| Measure | Value |
| --- | --- |
| Calibration slope (ideal 1.0) | 0.906 |
| Calibration intercept (ideal 0.0) | -0.130 |
| Predicted / observed rate, by account | 22.55% / 22.13% |
| Predicted / observed rate, by exposure | 18.40% / 18.91% |
| Actual-to-expected, by account | 0.981 |
| Actual-to-expected, by exposure | 1.028 |

**Input sensitivity.** Each driver shocked alone, by +/-10%, 20% and 50% for continuous inputs
and +/-1 month for delinquency counts. Largest response per input:

| Input | Largest-response shock | Change in mean PD |
| --- | --- | --- |
| Latest months past due | +1 month | +24.9% |
| Worst months past due | -1 month | -10.3% |
| Months 2+ cycles late | +1 month | +5.3% |
| Credit limit | -50% | +3.9% |
| Latest bill | -50% | +3.2% |
| Available credit | -50% | +2.5% |
| Average bill (6 months) | -50% | +2.3% |
| Lowest payment ratio | +50% | -1.9% |
| Average utilisation | +50% | -0.3% |

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
