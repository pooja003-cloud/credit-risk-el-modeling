# Model risk and limitations appendix

**Model:** Consumer card behavioural PD model, v1.0 · **Use:** one-month expected loss for
research and portfolio reporting · **Tier:** would be High if used for credit decisions ·
**Basis:** 5,993-account held-out test fold

## Ratings

| Area | Rating | Basis |
| --- | --- | --- |
| Conceptual soundness | Pass with conditions | Sound method; horizon now stated, default definition open (F-01, F-02) |
| Data quality | Pass | 25-record lineage trace, 0.0% discrepancy; integrity checks on load |
| Performance testing | Pass | ROC-AUC 0.778 vs 0.70; 10/10 deciles pass binomial at 99%; calibration slope 0.91 |
| Governance | Not met | Self-assessed, not independent; no inventory, owner or approval (F-10) |

**Opinion: Approved with conditions** for research and portfolio-level reporting. **Not
approved** for credit decisioning.

## Open conditions

| ID | Severity | Condition | Remediation window |
| --- | --- | --- | --- |
| F-02 | Major | Default definition and LGD assumption may not match | 90 days |
| F-03 | Major | Model is slightly over-confident, and optimistic by exposure | 90 days |
| F-05 | Minor | PD is concentrated on one input | 180 days |
| F-08 | Major | No out-of-time validation is possible | 90 days |
| F-09 | Major | Protected characteristics are model inputs | 90 days |
| F-10 | Critical | Validation is not independent of development | Before any use beyond research |

Remediation windows follow the framework's severity scale and would run from the date of any
production approval. F-01, F-04 and F-07 were documentation issues, corrected in this release.

## What the outputs are, and are not

- **One-month figures.** Every PD and the NT$65.7m
  expected loss cover the month after September 2005. They are not 12-month or lifetime numbers.
- **Probably an upper bound on loss.** Default is not defined by days past due, and LGD is an
  assumed 65%. If some defaults cure, true loss is lower, possibly by a lot.
- **Slightly optimistic by exposure,** which pushes the other way but is small: actual-to-expected
  is 1.028 by exposure against 0.981
  by account, worth about 2.7% of loss.
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
| Approved-book default rate | Monthly | Above the 15% appetite target |
| Champion vs challenger expected loss | Quarterly | Gap above 5% |

**Revalidation:** annually if deployed at High tier, and on any change to inputs, default
definition, calibration or cutoff.

Full findings: [`model_validation_report.md`](model_validation_report.md).
