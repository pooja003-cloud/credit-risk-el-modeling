# Data dictionary

Source: UCI Machine Learning Repository, *Default of Credit Card Clients* (Yeh & Lien, 2009).
30,000 Taiwanese credit-card accounts, observed April-September 2005, with the October 2005
default outcome. Amounts are in New Taiwan dollars (NT$).

Timeline convention used throughout this project - index 1 is the **most recent** month:

| Index | Month | `PAY_i` | `BILL_AMTi` | `PAY_AMTi` |
| --- | --- | --- | --- | --- |
| 1 | September 2005 | repayment status | bill issued | amount paid |
| 2 | August 2005 | " | " | " |
| 3 | July 2005 | " | " | " |
| 4 | June 2005 | " | " | " |
| 5 | May 2005 | " | " | " |
| 6 | April 2005 | " | " | " |

The outcome month is **October 2005**, one month after the newest feature.

---

## Raw columns

| Column | Description | Notes |
| --- | --- | --- |
| `ID` | Account identifier | Dropped before modelling |
| `LIMIT_BAL` | Credit line, including any family supplementary credit | NT$10,000 - NT$1,000,000 |
| `SEX` | 1 = male, 2 = female | Used for segment monitoring; see the model card |
| `EDUCATION` | 1 = graduate school, 2 = university, 3 = high school, 4 = other | Codes 0, 5, 6 are undocumented and are collapsed into 4 (468 records) |
| `MARRIAGE` | 1 = married, 2 = single, 3 = other | Code 0 is undocumented and is collapsed into 3 (377 records) |
| `AGE` | Age in years | 21 - 79 |
| `PAY_0` ... `PAY_6` | Repayment status | Renamed `PAY_1`...`PAY_6`. -2 = no usage, -1 = paid in full, 0 = revolving (minimum paid), 1-8 = months past due |
| `BILL_AMT1` ... `BILL_AMT6` | Bill statement amount | Can be negative - a credit balance from an overpayment, not an error (3,932 values) |
| `PAY_AMT1` ... `PAY_AMT6` | Amount paid in that month | Settles the previous month's bill |
| `default.payment.next.month` | 1 = defaulted in October 2005 | Renamed `default`. Base rate 22.12% raw, 22.13% after cleaning |

**The `PAY_*` scale is not monotone in risk.** -2 ("no usage") and -1 ("paid in full") are both
"better" than 0 ("revolving"), but they are qualitatively different states rather than negative
amounts of delinquency. The pipeline therefore never feeds the raw codes to a model as a
continuous variable.

---

## Engineered features

### Exposure and utilisation

| Feature | Definition |
| --- | --- |
| `log_limit_bal` | `log(1 + LIMIT_BAL)` |
| `util_1` ... `util_6` | `max(BILL_AMTi, 0) / LIMIT_BAL` |
| `util_latest` | Utilisation in September 2005 |
| `util_mean`, `util_max` | Mean and maximum utilisation over six months |
| `util_trend` | `util_1 - util_6` - rising or falling utilisation |
| `util_volatility` | Standard deviation of the six monthly utilisation values |
| `months_over_90pct_util` | Count of months above 90% utilisation |
| `available_credit` | `max(LIMIT_BAL - BILL_AMT1, 0)` |

### Repayment behaviour

| Feature | Definition |
| --- | --- |
| `pay_ratio_1` ... `pay_ratio_5` | `PAY_AMTi / (max(BILL_AMT(i+1), 0) + 1)`, capped at 2.0 - payment against the bill it settles |
| `pay_ratio_latest`, `pay_ratio_mean`, `pay_ratio_min` | Latest, mean and minimum of the above |
| `paid_to_billed_6m` | Total paid over six months / total billed, capped at 2.0 |
| `months_paid_nothing` | Count of months with a zero payment |
| `total_paid_6m`, `total_billed_6m` | Six-month sums |

### Delinquency

| Feature | Definition |
| --- | --- |
| `dpd_1` ... `dpd_6` | `max(PAY_i, 0)` - months past due, floored at zero |
| `dpd_latest`, `dpd_max` | Most recent and worst arrears status |
| `months_delinquent` | Count of months with any arrears |
| `months_2plus_delinquent` | Count of months two or more cycles down |
| `dpd_trend` | `dpd_1 - dpd_6` - deteriorating or curing |
| `ever_delinquent` | 1 if the account was ever late in the window |
| `revolver_months` | Count of months with `PAY_i == 0` (minimum payment only) |
| `full_payer_months` | Count of months with `PAY_i == -1` (paid in full) |
| `inactive_months` | Count of months with `PAY_i == -2` (no usage) |

### Balance dynamics

| Feature | Definition |
| --- | --- |
| `bill_latest`, `bill_mean` | September bill and six-month mean bill |
| `bill_growth_6m` | `(BILL_AMT1 - BILL_AMT6) / (abs(BILL_AMT6) + 1)`, clipped to [-5, 5] |

### Segment labels (reporting, not all modelled)

| Feature | Definition |
| --- | --- |
| `age_band` | 21-25, 26-30, 31-35, 36-40, 41-50, 51-60, 60+ |
| `limit_band` | Credit-line quintile, Q1 lowest to Q5 highest |
| `sex_label`, `education_label`, `marriage_label` | Readable versions of the coded columns |

---

## Feature sets

| Set | Features | Purpose |
| --- | --- | --- |
| **Behavioural** | 28 - everything above except the segment-only bands | The champion model. What an issuer observes on an existing account |
| **Origination** | 5 - `log_limit_bal`, `AGE`, sex, education, marital status | What is known when an account is opened. The fair benchmark for an application scorecard |

---

## Derived risk quantities

| Quantity | Definition | Status |
| --- | --- | --- |
| `pd` | Calibrated model output | **Estimated** |
| `ead` | `min(BILL_AMT1, LIMIT_BAL) + CCF x undrawn line` | Balance and line observed; **CCF assumed** (0.35 baseline) |
| `lgd` | Configured constant | **Assumed** (0.65 baseline) - no recovery data exists in this dataset |
| `expected_loss` | `pd x lgd x ead` | Inherits both assumptions |
| `risk_decile` | Decile of predicted PD, 1 = safest | Derived |

---

## Cleaning summary

| Step | Records affected |
| --- | --- |
| Exact duplicate rows dropped (identical except `ID`) | 35 |
| `EDUCATION` codes 0/5/6 collapsed into "Other" | 468 |
| `MARRIAGE` code 0 collapsed into "Other" | 377 |
| Negative bill amounts retained as credit balances | 3,932 |
| Missing values | 0 |
| **Rows after cleaning** | **29,965** |

Machine-readable version: `outputs/tables/data_preparation_report.json`.
