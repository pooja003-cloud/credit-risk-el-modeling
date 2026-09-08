# Data source

`default_of_credit_card_clients.csv` - UCI Machine Learning Repository,
**Default of Credit Card Clients** (Yeh & Lien, 2009).

- Canonical page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- 30,000 Taiwanese credit-card accounts, observed April-September 2005, with a binary
  flag for default on the October 2005 payment.
- Amounts are in New Taiwan dollars (NT$).
- Licence: CC BY 4.0 (per the UCI repository listing).

The copy committed here is the widely mirrored CSV form of the original `.xls` file
(column `PAY_0` retained as distributed; target column named
`default.payment.next.month`). It is verified on load against the published dataset
statistics - 30,000 rows, 6,636 defaults, a 22.12% default rate and no missing values -
by `src.data_prep.validate_raw`, which raises if the file does not match. The file is
committed so that the pipeline runs offline and every result is reproducible without a
network fetch.

## Citation

Yeh, I.-C. and Lien, C.-H. (2009). "The comparisons of data mining techniques for the
predictive accuracy of probability of default of credit card clients."
*Expert Systems with Applications*, 36(2), 2473-2480.
