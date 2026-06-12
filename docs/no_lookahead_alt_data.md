# No-Lookahead Alt Data

All alternative-data joins are backward-looking as-of joins. Each data source applies a publication
lag before it can affect a market date.

Glassdoor reviews use `review_date + glassdoor_publication_lag_days`. WSB mentions use
`date + wsb_publication_lag_days`. Valuation rows use `date + valuation_publication_lag_days`.

Future reviews, future mentions, future EPS, future intrinsic values, labels, targets, and realised
future alpha fields are blocked from HMM observations.
