# Glassdoor Features

Glassdoor inputs are local CSV/parquet exports only. The project does not scrape Glassdoor.

The feature block normalizes common column aliases, applies `glassdoor_publication_lag_days`, and
aggregates trailing review windows. It reports rating means/stds, positive/neutral/negative shares,
review counts, recency, review intensity, sentiment score, stress score, and stress z-score.

Glassdoor features are slow employee-sentiment and internal-stress diagnostics. They are best used as
regime priors or position-size modifiers, not short-term buy/sell signals.
