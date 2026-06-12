# WSB Attention

WSB inputs are local mention tables or user-provided exports. Reddit/PRAW/Pushshift are not required
by the base install.

The WSB feature block applies `wsb_publication_lag_days`, aggregates mentions by date and ticker,
computes mention z-scores, mention changes, cross-sectional rank percentile, title/body/cashtag
shares, attention volatility, attention shock, and meme-risk score.

WSB attention is treated as crowding and volatility risk, not a bullish fundamental signal.
