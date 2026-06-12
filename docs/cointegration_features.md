# Cointegration Features

Cointegration and spread-stress features compare the asset with configured economic peers, sector
proxies, or the benchmark fallback. The code avoids scanning many pairs and does not select a peer
by best p-value, which helps avoid multiple-comparison bias.

Features include spread z-score, spread volatility, spread half-life, break flags, and optional
cointegration p-values through `statsmodels`. If `statsmodels` is unavailable, p-value fields are
missing and structured warnings are emitted.
