# Alternative-Data Overlays

The alt-data layer is optional and disabled by default. It does not replace the HMM regime model.
Glassdoor, WSB, and valuation features act as confidence scalers, risk gates, diagnostics, and
portfolio rank modifiers.

`alt_data_position_multiplier` is the product of Glassdoor, WSB, and valuation multipliers, clipped
to `[0.25, 1.50]`. It only scales an existing HMM-driven target position when
`use_alt_data_position_overlay=true`; it never creates exposure from zero.

Outputs include `alt_data_overlay.csv`, `alt_data_feature_summary.csv`, and `alt_data_warnings.csv`.
