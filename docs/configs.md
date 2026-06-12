# Configs

Configs use `PipelineConfig` field names. New configs should use canonical portfolio fields:
`portfolio_enabled` and string `portfolio_mode` (`single_asset`, `top_k`, `risk_budgeted`).
The loader still accepts the old boolean `portfolio_mode` alias for compatibility.

Example configs in `configs/` are scoped by purpose: `basic_hmm.json` is the conservative
baseline, research-public configs enable statistical diagnostics, portfolio configs exercise
allocation modes, `paper_safe.json` is simulation-only, and `full_research_stack.json` is a
slower experimental smoke profile.

Validation emits `info`, `warning`, or `error`. Errors stop the run. Warnings are printed and
captured in reporting outputs where possible.
