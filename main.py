import argparse
from pathlib import Path

from src.markov_hmm_trader.config import (
    PROFILE_OVERRIDES,
    apply_overrides,
    apply_profile,
    apply_quick_mode,
    load_config,
)
from src.markov_hmm_trader.dependency_check import print_dependency_health
from src.markov_hmm_trader.runner import (
    run_model_comparison_mode,
    run_nested_mode,
    run_paper_mode,
    run_portfolio_mode,
    run_single,
    run_un