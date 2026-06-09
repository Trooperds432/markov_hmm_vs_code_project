from __future__ import annotations

from hmmlearn.hmm import GMMHMM, GaussianHMM

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.hmm_model import build_hmm


def test_gaussian_hmm_factory_passes_parameters():
    cfg = PipelineConfig(
        n_states=4,
        covariance_type="diag",
        hmm_min_covar=0.01,
        hmm_transmat_prior=5.0,
        hmm_params="tmc",
        hmm_init_params="mc",
    )

    model = build_hmm(cfg, random_state=11)

    assert isinstance(model, GaussianHMM)
    assert model.n_components == 4
    assert model.covariance_type == "diag"
    assert model.min_covar == 0.01
    assert model.transmat_prior == 5.0
    assert model.params == "tmc"
    assert model.init_params == "mc"


def test_gmm_hmm_factory_passes_parameters():
    cfg = PipelineConfig(
        hmm_model_type="gmm",
        n_states=3,
        hmm_n_mix=2,
        covariance_type="diag",
        hmm_min_covar=0.002,
        hmm_transmat_prior=10.0,
    )

    model = build_hmm(cfg, random_state=7)

    assert isinstance(model, GMMHMM)
    assert model.n_components == 3
    assert model.n_mix == 2
    assert model.covariance_type == "diag"
    assert model.min_covar == 0.002
    assert model.transmat_prior == 10.0
