"""Tests for causal continuation across contiguous observation chunks."""

import numpy as np
import pytest

from src.inference.variational_switching import SwitchingSSM


def test_filter_state_continuation_matches_single_pass():
    rng = np.random.default_rng(123)
    observations = rng.normal(size=(40, 3))

    model = SwitchingSSM(n_regimes=2, state_dim=2, obs_dim=3)

    full = model.filter(observations)
    first = model.filter(observations[:17])
    second = model.filter(observations[17:], initial_state=first.final_state)

    np.testing.assert_allclose(
        second.regime_probs, full.regime_probs[17:], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        second.filtered_means, full.filtered_means[17:], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        second.filtered_covs, full.filtered_covs[17:], rtol=1e-10, atol=1e-10
    )


def test_filter_rejects_malformed_continuation_state():
    model = SwitchingSSM(n_regimes=2, state_dim=2, obs_dim=3)
    observations = np.zeros((3, 3))

    with pytest.raises(ValueError, match="initial_state must contain"):
        model.filter(observations, initial_state={"regime_probs": np.array([0.5, 0.5])})
