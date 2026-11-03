# tests/test_switching_ssm.py
"""Unit tests for the Switching SSM."""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.variational_switching import SwitchingSSM


def make_synthetic_obs(T=200, obs_dim=5, n_regimes=3, seed=42):
    np.random.seed(seed)
    segment = T // n_regimes
    parts = []
    for k in range(n_regimes):
        mu = np.random.randn(obs_dim) * (k + 1)
        parts.append(np.random.randn(segment, obs_dim) * 0.5 + mu)
    return np.vstack(parts)


def test_switching_ssm_output_shapes():
    T, obs_dim, state_dim, K = 150, 5, 3, 3
    obs = make_synthetic_obs(T, obs_dim, K)
    model = SwitchingSSM(n_regimes=K, state_dim=state_dim, obs_dim=obs_dim)
    result = model.filter(obs)

    assert result.regime_probs.shape == (T, K)
    assert result.filtered_means.shape == (T, state_dim)
    assert result.viterbi_path.shape == (T,)


def test_regime_probs_sum_to_one():
    T, obs_dim, state_dim, K = 100, 4, 2, 3
    obs = make_synthetic_obs(T, obs_dim, K)
    model = SwitchingSSM(n_regimes=K, state_dim=state_dim, obs_dim=obs_dim)
    result = model.filter(obs)

    sums = result.regime_probs.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_viterbi_path_valid_regimes():
    T, obs_dim, state_dim, K = 100, 4, 2, 4
    obs = make_synthetic_obs(T, obs_dim, K)
    model = SwitchingSSM(n_regimes=K, state_dim=state_dim, obs_dim=obs_dim)
    result = model.filter(obs)

    assert np.all(result.viterbi_path >= 0)
    assert np.all(result.viterbi_path < K)


def test_fit_keeps_transition_matrix_valid_and_history_finite():
    T, obs_dim, state_dim, K = 200, 5, 3, 3
    obs = make_synthetic_obs(T, obs_dim, K)
    model = SwitchingSSM(n_regimes=K, state_dim=state_dim, obs_dim=obs_dim)
    lls = model.fit(obs, n_iter=5, verbose=False)

    assert len(lls) == 5
    assert np.isfinite(lls).all()
    np.testing.assert_allclose(model.transition_matrix.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(model.transition_matrix > 0)
    np.testing.assert_allclose(model.initial_regime_probs.sum(), 1.0, atol=1e-8)
    assert np.all(model.initial_regime_probs > 0)



def test_filter_rejects_invalid_observations():
    model = SwitchingSSM(n_regimes=2, state_dim=2, obs_dim=3)
    invalid_inputs = [np.ones(3), np.ones((4, 2)), np.array([[1.0, np.nan, 2.0]]), np.empty((0, 3))]
    for invalid in invalid_inputs:
        try:
            model.filter(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid observations should raise ValueError.")


def test_filtered_covariances_are_symmetric_and_positive_semidefinite():
    obs = make_synthetic_obs(T=60, obs_dim=3, n_regimes=2)
    model = SwitchingSSM(n_regimes=2, state_dim=2, obs_dim=3)
    result = model.filter(obs)
    for covariance in result.filtered_covs:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-8)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-8


def test_reported_regime_path_matches_posterior_map():
    obs = make_synthetic_obs(T=60, obs_dim=3, n_regimes=2)
    model = SwitchingSSM(n_regimes=2, state_dim=2, obs_dim=3)
    result = model.filter(obs)
    np.testing.assert_array_equal(result.viterbi_path, np.argmax(result.regime_probs, axis=1))


def test_fit_does_not_modify_state_space_parameters():
    obs = make_synthetic_obs(T=60, obs_dim=3, n_regimes=2)
    model = SwitchingSSM(n_regimes=2, state_dim=2, obs_dim=3)
    before = [(r.A.copy(), r.C.copy(), r.Q.copy(), r.R.copy()) for r in model.regimes]
    model.fit(obs, n_iter=2, verbose=False)
    for regime, original in zip(model.regimes, before):
        for current, expected in zip((regime.A, regime.C, regime.Q, regime.R), original):
            np.testing.assert_array_equal(current, expected)

if __name__ == "__main__":
    test_switching_ssm_output_shapes()
    test_regime_probs_sum_to_one()
    test_viterbi_path_valid_regimes()
    test_fit_keeps_transition_matrix_valid_and_history_finite()
    test_filter_rejects_invalid_observations()
    test_filtered_covariances_are_symmetric_and_positive_semidefinite()
    test_reported_regime_path_matches_posterior_map()
    test_fit_does_not_modify_state_space_parameters()
    print("All Switching SSM tests passed!")
