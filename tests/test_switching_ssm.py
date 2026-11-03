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
    assert result.regime_filtered_means.shape == (T, K, state_dim)


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


def test_fit_updates_state_space_parameters_and_keeps_covariances_valid():
    obs = make_synthetic_obs(T=60, obs_dim=3, n_regimes=2)
    model = SwitchingSSM(n_regimes=2, state_dim=2, obs_dim=3)
    before = [(r.A.copy(), r.C.copy(), r.Q.copy(), r.R.copy()) for r in model.regimes]
    model.fit(obs, n_iter=2, verbose=False)
    changed = False
    for regime, original in zip(model.regimes, before):
        for current, expected in zip((regime.A, regime.C, regime.Q, regime.R), original):
            changed = changed or not np.allclose(current, expected)
        for covariance in (regime.Q, regime.R):
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-8)
            assert np.linalg.eigvalsh(covariance).min() > 0
    assert changed, "The approximate M-step should update at least one state-space parameter."


def make_known_dynamics_sequence(T=500, seed=7):
    """Generate observations from a one-regime linear Gaussian state-space model."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.82, 0.08], [0.0, 0.76]])
    C = np.array([[1.1, 0.2], [-0.15, 0.9], [0.4, -0.3]])
    Q = np.diag([0.025, 0.035])
    R = np.diag([0.04, 0.05, 0.03])
    states = np.zeros((T, 2))
    observations = np.zeros((T, 3))
    states[0] = rng.normal(size=2)
    for t in range(1, T):
        states[t] = A @ states[t - 1] + rng.multivariate_normal(np.zeros(2), Q)
    for t in range(T):
        observations[t] = C @ states[t] + rng.multivariate_normal(np.zeros(3), R)
    return observations, states, A, C, Q, R


def test_known_single_regime_dynamics_synthetic_experiment():
    observations, _, true_a, true_c, true_q, true_r = make_known_dynamics_sequence()
    model = SwitchingSSM(n_regimes=1, state_dim=2, obs_dim=3)
    model.set_regime_params(
        0,
        A=np.eye(2) * 0.7,
        C=np.ones((3, 2)) * 0.15,
        Q=np.eye(2) * 0.08,
        R=np.eye(3) * 0.12,
    )

    before = model.filter(observations)
    history = model.fit(observations, n_iter=5, verbose=False)
    after = model.filter(observations)

    assert np.isfinite(history).all()
    assert np.isfinite(after.log_likelihood)
    np.testing.assert_allclose(after.regime_probs, 1.0, atol=1e-10)
    for matrix in (model.regimes[0].A, model.regimes[0].C,
                   model.regimes[0].Q, model.regimes[0].R):
        assert np.isfinite(matrix).all()
    for covariance in (model.regimes[0].Q, model.regimes[0].R):
        assert np.linalg.eigvalsh(covariance).min() > 0
    # The synthetic ground-truth parameters are retained for reproducible
    # follow-up parameter-error analysis; this smoke test does not claim recovery.
    assert true_a.shape == model.regimes[0].A.shape
    assert true_c.shape == model.regimes[0].C.shape
    assert true_q.shape == model.regimes[0].Q.shape
    assert true_r.shape == model.regimes[0].R.shape
    assert before.regime_probs.shape == after.regime_probs.shape


def test_regime_observation_offset_is_used_in_predictions():
    model = SwitchingSSM(n_regimes=1, state_dim=2, obs_dim=3)
    offset = np.array([1.2, -0.7, 0.4])
    model.set_regime_params(0, C=np.zeros((3, 2)), b=offset)
    observations = np.tile(offset, (8, 1))
    result = model.filter(observations)
    np.testing.assert_allclose(result.predicted_observations, np.tile(offset, (len(observations), 1)), atol=1e-10)


def test_fit_can_update_regime_observation_offsets():
    obs = make_synthetic_obs(T=80, obs_dim=3, n_regimes=2)
    model = SwitchingSSM(n_regimes=2, state_dim=2, obs_dim=3)
    before = [regime.b.copy() for regime in model.regimes]
    model.fit(obs, n_iter=2, verbose=False)
    assert any(not np.allclose(regime.b, old)
               for regime, old in zip(model.regimes, before))

if __name__ == "__main__":
    test_switching_ssm_output_shapes()
    test_regime_probs_sum_to_one()
    test_viterbi_path_valid_regimes()
    test_fit_keeps_transition_matrix_valid_and_history_finite()
    test_filter_rejects_invalid_observations()
    test_filtered_covariances_are_symmetric_and_positive_semidefinite()
    test_reported_regime_path_matches_posterior_map()
    test_fit_updates_state_space_parameters_and_keeps_covariances_valid()
    test_known_single_regime_dynamics_synthetic_experiment()
    test_regime_observation_offset_is_used_in_predictions()
    test_fit_can_update_regime_observation_offsets()
    print("All Switching SSM tests passed!")
