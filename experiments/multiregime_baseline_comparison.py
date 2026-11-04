"""
Controlled multi-regime synthetic benchmark.

Compares the SwitchingSSM posterior regimes against a Gaussian Mixture Model
(GMM) baseline using ARI and NMI. Dataset regimes are generated from distinct
linear Gaussian state dynamics. Results are descriptive and do not establish
performance on real network traffic.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.variational_switching import SwitchingSSM
from src.inference.kalman_filter import KalmanFilter


def generate_regime_sequence(length_per_regime: int = 400, seed: int = 2027):
    rng = np.random.default_rng(seed)
    transition_matrices = [
        np.array([[0.88, 0.02], [0.00, 0.84]]),
        np.array([[0.62, -0.12], [0.15, 0.70]]),
        np.array([[0.78, 0.10], [-0.08, 0.58]]),
    ]
    observation_matrix = np.array([
        [1.0, 0.25],
        [-0.2, 0.9],
        [0.35, -0.4],
        [0.6, 0.15],
    ])
    process_covariances = [
        np.diag([0.018, 0.025]),
        np.diag([0.055, 0.035]),
        np.diag([0.025, 0.065]),
    ]
    observation_covariance = np.diag([0.035, 0.04, 0.03, 0.045])
    offsets = [
        np.array([-0.8, -0.4, 0.2, -0.3]),
        np.array([1.0, 0.7, -0.5, 0.8]),
        np.array([-0.2, 1.1, 0.9, -0.7]),
    ]

    observations, labels = [], []
    for regime_id in range(3):
        states = np.zeros((length_per_regime, 2))
        states[0] = rng.normal(scale=0.3, size=2)
        for t in range(1, length_per_regime):
            states[t] = (
                transition_matrices[regime_id] @ states[t - 1]
                + rng.multivariate_normal(np.zeros(2), process_covariances[regime_id])
            )
        noise = rng.multivariate_normal(
            np.zeros(4), observation_covariance, size=length_per_regime
        )
        observations.append(states @ observation_matrix.T + offsets[regime_id] + noise)
        labels.append(np.full(length_per_regime, regime_id, dtype=int))

    return np.vstack(observations), np.concatenate(labels)


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "adjusted_rand_index": float(adjusted_rand_score(y_true, y_pred)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(y_true, y_pred)
        ),
    }


def main():
    X, y = generate_regime_sequence()
    model = SwitchingSSM(n_regimes=3, state_dim=2, obs_dim=X.shape[1])

    # Data-driven observation-offset initialization. This gives each regime
    # a distinct mean before filtering; it is unsupervised but uses the full
    # benchmark sequence, so treat the result as a transductive diagnostic.
    offset_initializer = KMeans(n_clusters=3, random_state=2027, n_init=10)
    offset_labels = offset_initializer.fit_predict(X)
    model.transition_matrix = np.full((3, 3), 0.025)
    np.fill_diagonal(model.transition_matrix, 0.95)
    model.initial_regime_probs = np.full(3, 1.0 / 3.0)
    for k in range(3):
        model.set_regime_params(k, b=offset_initializer.cluster_centers_[k])

    initial_result = model.filter(X)
    model.fit(X, n_iter=5, verbose=False)
    trained_result = model.filter(X)
    pard_initial = np.argmax(initial_result.regime_probs, axis=1)
    pard_trained = np.argmax(trained_result.regime_probs, axis=1)

    gmm = GaussianMixture(
        n_components=3, covariance_type="full", random_state=2027, n_init=5
    )
    gmm_labels = gmm.fit_predict(X)

    hmm = GaussianHMM(
        n_components=3, covariance_type="diag", n_iter=100,
        random_state=2027, min_covar=1e-4,
    )
    hmm.fit(X)
    hmm_labels = hmm.predict(X)

    # A standard Kalman filter has no discrete regime labels. Evaluate it
    # using one-step observation prediction error and average predictive NLL.
    kalman = KalmanFilter.init_from_data(X, state_dim=2, noise_scale=0.1)
    kalman_result = kalman.filter(X)
    kalman_predicted_observations = kalman_result.predicted_means @ kalman.C.T
    kalman_prediction_mse = float(np.mean((X - kalman_predicted_observations) ** 2))
    kalman_average_nll = float(-kalman_result.log_likelihood / len(X))

    report = {
        "experiment": "three_regime_linear_gaussian_sequence",
        "seed": 2027,
        "sequence_length": int(len(X)),
        "regime_count": 3,
        "regime_segment_lengths": [400, 400, 400],
        "metrics": {
            "pard_ssm_before_training": clustering_metrics(y, pard_initial),
            "pard_ssm_after_offset_initialization": clustering_metrics(y, pard_initial),
            "pard_ssm_after_offset_initialization_and_approximate_generalized_em": clustering_metrics(
                y, pard_trained
            ),
            "kmeans_offset_initializer": clustering_metrics(y, offset_labels),
            "gaussian_mixture_baseline": clustering_metrics(y, gmm_labels),
            "gaussian_hidden_markov_model": clustering_metrics(y, hmm_labels),
            "standard_kalman_filter": {
                "one_step_observation_prediction_mse": kalman_prediction_mse,
                "average_negative_log_likelihood": kalman_average_nll,
                "note": "Continuous-state baseline; ARI/NMI are not applicable because it has no discrete regimes.",
            },
        },
        "transition_diagnostics": {
            "learned_transition_matrix": model.transition_matrix.tolist(),
            "true_regime_order": [0, 1, 2],
            "learned_observation_offsets": [regime.b.tolist() for regime in model.regimes],
            "predicted_regime_switches_before_training": int(
                np.count_nonzero(np.diff(pard_initial))
            ),
            "predicted_regime_switches_after_training": int(
                np.count_nonzero(np.diff(pard_trained))
            ),
            "true_regime_switches": int(np.count_nonzero(np.diff(y))),
        },
        "limitations": [
            "Synthetic data use three long contiguous regime segments.",
            "The benchmark does not represent the class imbalance or complexity of CICIDS2017/UNSW-NB15.",
            "ARI and NMI evaluate clustering agreement, not operational intrusion detection.",
            "GMM is a static clustering baseline and does not model temporal transitions.",
            "Gaussian HMM models discrete temporal transitions but not continuous latent state dynamics.",
            "The standard Kalman filter has one continuous state and no discrete regime labels; its predictive metrics are not directly comparable to ARI/NMI.",
            "These baselines are trained and evaluated on the same synthetic sequence; this is a controlled diagnostic, not a held-out generalization test.",
            "The switching model uses approximate GPB1 filtering and approximate generalized-EM updates.",
            "KMeans observation-offset initialization uses the full synthetic sequence and is transductive; it is not an online initialization protocol.",
            "PARD-SSM and KMeans share the same unsupervised KMeans-derived cluster centers for this diagnostic, so do not interpret this as a fair isolated test of the switching model's learned offsets.",
            "Latent state labels are permutation-invariant; ARI and NMI avoid assuming matching state IDs.",
        ],
    }
    output = Path("multiregime_baseline_comparison.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {output.resolve()}")


if __name__ == "__main__":
    main()
