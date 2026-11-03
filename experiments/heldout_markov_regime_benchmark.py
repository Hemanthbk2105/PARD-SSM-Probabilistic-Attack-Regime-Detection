"""
Held-out benchmark with overlapping, Markov-switching synthetic regimes.

Unlike the earlier contiguous-segment diagnostic, regimes switch stochastically
throughout the sequence. Preprocessing and model fitting use only the training
portion; metrics are reported on a held-out chronological test portion.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from hmmlearn.hmm import GaussianHMM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.variational_switching import SwitchingSSM


def generate_markov_sequence(length: int = 2400, seed: int = 2040):
    rng = np.random.default_rng(seed)
    transition = np.array([
        [0.965, 0.025, 0.010],
        [0.020, 0.955, 0.025],
        [0.015, 0.030, 0.955],
    ])
    A = [
        np.array([[0.82, 0.05], [0.00, 0.78]]),
        np.array([[0.70, -0.08], [0.10, 0.75]]),
        np.array([[0.77, 0.07], [-0.06, 0.68]]),
    ]
    C = np.array([
        [1.0, 0.20],
        [-0.15, 0.85],
        [0.30, -0.25],
        [0.55, 0.10],
    ])
    offsets = np.array([
        [-0.30, -0.15, 0.10, -0.12],
        [0.15, 0.25, -0.18, 0.20],
        [0.05, -0.10, 0.22, -0.18],
    ])
    Q = [
        np.diag([0.035, 0.040]),
        np.diag([0.045, 0.030]),
        np.diag([0.030, 0.050]),
    ]
    R = np.diag([0.18, 0.16, 0.20, 0.17])

    regimes = np.zeros(length, dtype=int)
    states = np.zeros((length, 2))
    observations = np.zeros((length, 4))
    regimes[0] = rng.integers(0, 3)
    states[0] = rng.normal(scale=0.25, size=2)

    for t in range(length):
        if t > 0:
            regimes[t] = rng.choice(3, p=transition[regimes[t - 1]])
            k = regimes[t]
            states[t] = A[k] @ states[t - 1] + rng.multivariate_normal(np.zeros(2), Q[k])
        k = regimes[t]
        observations[t] = (
            C @ states[t] + offsets[k]
            + rng.multivariate_normal(np.zeros(4), R)
        )
    return observations, regimes, transition


def boundaries(labels):
    return (np.flatnonzero(np.diff(labels) != 0) + 1).astype(int)


def boundary_metrics(y_true, y_pred, tolerance=5):
    true_points, pred_points = boundaries(y_true), boundaries(y_pred)
    remaining = set(range(len(true_points)))
    matched = 0
    distances = []
    for point in pred_points:
        candidates = [(abs(int(true_points[i]) - int(point)), i)
                      for i in remaining
                      if abs(int(true_points[i]) - int(point)) <= tolerance]
        if candidates:
            distance, index = min(candidates)
            remaining.remove(index)
            matched += 1
            distances.append(distance)
    precision = matched / len(pred_points) if len(pred_points) else 0.0
    recall = matched / len(true_points) if len(true_points) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "actual_boundaries": int(len(true_points)),
        "predicted_boundaries": int(len(pred_points)),
        "matched": int(matched),
        "false_positives": int(len(pred_points) - matched),
        "missed": int(len(true_points) - matched),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_boundary_error_steps": float(np.mean(distances)) if distances else None,
        "tolerance_steps": tolerance,
    }


def score(y_true, y_pred):
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "boundary": boundary_metrics(y_true, y_pred),
    }


def causal_probability_smoothing(probabilities: np.ndarray, window: int) -> np.ndarray:
    """Average only current and past posterior probabilities; no future leakage."""
    smoothed = np.empty_like(probabilities)
    cumulative = np.vstack([np.zeros((1, probabilities.shape[1])),
                            np.cumsum(probabilities, axis=0)])
    for t in range(len(probabilities)):
        start = max(0, t - window + 1)
        smoothed[t] = (cumulative[t + 1] - cumulative[start]) / (t - start + 1)
    return smoothed


def run_seed(seed: int):
    X, y, true_transition = generate_markov_sequence(seed=seed)
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_test = y[split:]

    scaler = StandardScaler().fit(X_train)
    train = scaler.transform(X_train)
    test = scaler.transform(X_test)

    # KMeans is fitted only on training data. Its centers initialize the
    # observation offsets; no test observations are used for initialization.
    initializer = KMeans(n_clusters=3, random_state=seed, n_init=10)
    initializer.fit(train)
    model = SwitchingSSM(n_regimes=3, state_dim=2, obs_dim=train.shape[1])
    model.transition_matrix = np.full((3, 3), 0.02)
    np.fill_diagonal(model.transition_matrix, 0.96)
    model.initial_regime_probs = np.full(3, 1 / 3)
    for k in range(3):
        model.set_regime_params(k, b=initializer.cluster_centers_[k])
    model.fit(train, n_iter=5, verbose=False)
    pard_probabilities = model.filter(test).regime_probs
    pard_prediction = np.argmax(pard_probabilities, axis=1)
    # Fixed causal smoothing windows are diagnostic post-processing choices,
    # not model retraining. They use no future test observations.
    smoothed_predictions = {
        f"pard_ssm_causal_posterior_window_{window}": np.argmax(
            causal_probability_smoothing(pard_probabilities, window), axis=1
        )
        for window in (5, 11, 21)
    }

    gmm = GaussianMixture(n_components=3, covariance_type="full",
                          random_state=seed, n_init=5)
    gmm.fit(train)
    gmm_prediction = gmm.predict(test)

    hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=150,
                      random_state=seed, min_covar=1e-3)
    hmm.fit(train)
    hmm_prediction = hmm.predict(test)

    return {
        "seed": seed,
        "train_steps": int(len(train)),
        "test_steps": int(len(test)),
        "true_test_regime_switches": int(np.count_nonzero(np.diff(y_test))),
        "true_transition_matrix": true_transition.tolist(),
        "metrics": {
            "pard_ssm_raw": score(y_test, pard_prediction),
            **{
                name: score(y_test, prediction)
                for name, prediction in smoothed_predictions.items()
            },
            "gaussian_mixture": score(y_test, gmm_prediction),
            "gaussian_hmm": score(y_test, hmm_prediction),
        },
        "pard_learned_transition_matrix": model.transition_matrix.tolist(),
    }


def aggregate(rows):
    result = {}
    methods = (
        "pard_ssm_raw",
        "pard_ssm_causal_posterior_window_5",
        "pard_ssm_causal_posterior_window_11",
        "pard_ssm_causal_posterior_window_21",
        "gaussian_mixture",
        "gaussian_hmm",
    )
    for method in methods:
        result[method] = {}
        for metric in ("ari", "nmi"):
            values = [row["metrics"][method][metric] for row in rows]
            result[method][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
        for metric in ("precision", "recall", "f1", "false_positives", "missed"):
            values = [row["metrics"][method]["boundary"][metric] for row in rows]
            result[method][f"boundary_{metric}"] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
    return result


def main():
    seeds = [2040, 2041, 2042, 2043, 2044]
    rows = [run_seed(seed) for seed in seeds]
    report = {
        "experiment": "held_out_markov_switching_overlapping_regimes",
        "seeds": seeds,
        "chronological_split": "70% train / 30% test",
        "preprocessing": "StandardScaler fit on training data only",
        "summary_mean_and_sample_std": aggregate(rows),
        "per_seed_results": rows,
        "limitations": [
            "Synthetic benchmark only; not evidence of real-world intrusion detection.",
            "The regimes are overlapping and switch according to a known Markov chain.",
            "KMeans initialisation is fitted only on training observations.",
            "The model is reset at the beginning of the test segment with its learned initial regime distribution.",
            "ARI/NMI are permutation-invariant but do not measure operational alert quality.",
            "Five seeds are an initial stability check, not a definitive statistical study.",
            "PARD-SSM uses approximate GPB1 filtering and approximate generalized-EM updates.",
            "Causal posterior smoothing windows are fixed diagnostics and use no future test data.",
            "Smoothing results are post-processing comparisons, not a learned model improvement.",
        ],
    }
    output = Path("heldout_markov_regime_benchmark.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {output.resolve()}")


if __name__ == "__main__":
    main()
