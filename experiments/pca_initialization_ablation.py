"""Compare default and train-only PCA/cluster initialization on held-out Markov data.

This is an initialization ablation, not a change to the model's default behavior.
All scaling, PCA, clustering, and parameter initialization use training data only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.variational_switching import SwitchingSSM
from heldout_markov_regime_benchmark import (
    generate_markov_sequence, score, causal_probability_smoothing,
    run_seed, aggregate,
)

SEEDS = [2040, 2041, 2042, 2043, 2044]


def _covariance(residuals: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    if len(residuals) < 2:
        return np.eye(residuals.shape[1]) * floor
    cov = np.cov(residuals, rowvar=False)
    cov = np.atleast_2d(cov)
    values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
    return (vectors * np.maximum(values, floor)) @ vectors.T


def run_pca_initialized_seed(seed: int) -> dict:
    X, y, true_transition = generate_markov_sequence(seed=seed)
    split = int(0.7 * len(X))
    scaler = StandardScaler().fit(X[:split])
    train, test = scaler.transform(X[:split]), scaler.transform(X[split:])
    y_test = y[split:]

    # Fit dimensionality reduction and regime clusters on training only.
    pca = PCA(n_components=2, random_state=seed).fit(train)
    z_train = pca.transform(train)
    clusterer = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(z_train)
    labels = clusterer.labels_

    model = SwitchingSSM(n_regimes=3, state_dim=2, obs_dim=train.shape[1])
    C = pca.components_.T.copy()
    residual = train - z_train @ C.T
    counts = np.ones((3, 3), dtype=float)
    for t in range(1, len(labels)):
        counts[labels[t - 1], labels[t]] += 1.0
    model.transition_matrix = counts / counts.sum(axis=1, keepdims=True)
    frequencies = np.bincount(labels, minlength=3).astype(float) + 1.0
    model.initial_regime_probs = frequencies / frequencies.sum()

    for k in range(3):
        mask = labels == k
        regime_z = z_train[mask]
        model.mu0[k] = regime_z.mean(axis=0)
        model.P0[k] = _covariance(regime_z - model.mu0[k])
        model.set_regime_params(k, C=C, b=residual[mask].mean(axis=0))
        if np.count_nonzero(mask[1:] & mask[:-1]) >= 5:
            pair_mask = (labels[1:] == k) & (labels[:-1] == k)
            x_prev, x_now = z_train[:-1][pair_mask], z_train[1:][pair_mask]
            if len(x_prev) >= 5:
                lhs = x_prev.T @ x_prev + 1e-2 * np.eye(2)
                A = np.linalg.solve(lhs, x_prev.T @ x_now).T
                radius = float(np.max(np.abs(np.linalg.eigvals(A))))
                if radius > 0.995:
                    A *= 0.995 / radius
                q = _covariance(x_now - x_prev @ A.T)
                model.set_regime_params(k, A=A, Q=q)
        model.set_regime_params(k, R=_covariance(residual[mask]))

    model.fit(train, n_iter=5, verbose=False,
              sticky_prior_strength=0.0, transition_prior_strength=0.0)
    probabilities = model.filter(test).regime_probs
    predictions = {
        "pard_pca_initialized_raw": np.argmax(probabilities, axis=1),
        "pard_pca_initialized_window_11": np.argmax(
            causal_probability_smoothing(probabilities, 11), axis=1
        ),
    }
    return {
        "seed": seed,
        "train_steps": len(train),
        "test_steps": len(test),
        "metrics": {name: score(y_test, pred) for name, pred in predictions.items()},
        "learned_transition_matrix": model.transition_matrix.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def _summary(rows: list[dict], method: str) -> dict:
    result = {}
    for metric in ("ari", "nmi"):
        values = [row["metrics"][method][metric] for row in rows]
        result[metric] = {"mean": float(np.mean(values)),
                          "std": float(np.std(values, ddof=1))}
    for metric in ("precision", "recall", "f1", "false_positives", "missed"):
        values = [row["metrics"][method]["boundary"][metric] for row in rows]
        result[f"boundary_{metric}"] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
    return result


def main() -> None:
    initialized = [run_pca_initialized_seed(seed) for seed in SEEDS]
    baseline = [run_seed(seed, sticky_prior_strength=0.0,
                         transition_prior_strength=0.0) for seed in SEEDS]
    report = {
        "experiment": "train_only_pca_regime_initialization_ablation",
        "seeds": SEEDS,
        "split": "chronological 70% train / 30% test",
        "initialization": (
            "StandardScaler and 2D PCA fit on training only; KMeans on training "
            "PCA scores; regime-specific A/Q/R/b and initial state moments "
            "estimated from training clusters."
        ),
        "summary_mean_and_sample_std": {
            "existing_default_window_11": _summary(
                baseline, "pard_ssm_causal_posterior_window_11"),
            "pca_initialized_raw": _summary(
                initialized, "pard_pca_initialized_raw"),
            "pca_initialized_window_11": _summary(
                initialized, "pard_pca_initialized_window_11"),
            "gaussian_hmm": _summary(baseline, "gaussian_hmm"),
        },
        "per_seed_results": initialized,
        "limitations": [
            "Synthetic benchmark only; not evidence of real-world intrusion detection.",
            "Five seeds are an initial ablation, not a definitive statistical study.",
            "PCA and KMeans initialization use training data only.",
            "The model still uses approximate GPB1 filtering and approximate generalized EM.",
            "A better initialization on this generator does not establish superiority on network traffic.",
        ],
    }
    output = Path("pca_initialization_ablation.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {output.resolve()}")


if __name__ == "__main__":
    main()
