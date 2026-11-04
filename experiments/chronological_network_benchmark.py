"""Leakage-safe chronological benchmark for prepared network-flow arrays.

When timestamps are supplied, the train/test boundary is selected between
unique timestamps so records sharing a timestamp cannot cross the boundary.
RobustScaler and PCA are fitted on training rows only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.variational_switching import SwitchingSSM
from experiments.heldout_markov_regime_benchmark import boundary_metrics


def _format_timestamp(value, dtype):
    if dtype.kind == "M":
        return np.datetime_as_string(value, unit="auto")
    return str(value)


def _causal_hmm_predict(hmm, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Predict test regimes with a forward-only HMM filter carried from train.

    The HMM is fitted on train only. Its filtered posterior is propagated through
    the training sequence, then carried into test one observation at a time.
    No future test observations are used to decode an earlier test observation.
    """
    log_transition = np.log(np.maximum(hmm.transmat_, 1e-300))
    log_start = np.log(np.maximum(hmm.startprob_, 1e-300))

    train_log_likelihood = hmm._compute_log_likelihood(train)
    test_log_likelihood = hmm._compute_log_likelihood(test)

    log_alpha = log_start + train_log_likelihood[0]
    log_alpha -= logsumexp(log_alpha)

    for emission in train_log_likelihood[1:]:
        log_alpha = logsumexp(log_alpha[:, None] + log_transition, axis=0) + emission
        log_alpha -= logsumexp(log_alpha)

    predictions = np.empty(len(test), dtype=int)
    for index, emission in enumerate(test_log_likelihood):
        log_alpha = logsumexp(log_alpha[:, None] + log_transition, axis=0) + emission
        log_alpha -= logsumexp(log_alpha)
        predictions[index] = int(np.argmax(log_alpha))

    return predictions


def run(
    data_path,
    labels_path,
    timestamps_path=None,
    test_ratio=0.2,
    n_components=10,
    state_dim=8,
    em_iters=10,
    seed=42,
):
    X, y = np.load(data_path), np.load(labels_path)

    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("Features must be 2D and labels 1D with matching row counts.")
    if len(X) < 20 or not np.isfinite(X).all():
        raise ValueError("Need at least 20 rows and finite feature values.")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    split_details = {
        "method": "row_index",
        "warning": (
            "No timestamps supplied; this is a row-based split and may divide "
            "records with equal timestamps across train and test."
        ),
    }

    if timestamps_path is not None:
        timestamps = np.load(timestamps_path)
        if timestamps.ndim != 1 or len(timestamps) != len(X):
            raise ValueError("Timestamps must be 1D and have the same row count as features.")

        if timestamps.dtype.kind == "M":
            if np.isnat(timestamps).any():
                raise ValueError("Timestamps contain NaT values.")
        elif timestamps.dtype.kind in "iuf":
            if not np.isfinite(timestamps).all():
                raise ValueError("Timestamps contain non-finite values.")
        else:
            raise ValueError("Timestamps must use a datetime or numeric dtype.")

        if np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError("Timestamps must be in non-decreasing chronological order.")

        unique_timestamps = np.unique(timestamps)
        train_timestamp_count = int(np.floor(len(unique_timestamps) * (1 - test_ratio)))
        if train_timestamp_count < 1 or train_timestamp_count >= len(unique_timestamps):
            raise ValueError("Not enough unique timestamps for the requested split.")

        boundary_timestamp = unique_timestamps[train_timestamp_count]
        split = int(np.searchsorted(timestamps, boundary_timestamp, side="left"))
        if split < 10 or len(X) - split < 10:
            raise ValueError(
                "Timestamp-aware split produced fewer than 10 rows in one segment. "
                "Try a different test_ratio."
            )

        split_details = {
            "method": "unique_timestamp_boundary",
            "unique_timestamps": int(len(unique_timestamps)),
            "train_unique_timestamps": int(train_timestamp_count),
            "test_unique_timestamps": int(len(unique_timestamps) - train_timestamp_count),
            "first_timestamp": _format_timestamp(timestamps[0], timestamps.dtype),
            "last_timestamp": _format_timestamp(timestamps[-1], timestamps.dtype),
            "first_test_timestamp": _format_timestamp(boundary_timestamp, timestamps.dtype),
            "last_train_timestamp": _format_timestamp(timestamps[split - 1], timestamps.dtype),
            "train_timestamp_range": [
                _format_timestamp(timestamps[0], timestamps.dtype),
                _format_timestamp(timestamps[split - 1], timestamps.dtype),
            ],
            "test_timestamp_range": [
                _format_timestamp(timestamps[split], timestamps.dtype),
                _format_timestamp(timestamps[-1], timestamps.dtype),
            ],
            "same_timestamp_crosses_boundary": bool(
                timestamps[split - 1] == timestamps[split]
            ),
        }
        if split_details["same_timestamp_crosses_boundary"]:
            raise RuntimeError("Internal error: a timestamp was split across train and test.")

    else:
        split = int(len(X) * (1 - test_ratio))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_regimes = np.unique(y_train)
    test_regimes = np.unique(y_test)
    if len(train_regimes) < 2:
        raise ValueError(
            "Training segment contains fewer than two regimes. "
            "Try a larger training period or review the chronological label distribution."
        )

    scaler = RobustScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)

    components = min(n_components, train_scaled.shape[1], len(train_scaled) - 1)
    pca = PCA(n_components=components, random_state=seed).fit(train_scaled)
    train = pca.transform(train_scaled)
    test = pca.transform(test_scaled)

    regimes = len(train_regimes)
    model = SwitchingSSM(
        n_regimes=regimes,
        state_dim=min(state_dim, components),
        obs_dim=components,
    )
    model.fit(train, n_iter=em_iters, verbose=False)

    # Re-filter the training segment with the fitted parameters, then carry its
    # final posterior/state into test. This preserves chronological continuity
    # without fitting or tuning on test observations.
    train_filter = model.filter(train)
    test_filter = model.filter(test, initial_state=train_filter.final_state)
    pard = np.argmax(test_filter.regime_probs, axis=1)
    del train_filter

    hmm = GaussianHMM(
        n_components=regimes,
        covariance_type="diag",
        n_iter=150,
        random_state=seed,
        min_covar=1e-3,
    )
    hmm.fit(train)
    hmm_pred = _causal_hmm_predict(hmm, train, test)

    def metrics(pred):
        return {
            "ari": float(adjusted_rand_score(y_test, pred)),
            "nmi": float(normalized_mutual_info_score(y_test, pred)),
            "boundary": boundary_metrics(y_test, pred),
        }

    limitations = [
        "PARD-SSM and Gaussian HMM regime IDs are unsupervised; metrics compare clusters with ground-truth labels.",
        "PARD-SSM and Gaussian HMM are evaluated with causal forward filtering and state carryover from training into test.",
        "RobustScaler and PCA are fitted on training rows only.",
        "Results on CICIDS2017 do not by themselves establish generalization to live network traffic.",
    ]
    if len(np.setdiff1d(test_regimes, train_regimes)):
        limitations.append(
            "Some test regimes do not occur in training: "
            + str(list(map(int, np.setdiff1d(test_regimes, train_regimes))))
        )
    if timestamps_path is None:
        limitations.append(split_details["warning"])

    return {
        "rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "actual_test_ratio": float(len(X_test) / len(X)),
        "split": split_details,
        "preprocessing": (
            "RobustScaler and PCA fitted on training rows only; reused unchanged on test rows"
        ),
        "evaluation_protocol": (
            "Causal filtering: PARD-SSM and Gaussian HMM carry filtered state/posterior "
            "from the end of training into test; test observations are processed in order "
            "without using future test observations."
        ),
        "train_regimes": sorted(map(int, train_regimes)),
        "test_regimes": sorted(map(int, test_regimes)),
        "train_label_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(y_train, return_counts=True))
        },
        "test_label_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(y_test, return_counts=True))
        },
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "metrics": {
            "pard_ssm": metrics(pard),
            "gaussian_hmm": metrics(hmm_pred),
        },
        "limitations": limitations,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--timestamps", default=None)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--em-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="chronological_network_benchmark.json")
    args = parser.parse_args()

    report = run(
        args.data,
        args.labels,
        args.timestamps,
        args.test_ratio,
        args.components,
        args.state_dim,
        args.em_iters,
        args.seed,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
