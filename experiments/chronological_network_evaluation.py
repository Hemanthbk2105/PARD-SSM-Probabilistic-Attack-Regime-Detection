"""
Leakage-safe chronological evaluation on CICIDS2017 or UNSW-NB15.

This script loads the complete dataset in capture-time order, performs a
chronological holdout, fits preprocessing on training rows only, and compares
PARD-SSM with a Gaussian HMM. It requires the real dataset files locally.

Example:
    python experiments/chronological_network_evaluation.py \
        --dataset cicids2017 --data-dir data/raw/cicids2017 \
        --output experiments/results/cicids2017_chronological.json

    python experiments/chronological_network_evaluation.py \
        --dataset unswnb15 --data-dir data/raw/unswnb15 \
        --output experiments/results/unswnb15_chronological.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing.dataset_loader import load_cicids2017, load_unswnb15
from src.inference.variational_switching import SwitchingSSM


def find_boundaries(labels: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.diff(labels) != 0) + 1


def boundary_metrics(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 5) -> dict:
    true_points = find_boundaries(y_true)
    pred_points = find_boundaries(y_pred)
    unmatched = set(range(len(true_points)))
    matched = 0
    errors = []
    for point in pred_points:
        candidates = [
            (abs(int(true_points[i]) - int(point)), i)
            for i in unmatched
            if abs(int(true_points[i]) - int(point)) <= tolerance
        ]
        if candidates:
            distance, index = min(candidates)
            unmatched.remove(index)
            matched += 1
            errors.append(distance)
    precision = matched / len(pred_points) if len(pred_points) else 0.0
    recall = matched / len(true_points) if len(true_points) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_boundaries": int(len(true_points)),
        "predicted_boundaries": int(len(pred_points)),
        "matched_boundaries": int(matched),
        "false_positives": int(len(pred_points) - matched),
        "missed_boundaries": int(len(true_points) - matched),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_error_steps": float(np.mean(errors)) if errors else None,
        "tolerance_steps": tolerance,
    }


def evaluate(dataset: str, data_dir: str, output: str, test_ratio: float = 0.2,
             components: int = 8, state_dim: int = 4, em_iters: int = 5,
             sample_size: int = 0) -> dict:
    if not 0.05 <= test_ratio <= 0.5:
        raise ValueError("test_ratio must be between 0.05 and 0.5.")
    if components < 1 or state_dim < 1 or em_iters < 1:
        raise ValueError("components, state_dim and em_iters must be positive.")

    if dataset == "cicids2017":
        X, y = load_cicids2017(data_dir, sample_size=sample_size)
    elif dataset == "unswnb15":
        X, y = load_unswnb15(data_dir, sample_size=sample_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if len(X) != len(y) or len(X) < 100:
        raise ValueError("Dataset must contain matching features/labels and at least 100 rows.")
    if not np.isfinite(X).all():
        raise ValueError("Features contain non-finite values after loading.")
    if len(np.unique(y)) < 2:
        raise ValueError("At least two label regimes are required.")

    split = int(len(X) * (1.0 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    n_regimes = int(max(np.unique(y))) + 1

    # Fit all transforms on training rows only. Never fit on held-out rows.
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_components = min(components, X_train_scaled.shape[1], X_train_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    pard = SwitchingSSM(n_regimes=n_regimes, state_dim=state_dim, obs_dim=n_components)
    pard.fit(X_train_pca, n_iter=em_iters, verbose=False)
    pard_pred = pard.filter(X_test_pca).viterbi_path

    hmm = GaussianHMM(
        n_components=n_regimes, covariance_type="diag",
        n_iter=150, random_state=42, min_covar=1e-3,
    )
    hmm.fit(X_train_pca)
    hmm_pred = hmm.predict(X_test_pca)

    def score(prediction: np.ndarray) -> dict:
        return {
            "ari": float(adjusted_rand_score(y_test, prediction)),
            "nmi": float(normalized_mutual_info_score(y_test, prediction)),
            "boundaries": boundary_metrics(y_test, prediction),
        }

    report = {
        "dataset": dataset,
        "chronological_split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_ratio": float(test_ratio),
            "sampling": "disabled" if sample_size <= 0 else "deterministic first-N rows",
        },
        "preprocessing": {
            "scaler": "RobustScaler fitted on training rows only",
            "pca_components": int(n_components),
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "pca_fitted_on": "training rows only",
        },
        "label_distribution": {
            "train": {str(int(k)): int((y_train == k).sum()) for k in np.unique(y_train)},
            "test": {str(int(k)): int((y_test == k).sum()) for k in np.unique(y_test)},
        },
        "metrics": {
            "pard_ssm": score(pard_pred),
            "gaussian_hmm": score(hmm_pred),
        },
        "limitations": [
            "Requires locally downloaded real dataset files; no results are fabricated.",
            "Dataset label IDs are used as ground truth; model cluster IDs are permutation-invariant metrics.",
            "The loaders sort by available capture timestamps, but capture-session continuity must be verified for the downloaded release.",
            "This is an offline retrospective benchmark, not a live intrusion detection deployment.",
            "PARD-SSM uses approximate GPB1 filtering and approximate generalized-EM.",
        ],
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leakage-safe chronological network dataset evaluation")
    parser.add_argument("--dataset", choices=["cicids2017", "unswnb15"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="experiments/results/chronological_evaluation.json")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--em-iters", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=0,
                        help="0 disables sampling and keeps all rows (recommended for temporal evaluation)")
    args = parser.parse_args()
    evaluate(
        dataset=args.dataset, data_dir=args.data_dir, output=args.output,
        test_ratio=args.test_ratio, components=args.components,
        state_dim=args.state_dim, em_iters=args.em_iters,
        sample_size=args.sample_size,
    )
