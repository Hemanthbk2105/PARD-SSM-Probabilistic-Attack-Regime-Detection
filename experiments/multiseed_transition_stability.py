"""
Multi-seed transition stability benchmark for PARD-SSM.

Evaluates regime clustering and transition-boundary detection across fixed
synthetic seeds. Boundary matching uses a configurable tolerance and greedy
one-to-one matching. This is a controlled synthetic diagnostic, not real-world
intrusion-detection evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.variational_switching import SwitchingSSM
from experiments.multiregime_baseline_comparison import generate_regime_sequence


def boundary_indices(labels: np.ndarray) -> list[int]:
    return (np.flatnonzero(np.diff(labels) != 0) + 1).astype(int).tolist()


def boundary_scores(truth: np.ndarray, predicted: np.ndarray, tolerance: int = 5):
    true_points = boundary_indices(truth)
    predicted_points = boundary_indices(predicted)
    unmatched = set(range(len(true_points)))
    matches = 0
    errors = []
    for point in predicted_points:
        candidates = [
            (abs(true_points[i] - point), i)
            for i in unmatched
            if abs(true_points[i] - point) <= tolerance
        ]
        if candidates:
            distance, chosen = min(candidates)
            unmatched.remove(chosen)
            matches += 1
            errors.append(distance)

    precision = matches / len(predicted_points) if predicted_points else 0.0
    recall = matches / len(true_points) if true_points else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )
    return {
        "true_boundaries": len(true_points),
        "predicted_boundaries": len(predicted_points),
        "matched_boundaries": matches,
        "false_positive_boundaries": len(predicted_points) - matches,
        "missed_boundaries": len(true_points) - matches,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_absolute_boundary_error_steps": (
            float(np.mean(errors)) if errors else None
        ),
        "matching_tolerance_steps": tolerance,
    }


def run_seed(seed: int):
    X, y = generate_regime_sequence(length_per_regime=300, seed=seed)
    model = SwitchingSSM(n_regimes=3, state_dim=2, obs_dim=X.shape[1])

    # Unsupervised initialization from observations; explicitly transductive.
    initializer = KMeans(n_clusters=3, random_state=seed, n_init=10)
    initial_clusters = initializer.fit_predict(X)
    model.transition_matrix = np.full((3, 3), 0.025)
    np.fill_diagonal(model.transition_matrix, 0.95)
    model.initial_regime_probs = np.full(3, 1 / 3)
    for k in range(3):
        model.set_regime_params(k, b=initializer.cluster_centers_[k])

    before = model.filter(X)
    model.fit(X, n_iter=5, verbose=False)
    after = model.filter(X)
    predicted_before = np.argmax(before.regime_probs, axis=1)
    predicted_after = np.argmax(after.regime_probs, axis=1)

    return {
        "seed": seed,
        "sequence_length": int(len(X)),
        "ari": {
            "kmeans_initializer": float(adjusted_rand_score(y, initial_clusters)),
            "pard_before_training": float(adjusted_rand_score(y, predicted_before)),
            "pard_after_training": float(adjusted_rand_score(y, predicted_after)),
        },
        "nmi": {
            "kmeans_initializer": float(normalized_mutual_info_score(y, initial_clusters)),
            "pard_before_training": float(normalized_mutual_info_score(y, predicted_before)),
            "pard_after_training": float(normalized_mutual_info_score(y, predicted_after)),
        },
        "boundary_detection": {
            "pard_before_training": boundary_scores(y, predicted_before),
            "pard_after_training": boundary_scores(y, predicted_after),
        },
        "transition_matrix": model.transition_matrix.tolist(),
    }


def mean_std(values):
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "std": float(np.std(array, ddof=1))}


def main():
    seeds = [2027, 2028, 2029, 2030, 2031]
    results = [run_seed(seed) for seed in seeds]
    summary = {}
    for metric in ("ari", "nmi"):
        summary[metric] = {}
        for method in ("kmeans_initializer", "pard_before_training", "pard_after_training"):
            summary[metric][method] = mean_std([row[metric][method] for row in results])

    summary["boundary_f1"] = {
        method: mean_std([
            row["boundary_detection"][method]["f1"] for row in results
        ])
        for method in ("pard_before_training", "pard_after_training")
    }
    summary["false_positive_boundaries"] = {
        method: mean_std([
            row["boundary_detection"][method]["false_positive_boundaries"]
            for row in results
        ])
        for method in ("pard_before_training", "pard_after_training")
    }
    summary["missed_boundaries"] = {
        method: mean_std([
            row["boundary_detection"][method]["missed_boundaries"]
            for row in results
        ])
        for method in ("pard_before_training", "pard_after_training")
    }

    report = {
        "experiment": "multi_seed_regime_transition_stability",
        "seeds": seeds,
        "summary_mean_and_sample_std": summary,
        "per_seed_results": results,
        "interpretation_notes": [
            "Boundary events are matched one-to-one within five time steps.",
            "ARI and NMI are permutation-invariant clustering metrics.",
            "KMeans initialization uses the full sequence and is transductive.",
            "The same synthetic sequence is used for fitting and evaluation.",
            "Five seeds provide a small stability check, not a definitive statistical study.",
            "No real CICIDS2017 or UNSW-NB15 performance claim follows from this experiment.",
        ],
    }
    output = Path("multiseed_transition_stability.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {output.resolve()}")


if __name__ == "__main__":
    main()
