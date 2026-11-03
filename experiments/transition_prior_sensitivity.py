"""Sensitivity analysis for a prior anchored to the initial transition matrix.

This experiment tests whether a prior expressing persistence at fit start
prevents the transition matrix from becoming unrealistically diffuse.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from heldout_markov_regime_benchmark import run_seed

SEEDS = [2040, 2041, 2042, 2043, 2044]
STRENGTHS = [0.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]


def main():
    report = {
        "experiment": "initial_transition_matrix_prior_sensitivity",
        "seeds": SEEDS,
        "strengths": STRENGTHS,
        "results": {},
        "limitations": [
            "Synthetic data only; not evidence of real-world intrusion detection.",
            "Five seeds provide an initial sensitivity check, not a definitive statistical study.",
            "The prior is anchored to the transition matrix supplied before fitting.",
            "The prior strength is the total pseudocount added per transition row per EM iteration.",
            "PARD-SSM still uses approximate GPB1 filtering and approximate generalized-EM updates.",
        ],
    }
    for strength in STRENGTHS:
        rows = [
            run_seed(seed, sticky_prior_strength=0.0,
                     transition_prior_strength=strength)
            for seed in SEEDS
        ]
        methods = [
            "pard_ssm_raw",
            "pard_ssm_causal_posterior_window_5",
            "pard_ssm_causal_posterior_window_11",
            "pard_ssm_causal_posterior_window_21",
            "gaussian_hmm",
        ]
        summary = {}
        for method in methods:
            summary[method] = {}
            for metric in ("ari", "nmi"):
                values = [row["metrics"][method][metric] for row in rows]
                summary[method][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                }
            for metric in ("f1", "precision", "recall", "false_positives", "missed"):
                values = [row["metrics"][method]["boundary"][metric] for row in rows]
                summary[method][f"boundary_{metric}"] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                }
        diagonal = [
            np.mean(np.diag(row["pard_learned_transition_matrix"]))
            for row in rows
        ]
        summary["pard_learned_transition_diagonal_mean"] = {
            "mean": float(np.mean(diagonal)),
            "std": float(np.std(diagonal, ddof=1)),
        }
        report["results"][str(strength)] = summary
        print(f"Completed transition prior strength={strength}", flush=True)

    output = Path("transition_prior_sensitivity.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {output.resolve()}")


if __name__ == "__main__":
    main()
