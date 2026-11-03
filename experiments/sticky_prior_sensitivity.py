"""Sensitivity analysis for the PARD-SSM sticky transition prior.

Runs identical held-out Markov sequences across prior strengths. Baselines are
included once per seed; metrics are aggregated over the same five seeds.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from heldout_markov_regime_benchmark import run_seed

SEEDS = [2040, 2041, 2042, 2043, 2044]
STRENGTHS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def main():
    report = {
        "experiment": "sticky_transition_prior_sensitivity",
        "seeds": SEEDS,
        "strengths": STRENGTHS,
        "results": {},
        "limitations": [
            "Synthetic data only; not evidence of real-world intrusion detection.",
            "Five seeds provide an initial sensitivity check, not a definitive statistical study.",
            "The sticky prior is a diagonal Dirichlet pseudocount per EM iteration.",
            "PARD-SSM inference and parameter updates remain approximate.",
        ],
    }
    for strength in STRENGTHS:
        rows = [run_seed(seed, sticky_prior_strength=strength) for seed in SEEDS]
        methods = [
            "pard_ssm_raw",
            "pard_ssm_causal_posterior_window_5",
            "pard_ssm_causal_posterior_window_11",
            "pard_ssm_causal_posterior_window_21",
        ]
        summary = {}
        for method in methods:
            summary[method] = {}
            for metric in ("ari", "nmi"):
                values = [r["metrics"][method][metric] for r in rows]
                summary[method][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                }
            for metric in ("f1", "false_positives", "missed", "precision", "recall"):
                values = [r["metrics"][method]["boundary"][metric] for r in rows]
                summary[method][f"boundary_{metric}"] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                }
        report["results"][str(strength)] = summary
        print(f"Completed sticky prior strength={strength}", flush=True)

    output = Path("sticky_prior_sensitivity.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {output.resolve()}")


if __name__ == "__main__":
    main()
