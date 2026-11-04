"""
Reproducible single-regime parameter recovery experiment.

This experiment compares:
1. SwitchingSSM's approximate generalized-EM updates.
2. Its initial parameter values (no-learning baseline).
3. An oracle least-squares estimate using the generated hidden states.

The oracle is an upper-bound diagnostic, not a deployable baseline, because
real network telemetry does not expose the hidden state sequence.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.variational_switching import SwitchingSSM


def generate_sequence(length: int = 3000, seed: int = 2026):
    rng = np.random.default_rng(seed)
    true_a = np.array([[0.78, 0.08], [-0.04, 0.72]], dtype=float)
    true_c = np.array([[1.0, 0.25], [-0.2, 0.9], [0.35, -0.4]], dtype=float)
    true_q = np.diag([0.025, 0.035])
    true_r = np.diag([0.04, 0.05, 0.03])
    states = np.zeros((length, 2), dtype=float)
    observations = np.zeros((length, 3), dtype=float)
    states[0] = rng.multivariate_normal(np.zeros(2), np.eye(2))
    for t in range(1, length):
        states[t] = true_a @ states[t - 1] + rng.multivariate_normal(np.zeros(2), true_q)
    for t in range(length):
        observations[t] = true_c @ states[t] + rng.multivariate_normal(np.zeros(3), true_r)
    return observations, states, {"A": true_a, "C": true_c, "Q": true_q, "R": true_r}


def relative_frobenius_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth, ord="fro") /
                 max(np.linalg.norm(truth, ord="fro"), 1e-12))


def oracle_least_squares(states: np.ndarray, observations: np.ndarray):
    """Estimate parameters with access to true latent states (oracle diagnostic)."""
    x_prev, x_next = states[:-1], states[1:]
    a_hat = np.linalg.lstsq(x_prev, x_next, rcond=None)[0].T
    process_residual = x_next - x_prev @ a_hat.T
    q_hat = process_residual.T @ process_residual / len(process_residual)

    c_hat = np.linalg.lstsq(states, observations, rcond=None)[0].T
    observation_residual = observations - states @ c_hat.T
    r_hat = observation_residual.T @ observation_residual / len(observation_residual)
    return {"A": a_hat, "C": c_hat, "Q": q_hat, "R": r_hat}


def main():
    observations, states, truth = generate_sequence()
    model = SwitchingSSM(n_regimes=1, state_dim=2, obs_dim=3)
    initial = {name: getattr(model.regimes[0], name).copy()
               for name in ("A", "C", "Q", "R")}

    model.fit(observations, n_iter=12, verbose=False)
    learned = {name: getattr(model.regimes[0], name).copy()
               for name in ("A", "C", "Q", "R")}
    oracle = oracle_least_squares(states, observations)

    report = {
        "experiment": "single_regime_linear_gaussian_parameter_recovery",
        "seed": 2026,
        "sequence_length": int(len(observations)),
        "state_dim": int(states.shape[1]),
        "observation_dim": int(observations.shape[1]),
        "training_iterations": 12,
        "relative_frobenius_error": {
            name: {
                "initial_no_learning": relative_frobenius_error(initial[name], truth[name]),
                "approximate_generalized_em": relative_frobenius_error(learned[name], truth[name]),
                "oracle_hidden_state_least_squares": relative_frobenius_error(oracle[name], truth[name]),
            }
            for name in ("A", "C", "Q", "R")
        },
        "limitations": [
            "The oracle baseline uses true hidden states and is not available on real telemetry.",
            "This is a single-regime controlled experiment, not evidence of multi-regime recovery.",
            "The current trainer uses filtered state means and marginal responsibilities, not exact smoothed sufficient statistics.",
            "Parameter recovery can be affected by latent-state scale and rotation non-identifiability.",
        ],
    }
    output = Path("synthetic_parameter_recovery.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {output.resolve()}")


if __name__ == "__main__":
    main()
