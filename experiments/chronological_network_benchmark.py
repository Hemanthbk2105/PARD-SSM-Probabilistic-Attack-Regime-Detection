"""Leakage-safe chronological benchmark for prepared network-flow arrays.

Input arrays must already be ordered by capture time. RobustScaler and PCA
are fitted on training rows only and reused unchanged on the test rows.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.inference.variational_switching import SwitchingSSM
from experiments.heldout_markov_regime_benchmark import boundary_metrics

def run(data_path, labels_path, test_ratio=0.2, n_components=10, state_dim=8, em_iters=10, seed=42):
    X, y = np.load(data_path), np.load(labels_path)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("Features must be 2D and labels 1D with matching row counts.")
    if len(X) < 20 or not np.isfinite(X).all():
        raise ValueError("Need at least 20 rows and finite feature values.")
    if not 0 < test_ratio < 1: raise ValueError("test_ratio must be between 0 and 1.")
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("Both chronological segments need at least two regimes.")
    scaler = RobustScaler().fit(X_train)
    train_scaled, test_scaled = scaler.transform(X_train), scaler.transform(X_test)
    components = min(n_components, train_scaled.shape[1], len(train_scaled) - 1)
    pca = PCA(n_components=components, random_state=seed).fit(train_scaled)
    train, test = pca.transform(train_scaled), pca.transform(test_scaled)
    regimes = len(np.unique(y_train))
    model = SwitchingSSM(n_regimes=regimes, state_dim=min(state_dim, components), obs_dim=components)
    model.fit(train, n_iter=em_iters, verbose=False)
    pard = np.argmax(model.filter(test).regime_probs, axis=1)
    hmm = GaussianHMM(n_components=regimes, covariance_type="diag", n_iter=150, random_state=seed, min_covar=1e-3)
    hmm.fit(train)
    hmm_pred = hmm.predict(test)
    def metrics(pred):
        return {"ari": float(adjusted_rand_score(y_test, pred)),
                "nmi": float(normalized_mutual_info_score(y_test, pred)),
                "boundary": boundary_metrics(y_test, pred)}
    return {"rows": int(len(X)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "preprocessing": "RobustScaler and PCA fitted on training rows only; reused on test rows",
            "train_regimes": sorted(map(int, np.unique(y_train))),
            "test_regimes": sorted(map(int, np.unique(y_test))),
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            "metrics": {"pard_ssm": metrics(pard), "gaussian_hmm": metrics(hmm_pred)},
            "limitations": ["Input row order must be verified against capture timestamps.",
                            "This benchmark does not claim synthetic results transfer to real traffic."]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True); parser.add_argument("--labels", required=True)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--em-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="chronological_network_benchmark.json")
    args = parser.parse_args()
    report = run(args.data, args.labels, args.test_ratio, args.components, args.state_dim, args.em_iters, args.seed)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
