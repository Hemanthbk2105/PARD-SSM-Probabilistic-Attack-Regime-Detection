from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA_ROOT = ROOT / "platform" / "data"
RESULT_ROOT = ROOT / "platform" / "results"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

REGIME_NAMES = {
    0: "Normal",
    1: "Reconnaissance",
    2: "Intrusion",
    3: "DoS / DDoS",
    4: "Exfiltration",
}
REGIME_COLORS = ["#47d7ac", "#f5bd55", "#ff8a65", "#ff5f72", "#a78bfa"]

app = FastAPI(
    title="PARD-SSM Security API",
    description="Security analytics API for probabilistic attack-regime detection.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def dataset_dir(name: str) -> Path:
    if name not in {"cicids2017", "unswnb15"}:
        raise HTTPException(400, "Choose CICIDS2017 or UNSW-NB15.")
    path = DATA_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_result(name: str) -> Path:
    return RESULT_ROOT / f"{name}_latest.json"


def safe_json(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value)}")


@app.get("/api/health")
def health():
    return {"status": "online", "service": "PARD-SSM Security", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/datasets")
def datasets():
    output = []
    for name, label in [("cicids2017", "CICIDS2017"), ("unswnb15", "UNSW-NB15")]:
        folder = dataset_dir(name)
        files = sorted(p.name for p in folder.glob("*.csv"))
        processed = (folder / "features.npy").exists() and (folder / "labels.npy").exists()
        result_exists = latest_result(name).exists()
        output.append({
            "id": name,
            "name": label,
            "files": files,
            "file_count": len(files),
            "processed": processed,
            "analysed": result_exists,
            "samples": int(np.load(folder / "features.npy", mmap_mode="r").shape[0]) if processed else 0,
        })
    return {"datasets": output}


@app.post("/api/upload/{name}")
async def upload_dataset(name: str, files: list[UploadFile] = File(...)):
    folder = dataset_dir(name)
    if not files:
        raise HTTPException(400, "Select at least one CSV file.")
    saved = []
    for uploaded in files:
        filename = Path(uploaded.filename or "").name
        if not filename.lower().endswith(".csv"):
            raise HTTPException(400, f"{filename or 'File'} is not a CSV.")
        destination = folder / filename
        with destination.open("wb") as target:
            shutil.copyfileobj(uploaded.file, target)
        saved.append(filename)
    return {"message": f"Uploaded {len(saved)} file(s).", "files": saved}


@app.post("/api/prepare/{name}")
def prepare_dataset(name: str, sample_size: int = 50000):
    folder = dataset_dir(name)
    if not list(folder.glob("*.csv")):
        raise HTTPException(400, "Upload the dataset CSV files first.")
    try:
        from src.data_processing.dataset_loader import load_cicids2017, load_unswnb15
        loader = load_cicids2017 if name == "cicids2017" else load_unswnb15
        X, y = loader(str(folder), sample_size=sample_size)
        if X.ndim != 2 or len(X) != len(y) or len(X) < 120:
            raise ValueError("The prepared dataset must contain at least 120 matching rows.")
        np.save(folder / "features.npy", X.astype(np.float32))
        np.save(folder / "labels.npy", y.astype(np.int32))
        return {
            "message": "Dataset prepared successfully.",
            "samples": int(len(X)),
            "features": int(X.shape[1]),
            "regime_distribution": {str(int(k)): int((y == k).sum()) for k in np.unique(y)},
            "note": "The repository's dataset loader samples rows randomly (seed 42). This is not a chronological capture-order evaluation.",
        }
    except Exception as exc:
        raise HTTPException(422, f"Dataset preparation failed: {exc}") from exc


@app.post("/api/analyse/{name}")
def analyse_dataset(name: str, regimes: int = 4, state_dim: int = 8, obs_dim: int = 10, window_size: int = 100, em_iterations: int = 8):
    folder = dataset_dir(name)
    x_path, y_path = folder / "features.npy", folder / "labels.npy"
    if not x_path.exists() or not y_path.exists():
        raise HTTPException(400, "Prepare the dataset before running an experiment.")
    if not 2 <= regimes <= 5 or not 2 <= obs_dim <= 20 or not 2 <= state_dim <= 20:
        raise HTTPException(400, "Model dimensions are outside supported limits.")
    try:
        from src.data_processing.feature_engineering import temporal_train_test_split, build_observation_sequence
        from src.inference.variational_switching import SwitchingSSM

        X = np.load(x_path)
        y = np.load(y_path)
        if len(np.unique(y)) < 2:
            raise ValueError("At least two label classes are required.")
        X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_ratio=0.2)
        if len(X_test) < window_size or len(X_train) < window_size:
            raise ValueError("Not enough samples for the selected window size.")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        components = min(obs_dim, X_train_scaled.shape[1], X_train_scaled.shape[0])
        pca = PCA(n_components=components, random_state=42)
        train_obs = pca.fit_transform(X_train_scaled)
        test_obs = pca.transform(X_test_scaled)

        # Use complete non-overlapping windows so the model receives a true sequence.
        train_count = len(train_obs) // window_size
        test_count = len(test_obs) // window_size
        train_seq = train_obs[:train_count * window_size].reshape(train_count, window_size, components)
        test_seq = test_obs[:test_count * window_size].reshape(test_count, window_size, components)
        train_labels = y_train[:train_count * window_size].reshape(train_count, window_size)
        test_labels = y_test[:test_count * window_size].reshape(test_count, window_size)
        if train_count == 0 or test_count == 0:
            raise ValueError("No complete observation windows could be formed.")

        model = SwitchingSSM(n_regimes=regimes, state_dim=state_dim, obs_dim=components)
        history = model.fit(train_seq[0], n_iter=em_iterations, verbose=False)
        # Evaluate each test window independently with the fitted model.
        probs, paths, hidden, pred_obs, ll_values, true = [], [], [], [], [], []
        for index in range(test_count):
            result = model.filter(test_seq[index])
            probs.append(result.regime_probs)
            paths.append(result.viterbi_path)
            hidden.append(result.filtered_means)
            pred_obs.append(result.predicted_observations)
            ll_values.append(float(result.log_likelihood))
            true.append(test_labels[index])
        regime_probs = np.concatenate(probs, axis=0)
        predicted = np.concatenate(paths, axis=0)
        actual = np.concatenate(true, axis=0)
        hidden_states = np.concatenate(hidden, axis=0)
        predicted_observations = np.concatenate(pred_obs, axis=0)
        length = min(len(actual), len(predicted), len(regime_probs))
        actual, predicted = actual[:length], predicted[:length]
        regime_probs, hidden_states = regime_probs[:length], hidden_states[:length]
        predicted_observations = predicted_observations[:length]
        observed = test_obs[:length]
        # Latent state IDs are arbitrary and do not map directly to dataset class IDs.
        # ARI and NMI compare clusterings without assuming a particular ID alignment.
        ari = float(adjusted_rand_score(actual, predicted))
        nmi = float(normalized_mutual_info_score(actual, predicted))
        attack_ids = [i for i in range(regimes) if i != 0]
        attack_ids = [i for i in attack_ids if i < regime_probs.shape[1]]
        attack_probability = regime_probs[:, attack_ids].sum(axis=1) if attack_ids else 1 - regime_probs[:, 0]
        # This is a regime-probability aggregate, not a calibrated binary attack classifier.
        result_payload = {
            "dataset": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "samples": int(len(X)),
            "feature_count": int(X.shape[1]),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "evaluated_steps": int(length),
            "regimes": int(regimes),
            "state_dim": int(state_dim),
            "obs_dim": int(components),
            "window_size": int(window_size),
            "em_iterations": int(em_iterations),
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "prediction_mse": float(np.mean((observed - predicted_observations) ** 2)),
            "log_likelihood": float(np.mean(ll_values)),
            "ground_truth_classes": sorted(int(v) for v in np.unique(actual)),
            "latent_regime_ids": sorted(int(v) for v in np.unique(predicted)),
            "regime_names": {str(i): REGIME_NAMES.get(i, f"Latent regime {i}") for i in range(regimes)},
            "regime_probabilities": regime_probs[::max(1, length // 500)].tolist(),
            "predicted_regimes": predicted[::max(1, length // 500)].tolist(),
            "true_regimes": actual[::max(1, length // 500)].tolist(),
            "attack_probability": attack_probability[::max(1, length // 500)].tolist(),
            "hidden_state": hidden_states[::max(1, length // 500), :min(3, hidden_states.shape[1])].tolist(),
            "training_log_likelihood": [float(v) for v in history],
            "regime_distribution": {str(int(k)): int((actual == k).sum()) for k in np.unique(actual)},
            "evaluation_note": "ARI and NMI compare inferred latent clusters with dataset labels without assuming that latent state IDs match class IDs. The dataset loader randomly samples rows before the holdout split, so this is not a strict chronological evaluation. The model is fitted on the first complete training window only; results should be treated as an exploratory baseline, not a validated production detector.",
        }
        latest_result(name).write_text(json.dumps(result_payload, default=safe_json, indent=2), encoding="utf-8")
        return {"message": "Experiment completed.", "result": result_payload}
    except Exception as exc:
        raise HTTPException(422, f"Experiment failed: {exc}") from exc


@app.get("/api/results/{name}")
def get_results(name: str):
    dataset_dir(name)
    path = latest_result(name)
    if not path.exists():
        raise HTTPException(404, "No experiment results yet. Prepare a dataset and run an experiment.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/reports/{name}")
def download_report(name: str):
    dataset_dir(name)
    path = latest_result(name)
    if not path.exists():
        raise HTTPException(404, "No report is available yet.")
    return FileResponse(path, media_type="application/json", filename=f"pard-ssm-{name}-report.json")
