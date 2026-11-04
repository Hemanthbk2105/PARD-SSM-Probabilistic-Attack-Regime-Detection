from __future__ import annotations

import json
import shutil
import sys
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

MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB per CSV file
REGIME_COLORS = ["#47d7ac", "#f5bd55", "#ff8a65", "#ff5f72", "#a78bfa"]

app = FastAPI(
    title="PARD-SSM Security API",
    description="Security analytics API for probabilistic attack-regime detection.",
    version="1.1.0",
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
    return {
        "status": "online",
        "service": "PARD-SSM Security",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


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
        temporary = destination.with_suffix(destination.suffix + ".uploading")
        total = 0
        try:
            with temporary.open("wb") as target:
                while chunk := await uploaded.read(8 * 1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"{filename} exceeds the 10 GiB per-file limit.")
                    target.write(chunk)
            temporary.replace(destination)
            saved.append({"name": filename, "size_bytes": total})
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            await uploaded.close()
    return {"message": f"Uploaded {len(saved)} file(s).", "files": saved}


@app.post("/api/prepare/{name}")
def prepare_dataset(name: str, sample_size: int = 50000):
    folder = dataset_dir(name)
    if not list(folder.glob("*.csv")):
        raise HTTPException(400, "Upload the dataset CSV files first.")
    if sample_size < 0:
        raise HTTPException(400, "sample_size must be zero (all rows) or a positive integer.")
    try:
        from src.data_processing.dataset_loader import load_cicids2017, load_unswnb15

        loader = load_cicids2017 if name == "cicids2017" else load_unswnb15
        # Load in capture-time order without the loader's random row sampler.
        X, y = loader(str(folder), sample_size=0)
        total_rows = len(X)
        if X.ndim != 2 or total_rows != len(y) or total_rows < 120:
            raise ValueError("The prepared dataset must contain at least 120 matching rows.")

        # If a runtime cap is selected, take evenly spaced positions. This keeps
        # the capture order intact, unlike random sampling before a time split.
        if sample_size and total_rows > sample_size:
            selected = np.linspace(0, total_rows - 1, num=sample_size, dtype=np.int64)
            X, y = X[selected], y[selected]

        if not np.isfinite(X).all():
            raise ValueError("Prepared features contain NaN or infinite values.")
        np.save(folder / "features.npy", X.astype(np.float32))
        np.save(folder / "labels.npy", y.astype(np.int32))
        return {
            "message": "Dataset prepared successfully.",
            "source_rows": int(total_rows),
            "samples": int(len(X)),
            "features": int(X.shape[1]),
            "sampling": "all_rows" if sample_size == 0 or total_rows <= sample_size else "evenly_spaced_chronological",
            "regime_distribution": {str(int(k)): int((y == k).sum()) for k in np.unique(y)},
            "note": "CSV rows are sorted by capture timestamp by the repository loader. Any sample cap uses evenly spaced row positions to preserve order. The holdout split is chronological by row order; verify source timestamps and session boundaries before interpreting it as a cross-session evaluation.",
        }
    except Exception as exc:
        raise HTTPException(422, f"Dataset preparation failed: {exc}") from exc


@app.post("/api/analyse/{name}")
def analyse_dataset(name: str, regimes: int = 4, state_dim: int = 8, obs_dim: int = 10, em_iterations: int = 8):
    folder = dataset_dir(name)
    x_path, y_path = folder / "features.npy", folder / "labels.npy"
    if not x_path.exists() or not y_path.exists():
        raise HTTPException(400, "Prepare the dataset before running an experiment.")
    if not 2 <= regimes <= 5 or not 2 <= obs_dim <= 20 or not 2 <= state_dim <= 20:
        raise HTTPException(400, "Model dimensions are outside supported limits.")
    if not 1 <= em_iterations <= 30:
        raise HTTPException(400, "EM iterations must be between 1 and 30.")
    try:
        from src.data_processing.feature_engineering import temporal_train_test_split
        from src.inference.variational_switching import SwitchingSSM

        X = np.load(x_path)
        y = np.load(y_path)
        if len(np.unique(y)) < 2:
            raise ValueError("At least two label classes are required.")
        X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_ratio=0.2)
        if len(X_test) < 2 or len(X_train) < 2:
            raise ValueError("The train and test splits must each contain at least two samples.")

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        components = min(obs_dim, X_train_scaled.shape[1], X_train_scaled.shape[0])
        pca = PCA(n_components=components, random_state=42)
        train_obs = pca.fit_transform(X_train_scaled)
        test_obs = pca.transform(X_test_scaled)

        model = SwitchingSSM(n_regimes=regimes, state_dim=state_dim, obs_dim=components)
        # Bound EM fitting cost, then replay the complete training sequence with
        # the fitted parameters to obtain the correct causal state for test time.
        training_steps = min(len(train_obs), 5000)
        history = model.fit(train_obs[:training_steps], n_iter=em_iterations, verbose=False)
        train_filter = model.filter(train_obs)
        result = model.filter(test_obs, initial_state=train_filter.final_state)

        regime_probs = result.regime_probs
        predicted = np.argmax(regime_probs, axis=1)
        actual = y_test[:len(predicted)]
        hidden_states = result.filtered_means
        predicted_observations = result.predicted_observations
        length = min(len(actual), len(predicted), len(regime_probs), len(predicted_observations))
        actual, predicted = actual[:length], predicted[:length]
        regime_probs, hidden_states = regime_probs[:length], hidden_states[:length]
        predicted_observations = predicted_observations[:length]
        observed = test_obs[:length]

        ari = float(adjusted_rand_score(actual, predicted))
        nmi = float(normalized_mutual_info_score(actual, predicted))
        regime_uncertainty = 1.0 - np.max(regime_probs, axis=1)
        sample_step = max(1, length // 500)
        result_payload = {
            "dataset": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "samples": int(len(X)),
            "feature_count": int(X.shape[1]),
            "train_samples": int(len(X_train)),
            "model_training_steps": int(training_steps),
            "test_samples": int(len(X_test)),
            "evaluated_steps": int(length),
            "regimes": int(regimes),
            "state_dim": int(state_dim),
            "obs_dim": int(components),
            "em_iterations": int(em_iterations),
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "prediction_mse": float(np.mean((observed - predicted_observations) ** 2)),
            "log_likelihood": float(result.log_likelihood / max(1, length)),
            "ground_truth_classes": sorted(int(v) for v in np.unique(actual)),
            "latent_regime_ids": sorted(int(v) for v in np.unique(predicted)),
            "regime_names": {str(i): f"Latent regime {i}" for i in range(regimes)},
            "regime_probabilities": regime_probs[::sample_step].tolist(),
            "predicted_regimes": predicted[::sample_step].tolist(),
            "true_regimes": actual[::sample_step].tolist(),
            "regime_uncertainty": regime_uncertainty[::sample_step].tolist(),
            "hidden_state": hidden_states[::sample_step, :min(3, hidden_states.shape[1])].tolist(),
            "training_log_likelihood": [float(v) for v in history],
            "ground_truth_distribution": {str(int(k)): int((actual == k).sum()) for k in np.unique(actual)},
            "evaluation_note": "The dataset loader sorts source rows by capture timestamp. If a sample cap is used, evenly spaced row positions preserve that order. The holdout is the final 20% of prepared rows. Confirm timestamp ordering, session boundaries and duplicate-time handling before treating this as a strict temporal benchmark. RobustScaler and PCA are fitted on training rows only. The SSSM is fitted on at most the first 5,000 training observations for runtime control; after fitting, the complete training sequence is filtered to carry its final posterior into the held-out test sequence. ARI and NMI compare inferred latent clusters with dataset labels without assuming state IDs match class IDs. The model uses approximate generalized-EM updates, not exact switching-Kalman EM. Regime uncertainty is not an attack probability. This is an experimental research workflow, not a validated production intrusion detector.",
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
