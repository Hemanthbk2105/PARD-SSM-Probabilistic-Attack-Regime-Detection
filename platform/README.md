# PARD-SSM Intelligence Platform

A React + FastAPI research interface for the existing PARD-SSM repository. The original model implementation remains the inference engine; the platform handles local dataset ingestion, preparation, experiment execution and result export.

## Architecture

- `frontend/`: React, Vite, Recharts and Lucide UI.
- `backend/main.py`: FastAPI endpoints for health, CSV upload, preprocessing, model execution and report download.
- `data/`: local uploaded CSV files and prepared NumPy arrays (not committed).
- `results/`: latest JSON experiment outputs (not committed).

## Requirements

- Python 3.10+
- Node.js 18+
- Dataset CSV files downloaded from the official CICIDS2017 or UNSW-NB15 sources.
- Sufficient RAM and free disk space for the selected dataset. Full CICIDS2017 processing can require substantial resources.

## Run locally (Windows PowerShell)

Terminal 1 — backend:

```powershell
cd platform/backend
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Terminal 2 — frontend:

```powershell
cd platform/frontend
npm install
npm run dev
```

Open http://localhost:5173. The frontend calls http://localhost:8000 by default. To use a hosted API, create `platform/frontend/.env` with:

```env
VITE_API_URL=https://your-api-host.example
```

## First experiment

1. Open Dataset Management.
2. Select CICIDS2017 or UNSW-NB15 in the top selector.
3. Upload the extracted CSV file(s) for that dataset.
4. Click **Prepare dataset**. The platform uses the repository's existing `src/data_processing/dataset_loader.py`.
5. The preparation endpoint loads rows in capture-time order. If a sample cap is selected, it uses evenly spaced row positions rather than random sampling, preserving the row order.
6. Open Model Configuration and run the experiment.
7. Review the actual output in Command Center, Threat Intelligence, Regime Explorer and Security Reports.

## API

- `GET /api/health`
- `GET /api/datasets`
- `POST /api/upload/{cicids2017|unswnb15}` — multipart CSV files; maximum 10 GiB per file.
- `POST /api/prepare/{dataset}?sample_size=50000` — use `sample_size=0` to retain all rows.
- `POST /api/analyse/{dataset}?regimes=4&state_dim=8&obs_dim=10&em_iterations=8`
- `GET /api/results/{dataset}`
- `GET /api/reports/{dataset}`

## Model transparency and evaluation notes

- No fabricated or hard-coded evaluation metrics are shown. Empty states remain until a real backend experiment is available.
- Dataset rows are sorted by capture timestamp by the repository loader. A sample cap uses evenly spaced positions so it does not randomly shuffle the prepared sequence.
- The train/test holdout uses the final 20% of prepared rows. Check timestamp order, session boundaries and duplicate timestamps before describing a result as a strict chronological or cross-session benchmark.
- RobustScaler and PCA are fitted on training rows only and then applied to test rows.
- The SSSM is fitted on at most the first 5,000 training observations for runtime control. After fitting, the complete training sequence is filtered with the fitted parameters, and its final posterior state is carried into the held-out test sequence.
- The model uses approximate generalized-EM updates, not exact switching-Kalman EM.
- Latent regime IDs are model states, not guaranteed semantic labels. The interface keeps this distinction visible.
- Regime uncertainty is not an attack probability. ARI and NMI compare clusters to dataset labels without assuming matching state IDs.
- Uploaded traffic files stay on the machine running the backend. Do not expose the API publicly without authentication, upload limits, storage controls, background job handling and deployment hardening.

## Current scope

This is a local research workflow and an experimental integration baseline. It does not yet provide a multi-user account system, durable experiment database/history, background job queue, chunk-resumable uploads, live packet capture or a validated production intrusion-detection service. Those must be implemented and tested before public deployment.

## Dataset sources

- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
