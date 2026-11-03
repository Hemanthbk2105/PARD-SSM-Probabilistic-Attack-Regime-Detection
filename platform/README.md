# PARD-SSM Intelligence Platform

A separate React + FastAPI research interface for the existing PARD-SSM repository. The original model code remains in place; this platform calls the existing dataset loaders, temporal split helper and SwitchingSSM implementation.

## Architecture

- \`frontend/\`: React, Vite, Recharts and Lucide UI.
- \`backend/main.py\`: FastAPI endpoints for health, CSV upload, preprocessing, model execution and report download.
- \`data/\`: local uploaded CSV files and prepared NumPy arrays (not committed).
- \`results/\`: latest JSON experiment outputs (not committed).

## Requirements

- Python 3.10+
- Node.js 18+
- Dataset CSV files downloaded from the official CICIDS2017 or UNSW-NB15 sources.

## Run locally (Windows PowerShell)

Terminal 1 — backend:

\`\`\`powershell
cd platform/backend
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
\`\`\`

Terminal 2 — frontend:

\`\`\`powershell
cd platform/frontend
npm install
npm run dev
\`\`\`

Open http://localhost:5173. The frontend calls http://localhost:8000 by default. To use a hosted API, create \`platform/frontend/.env\` with:

\`\`\`env
VITE_API_URL=https://your-api-host.example
\`\`\`

## First experiment

1. Open Dataset Studio.
2. Select CICIDS2017 or UNSW-NB15 in the top selector.
3. Upload the extracted CSV file(s) for that dataset.
4. Click **Prepare dataset**. The platform uses the repository's existing \`src/data_processing/dataset_loader.py\`.
5. Open Model Laboratory and run the experiment.
6. Review the actual output in Command Center, Threat Intelligence, Regime Explorer and Research Reports.

## API

- \`GET /api/health\`
- \`GET /api/datasets\`
- \`POST /api/upload/{cicids2017|unswnb15}\` — multipart CSV files
- \`POST /api/prepare/{dataset}?sample_size=50000\`
- \`POST /api/analyse/{dataset}?regimes=4&state_dim=8&obs_dim=10&window_size=100&em_iterations=8\`
- \`GET /api/results/{dataset}\`
- \`GET /api/reports/{dataset}\`

## Research integrity notes

- No fabricated or hard-coded evaluation metrics are shown. Empty states remain until a real backend experiment is available.
- The current repository loaders randomly sample rows before splitting. The dashboard explicitly reports this; do not describe those runs as strict chronological holdouts.
- Latent regime IDs are model states, not guaranteed semantic labels. The interface keeps this distinction visible.
- The current experiment trains the SSSM on the first training window and filters held-out windows. Treat the result as an experimental integration baseline, not a production intrusion detection service.
- Uploaded traffic files stay on the machine running the backend. Do not expose the API publicly without authentication, upload limits, storage controls and deployment hardening.

## Dataset sources

- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
