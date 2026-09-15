# NetSeer

Offline-first network attack forecasting across the MITRE ATT&CK chain.

## Architecture

```text
Vite/React → FastAPI → existing Python data, scanner, forecasting and ML logic
```

## Final structure

```text
NetSeer/
├── frontend/                  Vite + React UI
└── backend/                   FastAPI, Python services, ML, models and data
    ├── api/
    ├── src/
    ├── scripts/
    ├── models/
    ├── data/
    ├── results/
    ├── tests/
    ├── config.yaml
    └── requirements.txt
```

## Requirements

- Python 3.10+
- Node.js 18+
- Optional: Npcap for ARP discovery and packet capture

## Install and run

```bash
# Terminal 1 — Backend
cd backend
python -m pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

```bash
# Terminal 2 — Frontend
cd frontend
npm ci
npm run dev
```

URLs:

- React: http://localhost:5173
- FastAPI: http://localhost:8000
- API docs: http://localhost:8000/docs

## Environment

`frontend/.env`:

```env
VITE_NETSEER_API_URL=http://127.0.0.1:8000
```

## API

- `GET /api/health`, `/api/config`, `/api/benchmarks`
- `POST /api/forecast`, `/api/analysis`, `/api/simulate`, `/api/scout`
- `GET /api/live/interfaces`, `/api/live/status`
- `POST /api/live/start`, `/api/live/stop`

## Production build

```bash
cd frontend
npm run build
npm run preview
```

## Troubleshooting

- If React cannot reach FastAPI, verify both servers are running and `VITE_NETSEER_API_URL` matches the backend URL.
- If a model is missing, verify the required checkpoint remains under `backend/models/`.
- Network scanning requires an authorized CIDR; ARP capture may require Npcap.
