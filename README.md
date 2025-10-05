# Phenology — 10 Places

Split frontend/backend with NASA Harmony EVI fetch, NPN stub, and Node proxy.

## Run

### Backend
- `pip install -r backend/requirements.txt`
- `export EARTHDATA_USER=... EARTHDATA_PASS=...`
- `uvicorn backend.main:app --reload --port 8000`

### Frontend + Proxy
- `npm install`
- `npm run start`
- Open http://localhost:5173

## Precompute the JSON
- `python backend/precompute.py` (with API running on :8000)
- It writes `frontend/phenology_10_places.json`

## Notes
- If NASA calls fail, backend can be hit with `mock=1` to get a synthetic series.
- The UI uses your JSON by default and falls back to a synthetic dataset if missing.
- Wire `/api/npn_openflowers` to a real USA-NPN endpoint when ready.
