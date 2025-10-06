# project description

Are you a beekeeper worried about honey yields? BloomWatch turns global satellite streams into local, day-by-day bloom intelligence you can use. We built a simple open web app and predictive model that fuses NASA Earth observations (HLS EVI2, MODIS/VIIRS phenology) with local weather to map flowering, forecast peak windows, and estimate honey yield for major cities worldwide. Each location page displays daily bloom curves, month-by-month seasonality, a 20-year trend line illustrating shifting bloom timing, and practical guidance on pollination timing and hive placement.


## Run
Use of simulation models like 

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
- 

## Notes

# Bloom-X-Bee
