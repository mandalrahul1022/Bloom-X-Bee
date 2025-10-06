# project description

Are you a beekeeper worried about honey yields? BloomWatch turns global satellite streams into local, day-by-day bloom intelligence you can use. We built a simple open web app and predictive model that fuses NASA Earth observations (HLS EVI2, MODIS/VIIRS phenology) with local weather to map flowering, forecast peak windows, and estimate honey yield for major cities worldwide. Each location page displays daily bloom curves, month-by-month seasonality, a 20-year trend line illustrating shifting bloom timing, and practical guidance on pollination timing and hive placement.

# Why we built this
Bloom timing is shifting. Beekeepers have only a narrow window to place strong colonies, support pollination, and avoid lost yield—but the signals that predict those windows are buried across satellites and weather feeds. BloomWatch pulls that signal into one clear place so everyday decisions are easier.

# What BloomWatch does
 BloomWatch turns NASA satellite greenness and local weather into a simple daily bloom curve, a predicted peak window, and a practical honey-yield estimate for each city. The app also shows a month-by-month seasonal view and a 20-year trend so shifts are obvious at a glance. It’s built first for beekeepers and orchard crops, with an open design that researchers and educators can extend.

# How it works
We combine Harmonized Landsat–Sentinel (HLS) imagery with phenology from MODIS (Moderate Resolution Imaging Spectroradiometer) and VIIRS (Visible Infrared Imaging Radiometer Suite) plus local weather. After filtering clouds and bad pixels, we extract EVI2 (Enhanced Vegetation Index 2), smooth it to a city-level curve, and serve results through a clean API to the map and charts.

# What we used (short list)
Frontend: React + Vite, MapLibre/Leaflet, Chart.js/Recharts (responsive, accessible).
Backend: Python (FastAPI; pandas/numpy; xarray/rioxarray & rasterio for geospatial; scikit-learn/statsmodels for forecasting; asyncio + caching).
Data: NASA HLS (EVI2), MODIS/VIIRS phenology, local weather/reanalysis (e.g., ERA5/NOAA), accessed via Earthdata/AppEEARS with strict quality masks.
Dev workflow: virtualenv/pip, Node/NPM, Jupyter notebooks, Git/GitHub, .env configs; served with Uvicorn (Nginx ready). No special hardware—standard laptops.

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

# Bloom-X-Bee
