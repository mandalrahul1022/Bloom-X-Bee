# backend/main.py
import os
import math
import asyncio
import datetime as dt
from typing import Tuple, Dict, Any
from hashlib import sha1
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from datetime import datetime
from dotenv import load_dotenv

# === Additions for multi-year trend ===
import os, time, io, zipfile, json, datetime as dt
from dataclasses import dataclass
from typing import List
import requests
import numpy as np
import pandas as pd
from scipy.stats import linregress
import pymannkendall as mk
from fastapi import HTTPException, Query

APPEEARS_API = "https://appeears.earthdatacloud.nasa.gov/api"
MODIS_EVI_LAYERS = [
    # EVI (you can switch to EVI2 layer if you prefer)
    {"product": "MOD13Q1.061", "layer": "250m_16_days_EVI"},
]
# For EVI2, use: "250m_16_days_EVI2" in same product.

def _get_token():
    t = os.getenv("APPEEARS_TOKEN")
    if t:
        return t
    user = os.getenv("EARTHDATA_USER")
    pw = os.getenv("EARTHDATA_PASS")
    if not (user and pw):
        raise RuntimeError("Set APPEEARS_TOKEN or EARTHDATA_USER/EARTHDATA_PASS in .env")
    r = requests.post(f"{APPEEARS_API}/login", json={"username": user, "password": pw})
    r.raise_for_status()
    return r.json()["token"]

def _submit_task_point(token: str, lat: float, lon: float, start: str, end: str):
    # Build a small area around point (buffer ~1km) to be robust to pixel edges
    # AppEEARS wants GeoJSON for area; we’ll give a tiny square polygon.
    d = 0.01  # ~1.1 km at equator
    poly = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "site"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon-d, lat-d],[lon+d, lat-d],[lon+d, lat+d],[lon-d, lat+d],[lon-d, lat-d]
                ]]
            }
        }]
    }
    task = {
        "task_type": "area",
        "task_name": f"beehive_{lat:.4f}_{lon:.4f}_{start}_{end}",
        "params": {
            "dates": [{"startDate": start, "endDate": end}],
            "layers": [{"product": L["product"], "layer": L["layer"]} for L in MODIS_EVI_LAYERS],
            "geo": poly,
            "output": {"format": {"type": "csv"}},
        }
    }
    h = {"Authorization": f"Bearer {token}"}
    r = requests.post(f"{APPEEARS_API}/task", headers=h, json=task, timeout=60)
    r.raise_for_status()
    return r.json()["task_id"]

def _wait_for_task(token: str, task_id: str, poll=10, timeout=1800):
    h = {"Authorization": f"Bearer {token}"}
    t0 = time.time()
    while True:
        r = requests.get(f"{APPEEARS_API}/task/{task_id}", headers=h, timeout=30)
        r.raise_for_status()
        s = r.json()["status"]
        if s == "done":
            return
        if s in ("failed", "cancelled"):
            raise RuntimeError(f"AppEEARS task {task_id} failed: {s}")
        if time.time() - t0 > timeout:
            raise RuntimeError("AppEEARS task timed out")
        time.sleep(poll)

def _download_csv_bundle(token: str, task_id: str) -> pd.DataFrame:
    h = {"Authorization": f"Bearer {token}"}
    # list bundle files
    r = requests.get(f"{APPEEARS_API}/bundle/{task_id}", headers=h, timeout=60)
    r.raise_for_status()
    files = r.json()["files"]
    # pick the CSV (summary per date/pixel)
    csv_files = [f for f in files if f["file_ext"] == "csv"]
    if not csv_files:
        raise RuntimeError("No CSV in bundle")
    # download the first CSV
    fid = csv_files[0]["file_id"]
    r = requests.get(f"{APPEEARS_API}/bundle/{task_id}/{fid}", headers=h, timeout=120)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    # usually one CSV inside
    inner_csv = [n for n in zf.namelist() if n.endswith(".csv")][0]
    with zf.open(inner_csv) as f:
        df = pd.read_csv(f)
    return df

def _aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    # AppEEARS CSVs include date and value columns per layer. Normalize to ['date','value'].
    # Find the first value column that matches our layer name.
    value_col = [c for c in df.columns if c.lower().endswith("_evi") or c.lower().endswith("_evi2")]
    if not value_col:
        raise RuntimeError("EVI/EVI2 column not found in CSV")
    value_col = value_col[0]
    out = df[["Date", value_col]].copy()
    out["date"] = pd.to_datetime(out["Date"])
    out["value"] = pd.to_numeric(out[value_col], errors="coerce")
    # MODIS scale factor for EVI is 0.0001; fix if needed
    # Heuristic: values typically in [-2000,10000]; rescale if > 2 in magnitude
    if out["value"].abs().median() > 2:
        out["value"] = out["value"] * 0.0001
    out = out.dropna(subset=["value"]).sort_values("date")
    # If multiple pixels per date, take median across pixels
    daily = out.groupby("date", as_index=False)["value"].median()
    return daily

def _annual_metric(daily: pd.DataFrame, how="mean") -> pd.DataFrame:
    g = daily.copy()
    g["year"] = g["date"].dt.year
    if how == "mean":
        ann = g.groupby("year", as_index=False)["value"].mean()
    elif how == "max":
        ann = g.groupby("year", as_index=False)["value"].max()
    elif how == "p90":
        ann = g.groupby("year", as_index=False)["value"].quantile(0.90)
    else:
        ann = g.groupby("year", as_index=False)["value"].mean()
    return ann

def _trend_stats(ann: pd.DataFrame):
    years = ann["year"].to_numpy()
    vals = ann["value"].to_numpy()
    # Linear trend per year
    lr = linregress(years, vals)
    # Mann-Kendall non-parametric trend
    mkres = mk.original_test(vals) if len(vals) >= 8 else None
    return {
        "slope_per_year": lr.slope,
        "intercept": lr.intercept,
        "p_value_linear": lr.pvalue,
        "r2": (lr.rvalue ** 2),
        "mk_trend": (mkres.trend if mkres else None),
        "mk_p_value": (mkres.p if mkres else None),
    }

from fastapi import APIRouter
router = APIRouter()

@router.get("/api/trend")
def trend(
    lat: float = Query(...),
    lon: float = Query(...),
    start: str = Query("2001-01-01"),
    end: str = Query(dt.date.today().isoformat()),
    metric: str = Query("mean"),  # mean|max|p90
):
    try:
        token = _get_token()
        task_id = _submit_task_point(token, lat, lon, start, end)
        _wait_for_task(token, task_id)
        df = _download_csv_bundle(token, task_id)
        daily = _aggregate_to_daily(df)
        if daily.empty:
            raise HTTPException(404, detail="No EVI data returned for this location/time.")
        ann = _annual_metric(daily, how=metric)
        stats = _trend_stats(ann)
        return {
            "place_id": f"{lat:.4f},{lon:.4f}",
            "metric": metric,
            "years": ann["year"].tolist(),
            "values": [round(float(v), 4) for v in ann["value"].tolist()],
            "trend": stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Register the router on your existing FastAPI app object:
# app.include_router(router)


# Load secrets from .env (EARTHDATA_* ...)
load_dotenv()

app = FastAPI(title="Phenology API", version="1.1.1")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Auth / headers (prefer Bearer token, else Basic) ----
EARTHDATA_USER = os.getenv("EARTHDATA_USER")
EARTHDATA_PASS = os.getenv("EARTHDATA_PASS")
EARTHDATA_TOKEN = os.getenv("EARTHDATA_TOKEN")

HEADERS = {"Accept": "application/json"}
AUTH = None
if EARTHDATA_TOKEN:
    HEADERS["Authorization"] = f"Bearer {EARTHDATA_TOKEN}"
else:
    if EARTHDATA_USER and EARTHDATA_PASS:
        AUTH = httpx.BasicAuth(EARTHDATA_USER, EARTHDATA_PASS)
    # else leave AUTH=None; we'll raise later if neither token nor basic is available

logger = logging.getLogger("uvicorn.error")

# Harmony (LP DAAC) endpoints for MODIS VI (EVI + PixelReliability)
COLLS = [
    (
        "MOD13Q1",
        "https://harmony.earthdata.nasa.gov/harmony/services/gt/tile/collections/MOD13Q1/ogc-api-coverages/1.0.0/coverage",
    ),
    (
        "MYD13Q1",
        "https://harmony.earthdata.nasa.gov/harmony/services/gt/tile/collections/MYD13Q1/ogc-api-coverages/1.0.0/coverage",
    ),
    # If you want EVI2 instead, switch to HLS v2 collections and change `parameter`/parsing accordingly:
    # ("HLSS30.v2.0","https://harmony.earthdata.nasa.gov/harmony/services/gt/tile/collections/HLSS30.v2.0/ogc-api-coverages/1.0.0/coverage"),
    # ("HLSL30.v2.0","https://harmony.earthdata.nasa.gov/harmony/services/gt/tile/collections/HLSL30.v2.0/ogc-api-coverages/1.0.0/coverage"),
]

# ----------------------------- helpers ---------------------------------
def bbox_from_point(lat: float, lon: float, half_deg: float = 0.01) -> Tuple[float, float, float, float]:
    # (minLon, minLat, maxLon, maxLat)
    return (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)


def _safe_import_rasterio():
    try:
        import rasterio
        from rasterio.io import MemoryFile
        return rasterio, MemoryFile
    except Exception as e:
        raise HTTPException(
            500,
            f"rasterio not available: {e}. Install via conda or add to requirements.",
        ) from e


@app.get("/health")
def health():
    return {"status": "ok"}


async def fetch_modis_timeseries(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Use Harmony OGC Coverages to pull MODIS VI time series (JSON coverages).
    We probe the JSON to discover the actual range keys (e.g., 'EVI', 'EVI_16DAY',
    'PixelReliability', 'pixel_reliability') and extract center-pixel values.
    Returns columns: date, evi, qa_pr
    """
    if not (EARTHDATA_TOKEN or (EARTHDATA_USER and EARTHDATA_PASS)):
        raise HTTPException(500, "No Earthdata credentials found. Set EARTHDATA_TOKEN or EARTHDATA_USER/PASS in .env")

    # tiny bbox around the point; send subset as multiple params
    lon0, lat0, lon1, lat1 = bbox_from_point(lat, lon, 0.01)
    params_common = {
        # IMPORTANT: pass subset as a list so httpx encodes subset=lat(...)&subset=lon(...)
        "subset": [f"lat({lat0}:{lat1})", f"lon({lon0}:{lon1})"],
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "format": "application/json",
        # no explicit 'parameter' so we can discover keys in 'ranges'
    }

    rows = []
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
        for coll_name, cov_url in COLLS:
            logger.info("→ Harmony %s params=%s", coll_name, params_common)
            r = await client.get(cov_url, params=params_common, headers=HEADERS, auth=AUTH)
            logger.info("← Harmony %s %s", r.status_code, r.url)

            if r.status_code != 200:
                logger.error("Harmony error (%s): %s", r.status_code, r.text[:400])
                continue

            try:
                cov = r.json()
            except Exception as e:
                logger.error("JSON parse error: %s", e)
                continue

            domain = (cov.get("domain") or {}).get("axes") or {}
            t_vals = (domain.get("t") or {}).get("values") or []
            ranges = cov.get("ranges") or {}
            if not t_vals or not ranges:
                logger.error("No t axis or ranges in coverage JSON")
                continue

            # discover viable keys
            keys = {k.lower(): k for k in ranges.keys()}
            # try to pick EVI-like key
            evi_key = None
            for cand in ["evi", "evi_16day", "modis_evi"]:
                if cand in keys:
                    evi_key = keys[cand]; break
            if evi_key is None:
                # last resort: any key containing 'evi'
                for k in ranges.keys():
                    if "evi" in k.lower():
                        evi_key = k; break

            # QA key guesses
            qa_key = None
            for cand in ["pixelreliability", "pixel_reliability", "reliability"]:
                if cand in keys:
                    qa_key = keys[cand]; break
            if qa_key is None:
                # ok to proceed without QA (we’ll set 3 = poor)
                logger.info("No PixelReliability key found; proceeding without QA")

            evi_rng = ranges.get(evi_key) if evi_key else None
            if not evi_rng:
                logger.error("No EVI-like range key found in coverage; keys=%s", list(ranges.keys()))
                continue

            evi_vals = evi_rng.get("values") or []
            scale = float(evi_rng.get("scaleFactor", 1.0))
            qa_vals = (ranges.get(qa_key) or {}).get("values") if qa_key else None

            # Pick the center cell in the bbox across time: values are likely flattened per time
            # Many coverages return a single value per time for a tiny bbox; if it's a grid,
            # you may get multiple per t; we take the first (or center) per timestamp.
            # Align lengths conservatively:
            n = min(len(t_vals), len(evi_vals))
            for i in range(n):
                v = evi_vals[i]
                if v is None:
                    continue
                try:
                    vv = float(v) * scale
                except Exception:
                    continue
                # MODIS EVI usually scaled by 1e-4; clip to [0,1]
                evi = max(0.0, min(1.0, vv))
                qa_pr = int(qa_vals[i]) if (qa_vals and i < len(qa_vals) and qa_vals[i] is not None) else 3
                # t_vals can be ISO or ordinal; stringify robustly
                dstr = str(t_vals[i])[:10]
                dtobj = pd.to_datetime(dstr, errors="coerce")
                if pd.isna(dtobj):
                    continue
                rows.append({"date": dtobj, "evi": evi, "qa_pr": qa_pr, "src": coll_name})

    if not rows:
        raise HTTPException(
            502,
            "No MODIS rows returned. Likely causes: subset encoding, variable names, date window, or credentials.",
        )

    df = (
        pd.DataFrame(rows)
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    df["date"] = df["date"].dt.normalize()

    # Prefer QA 0/1 when present
    df["qa_good"] = df["qa_pr"].isin([0, 1])

    def agg(g: pd.DataFrame):
        g_good = g[g["qa_good"]]
        if len(g_good):
            evi = float(g_good["evi"].max())
            qa_pr = int(g_good.loc[g_good["evi"].idxmax()]["qa_pr"])
        else:
            ix = g["qa_pr"].idxmin()
            evi = float(g.loc[ix]["evi"])
            qa_pr = int(g.loc[ix]["qa_pr"])
        return pd.Series({"evi": evi, "qa_pr": qa_pr})

    return df.groupby("date").apply(agg).reset_index()


def to_daily(df16: pd.DataFrame) -> pd.DataFrame:
    """
    Convert sparse 16-day samples to a daily series (DatetimeIndex) using time interpolation.
    Keeps values in [0,1] and returns columns: date, evi2 (float).
    """
    df = df16[df16["qa_pr"].isin([0, 1])].copy()
    if df.empty:
        df = df16.copy()
    df = df.sort_values("date")

    idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    out = df.set_index("date").reindex(idx)

    out["evi"] = (
        out["evi"]
        .astype(float)
        .interpolate(method="time")
        .ffill()
        .bfill()
        .clip(lower=0.0, upper=1.0)
    )

    out["evi2"] = out["evi"]  # naming stays evi2 for downstream
    out["date"] = out.index
    return out.reset_index(drop=True)[["date", "evi2"]]


def phenophase_dates(daily: pd.DataFrame):
    """
    Detect SoS, Peak, EoS indices from a daily dataframe with columns: date, evi2.
    Uses a simple threshold: vmin + 0.2 * (vmax - vmin).
    """
    e = np.asarray(daily["evi2"].values, dtype=float)
    if e.size == 0:
        return 0, 0, 0
    vmin, vmax = float(np.min(e)), float(np.max(e))
    thr = vmin + 0.2 * (vmax - vmin)

    sos = next((i for i, v in enumerate(e) if v >= thr), None)
    eos = max((i for i, v in enumerate(e) if v >= thr), default=None)
    peak = int(np.argmax(e))

    if sos is None:
        sos = int(0.25 * len(e))
    if eos is None:
        eos = int(0.75 * len(e))
    return int(sos), int(peak), int(eos)


def uncertainty_days(daily: pd.DataFrame, iters: int = 100):
    """
    Bootstrap-style uncertainty (± days) for SoS/Peak/EoS.
    Works whether or not the input already has a DatetimeIndex.
    """
    rng = np.random.default_rng(42)

    base = daily.copy()
    use_time = False
    if not isinstance(base.index, pd.DatetimeIndex) and "date" in base.columns:
        try:
            base = base.set_index("date")
            use_time = isinstance(base.index, pd.DatetimeIndex)
        except Exception:
            use_time = False

    n = len(base)
    if n == 0:
        return {"sos": 7.0, "peak": 6.0, "eos": 8.0}

    sosL, pkL, eosL = [], [], []
    for _ in range(iters):
        mask = rng.choice([True, False], size=n, p=[0.85, 0.15])
        d = base.copy()
        d.loc[~mask, "evi2"] = np.nan
        if use_time:
            d["evi2"] = d["evi2"].interpolate(method="time").ffill().bfill()
        else:
            d["evi2"] = d["evi2"].interpolate(method="linear").ffill().bfill()
        tmp = d.reset_index()
        s, p, e = phenophase_dates(tmp)
        sosL.append(s)
        pkL.append(p)
        eosL.append(e)

    def half(a):
        lo, hi = np.percentile(a, [16, 84])
        return float((hi - lo) / 2.0)

    return {"sos": half(sosL), "peak": half(pkL), "eos": half(eosL)}


# ----------------------------- tiny cache ---------------------------------
_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL = 60 * 60  # 1 hour

def cache_get(key: str):
    rec = _CACHE.get(key)
    if not rec:
        return None
    if (dt.datetime.utcnow().timestamp() - rec["ts"]) > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return rec["val"]

def cache_put(key: str, val: Any):
    _CACHE[key] = {"ts": dt.datetime.utcnow().timestamp(), "val": val}


# ----------------------------- routes ---------------------------------
@app.get("/api/evi2")
async def api_evi2(
    lat: float,
    lon: float,
    start: str,
    end: str,
    place_id: str = "site",
    mock: int = 0,
):
    """
    Returns current-season daily EVI2 (from MODIS EVI), phenophase dates, QA summary and uncertainty.
    """
    key = sha1(f"{lat:.4f},{lon:.4f},{start},{end},{mock}".encode()).hexdigest()
    hit = cache_get(key)
    if hit is not None:
        return JSONResponse(hit)

    if mock:
        # Synthetic seasonal curve for demo/testing
        dates = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
        evi = np.clip(0.3 + 0.3 * np.sin(np.linspace(0, 2 * math.pi, len(dates))), 0, 1.0)
        daily = pd.DataFrame({"date": dates, "evi2": evi})
    else:
        df16 = await fetch_modis_timeseries(lat, lon, start, end)
        daily = to_daily(df16)

    s_i, p_i, e_i = phenophase_dates(daily)
    dates = pd.to_datetime(daily["date"]).dt.date
    sos_d, peak_d, eos_d = dates.iloc[s_i], dates.iloc[p_i], dates.iloc[e_i]
    unc = uncertainty_days(daily)

    qa_good = int((daily["evi2"] > 0).sum())
    qa_bad = 0  # if you later include explicit cloud/snow, update this
    valid_frac = float(qa_good) / float(max(1, qa_good + qa_bad))

    out = {
        "place_id": place_id,
        "series": [
            {"date": d.isoformat(), "evi2": float(v)}
            for d, v in zip(daily["date"].dt.date, daily["evi2"])
        ],
        "sos_date": sos_d.isoformat(),
        "peak_date": peak_d.isoformat(),
        "eos_date": eos_d.isoformat(),
        "uncertainty_days": {k: round(v, 1) for k, v in unc.items()},
        "quality": {
            "valid_frac": round(valid_frac, 3),
            "qa_good": qa_good,
            "qa_cloud_snow": qa_bad,
        },
    }
    cache_put(key, out)
    return JSONResponse(out)


@app.get("/api/evi2_history", summary="Api Evi2 History",
         description="Returns multi-year phenology summary for a site.")
async def api_evi2_history(
    lat: float,
    lon: float,
    years: int = 6,
    end: str | None = None,
    mock: int = 0,
):
    """
    Response:
      {
        "place_id": "lat,lon",
        "records": [
          {"year": 2025, "sos_doy": 52, "peak_doy": 94, "eos_doy": 210},
          ...
        ]
      }
    """
    years = int(max(1, min(10, years)))  # clamp 1..10
    end_dt = pd.to_datetime(end).normalize() if end else pd.Timestamp.today().normalize()
    year_list = [int((end_dt.year - i)) for i in range(years)]

    records: list[dict[str, Any]] = []

    if mock:
        # Fast synthetic history
        rng = np.random.default_rng(123)
        for y in year_list[::-1]:
            start_dt = pd.Timestamp(year=y, month=1, day=1)
            days = pd.date_range(start_dt, start_dt + pd.Timedelta(days=364), freq="D")
            phase = rng.normal(0, 6.0)
            width = 30 + rng.normal(0, 3.0)
            amp = 0.45 + rng.normal(0, 0.02)
            base = 0.20 + rng.normal(0, 0.01)

            def seasonal(doy, peak):
                x = (doy - peak) / width
                return max(0.0, min(1.0, base + amp * math.exp(-0.5 * x * x)))

            peak_day = 100 + int(phase)
            vals = []
            for i, d in enumerate(days, start=1):
                v = seasonal(i, peak_day) + rng.normal(0, 0.015)
                vals.append({"date": d, "evi2": float(np.clip(v, 0, 1))})
            daily = pd.DataFrame(vals)

            s_i, p_i, e_i = phenophase_dates(daily)
            records.append(
                {"year": y, "sos_doy": int(s_i + 1), "peak_doy": int(p_i + 1), "eos_doy": int(e_i + 1)}
            )

        records.sort(key=lambda r: r["year"])
        return JSONResponse({"place_id": f"{lat:.4f},{lon:.4f}", "records": records})

    # Real calls to Harmony (sequential to be gentle on rate limits)
    async def one_year(y: int):
        start_dt = pd.Timestamp(year=y, month=1, day=1)
        end_dt_y = pd.Timestamp(year=y, month=12, day=31)
        df16 = await fetch_modis_timeseries(
            lat, lon, start_dt.date().isoformat(), end_dt_y.date().isoformat()
        )
        daily = to_daily(df16)
        s_i, p_i, e_i = phenophase_dates(daily)
        base = pd.to_datetime(daily.iloc[0]["date"]).normalize()
        sos_day = (pd.to_datetime(daily.iloc[s_i]["date"]).normalize() - base).days + 1
        peak_day = (pd.to_datetime(daily.iloc[p_i]["date"]).normalize() - base).days + 1
        eos_day = (pd.to_datetime(daily.iloc[e_i]["date"]).normalize() - base).days + 1
        return {"year": y, "sos_doy": int(sos_day), "peak_doy": int(peak_day), "eos_doy": int(eos_day)}

    for y in year_list:
        try:
            rec = await one_year(y)
            records.append(rec)
        except HTTPException as e:
            if e.status_code in (401, 403):
                raise
        except Exception as ex:
            logger.error("history year %s failed: %s", y, ex)
            continue

    records.sort(key=lambda r: r["year"])
    if not records:
        return JSONResponse({"place_id": f"{lat:.4f},{lon:.4f}", "records": []})
    return JSONResponse({"place_id": f"{lat:.4f},{lon:.4f}", "records": records})


# USA-NPN open-flowers proxy — returns zeros unless you wire a real endpoint
@app.get("/api/npn_openflowers")
async def npn_openflowers(
    lat: float,
    lon: float,
    start: str = Query(default=None, alias="from"),
    end: str = Query(default=None, alias="to"),
    requestSource: str = "Caldwell University",
    contact: str = "Rahul Mandal",
):
    return JSONResponse(
        {"open": 0, "total": 0, "percent": 0.0, "source": requestSource, "contact": contact}
    )
