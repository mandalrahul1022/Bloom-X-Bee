import os, json, datetime as dt, asyncio
from pathlib import Path
import httpx

API = os.getenv("PHENO_API_URL", "http://127.0.0.1:8000")

async def get_place(client, lat, lon, pid, start, end):
    r = await client.get(f"{API}/api/evi2", params={"lat":lat, "lon":lon, "start":start, "end":end, "place_id":pid})
    r.raise_for_status()
    return r.json()

async def main():
    here = Path(__file__).resolve().parent
    places = json.loads((here/"places_10.json").read_text())
    end = dt.date.today()
    start = end - dt.timedelta(days=365)
    out = {"generated_on": dt.datetime.utcnow().isoformat(), "places": []}
    async with httpx.AsyncClient(timeout=300.0) as client:
        for p in places:
            print("Fetching", p["id"])
            j = await get_place(client, p["lat"], p["lon"], p["id"], start.isoformat(), end.isoformat())
            j["id"] = p["id"]; j["name"] = p["name"]; j["lat"] = p["lat"]; j["lon"] = p["lon"]
            j["species"] = p.get("species", []); j["npn"] = bool(p.get("npn", False))
            vals = [s["evi2"] for s in j["series"]]
            j["highest_bloom"] = float(max(vals)) if vals else 0.0
            j["lowest_bloom"] = float(min(vals)) if vals else 0.0
            j["bloom_range"] = round(j["highest_bloom"] - j["lowest_bloom"], 4)
            out["places"].append(j)
    out_path = (here.parent / "frontend" / "phenology_10_places.json")
    out_path.write_text(json.dumps(out, indent=2))
    print("Wrote", out_path)

if __name__ == "__main__":
    asyncio.run(main())
