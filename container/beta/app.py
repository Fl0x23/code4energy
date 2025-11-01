from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
import json
import csv
import os
from pathlib import Path

# Root-Prefix (für Reverse Proxy), z. B. "/beta"; per ENV ROOT_PATH anpassbar
ROOT_PATH = os.getenv("ROOT_PATH", "/beta")

APP_INFO = {
    "app": os.getenv("APP_NAME", "Code 4 Energy Beta API"),
    "description": os.getenv("APP_DESCRIPTION", "Dient als Vorlage"),
    "version": os.getenv("APP_VERSION", "1.0.0"),
    "root_path": ROOT_PATH,
    "docs_url": f"{ROOT_PATH}/docs",
    "redoc_url": f"{ROOT_PATH}/redoc",
    "openapi_url": f"{ROOT_PATH}/openapi.json",
}

app = FastAPI(
    title=APP_INFO["app"],
    description=APP_INFO["description"],
    version=APP_INFO["version"],
    # Wichtig hinter Reverse Proxy: stellt sicher, dass Routen und OpenAPI unter /beta funktionieren
    root_path=ROOT_PATH,
)

@app.get("/info")
def info():
    return APP_INFO


@app.get("/forecast")
def forecast_demo():
    """Liest eine Demo-CSV aus /data/price_forecast_demo.csv und gibt sie als JSON-Liste zurück.

    Erwartetes CSV-Format (Header): start,end,price,price_origin
    Zeiten als ISO-Strings, price in EUR/kWh.
    """
    csv_path = "/data/price_forecast_demo.csv"
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for r in reader:
                # Preis als float, rest als String belassen
                try:
                    r["price"] = float(r.get("price")) if r.get("price") not in (None, "") else None
                except Exception:
                    pass
                rows.append({
                    "start": r.get("start"),
                    "end": r.get("end"),
                    "price": r.get("price"),
                    "price_origin": r.get("price_origin") or "forecast",
                })
        return JSONResponse(content=rows)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="price_forecast_demo.csv nicht gefunden")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV konnte nicht gelesen werden: {e}")