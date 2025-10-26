from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
import json
import csv
import os
from pathlib import Path

APP_INFO = {
    "app": "Minimal beta API",
    "description": "Eine extrem einfache FastAPI",
    "version": "1.0.0",
}

app = FastAPI(
    title=APP_INFO["app"],
    description=APP_INFO["description"],
    version=APP_INFO["version"],
)

@app.get("/info")
def info():
    return APP_INFO

@app.get("/")
def read_root():
    """Weiterleitung auf /info (relativ, damit Nginx-Prefix erhalten bleibt)."""
    return RedirectResponse(url="info", status_code=307)

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