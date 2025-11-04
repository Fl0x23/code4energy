from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from core import APP_INFO
from services import read_forecast_demo

router = APIRouter()


@router.get("/info", tags=["Informationen"])
def info():
    return APP_INFO


@router.get("/forecast", tags=["Vorhersage"])
def forecast_demo():
    """Liest eine Demo-CSV aus /data/price_forecast_demo.csv und gibt sie als JSON-Liste zurück.

    Erwartetes CSV-Format (Header): start,end,price,price_origin
    Zeiten als ISO-Strings, price in EUR/kWh.
    """
    try:
        rows = read_forecast_demo()
        return JSONResponse(content=rows)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="price_forecast_demo.csv nicht gefunden")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV konnte nicht gelesen werden: {e}")
