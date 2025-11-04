from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from core import APP_INFO
from services import (
    compute_trend_stats,
    compute_deviation_stats,
    build_forecast_items,
    get_price_forecast_last_modified,
)

router = APIRouter()


@router.get("/info", tags=["Informationen"])
def info():
    """Metadaten-Endpoint der Anwendung."""
    return APP_INFO


@router.get("/trend", tags=["Vorhersage"])
def get_trend():
    """Gibt die Trend-Trefferquote (Richtungs-Hitrate) zwischen Marktpreis und Forecast zurück.

    Berechnung siehe csv_model.compute_trend_hit_rate().
    """
    result = compute_trend_stats()
    return JSONResponse(content=result)


@router.get("/deviation", tags=["Vorhersage"])
def get_deviation():
    """Gibt die durchschnittliche Abweichung (MAE) in ct/kWh zwischen Marktpreis und Forecast zurück.

    Berechnung siehe csv_model.compute_average_deviation(). Niedrigere Werte
    bedeuten präzisere Vorhersagen.
    """
    result = compute_deviation_stats()
    return JSONResponse(content=result)


@router.get("/forecast", tags=["Vorhersage"])
def get_forecast(
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: bool = False,
    days: int = 1,
    previous_forecast: bool = False,
):
    """Liefert Forecast-Daten (und optional Marktdaten) als JSON-Intervalle.

        Query-Parameter:
        - start: ISO-Zeitpunkt (interpretiert als lokale Zeit, falls ohne TZ).
                 Standard: 00:00 lokale Zeit des Tages des letzten verfügbaren
                 Marktwerts, minus "days" Tage. Sind keine Marktdaten vorhanden
                 und "start" fehlt, wird 404 zurückgegeben (bitte "start" angeben).
        - end:   ISO-Zeitpunkt (interpretiert als lokale Zeit, falls ohne TZ).
                 Standard: letztes Zeitfenster der Forecast-CSV (falls vorhanden), sonst jetzt.
        - market: Wenn true, werden zusätzlich vorhandene Marktdaten aus entsoe_prices_*.csv ausgegeben.
        - days:  Anzahl Tage zurück für den Standard-Start (wenn "start" nicht gesetzt ist).
                 Default: 1 (bezieht sich auf den Tag des letzten Marktwerts).
        - previous_forecast: Wenn false, werden Forecast-Werte, deren Zeitstempel
            nicht nach dem global letzten Marktzeitpunkt liegen, ausgeblendet
            (Filter: t > letzter globaler Marktzeitpunkt). Default: false.

        Ausgabe (Zeitstempel in lokaler System-Zeitzone): Liste von Objekten
            {
                "start": ISO-LOCAL,         # inkl. Millisekunden und Offset
                "end": ISO-LOCAL,           # = start + Schrittweite (z. B. 15min)
                "price": float,             # EUR/kWh (5 Nachkommastellen)
                "price_origin": "forecast" | "market"
            }

        Beispiele:
        - GET /forecast
        - GET /forecast?market=true
        - GET /forecast?start=2025-10-28&end=2025-10-29
        - GET /forecast?start=2025-10-28T00:00:00+01:00&end=2025-10-28T12:00:00+01:00&market=true
        - GET /forecast?start=2025-10-28T00:00&market=true
        """
    try:
        items = build_forecast_items(start=start, end=end, market=market, days=days, previous_forecast=previous_forecast)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Not found / fehlende Daten
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Erstellen der Forecast-Daten: {e}")

    # Meta: Letzte Änderung der Forecast-CSV als Header (UTC, ISO)
    headers: dict[str, str] = {}
    ts_iso = get_price_forecast_last_modified()
    if ts_iso:
        headers["X-Price-Forecast-Last-Modified"] = ts_iso

    return JSONResponse(content=items, headers=headers)
