"""FastAPI-Anwendung für Datenbeschaffung (ENTSO-E), Training und Bereitstellung
von Preis-Forecasts.

Kernkomponenten:
- Zeitgesteuerter Hintergrundjob (APScheduler):
    1) ENTSO-E-Daten als CSV speichert
    2) Diagnose-Logs aus CSVs schreibt (letzte Timestamps, Trainingsspannen)
    3) Ein CSV-basiertes Modell trainiert und Forecast-CSV erzeugt
- HTTP-Endpoints:
    - "/"     Gesundheits-/Status-Endpoint
    - "/forecast" Liefert die jüngste Forecast-CSV als vereinheitlichte JSON-Zeitfenster

Hinweis: Die Pfade zu CSVs zeigen standardmäßig auf "/data" im Container.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from apscheduler.schedulers.background import BackgroundScheduler
from tzlocal import get_localzone
from contextlib import asynccontextmanager
import datetime
import logging
import os
import glob
import pandas as pd
from zoneinfo import ZoneInfo
import fetcher
import model as csv_model
from typing import Optional

# Einheitliche Zeitzone aus dem System (z. B. via TZ=Europe/Berlin in Docker)
tz = get_localzone()
scheduler = BackgroundScheduler(timezone=tz)

# Log-Datei (einfache lokale Datei im Container)
LOG_FILE = "app.log"

# Root-Prefix (für Reverse Proxy), z. B. "/initial"; per ENV ROOT_PATH anpassbar
ROOT_PATH = os.getenv("ROOT_PATH", "/initial")

# Basis-App-Infos und FastAPI-Metadaten (per ENV überschreibbar)
APP_INFO = {
    "app": os.getenv("APP_NAME", "Code 4 Energy Initial API"),
    "description": os.getenv("APP_DESCRIPTION", "CSV-basierte FastAPI für ENTSO-E Daten und Preis-Forecasts"),
    "version": os.getenv("APP_VERSION", "1.0.0"),
    "root_path": ROOT_PATH,
    "docs_url": f"{ROOT_PATH}/docs",
    "redoc_url": f"{ROOT_PATH}/redoc",
    "openapi_url": f"{ROOT_PATH}/openapi.json",
}

# Hinweis: App-Instanz nach Definition von `lifespan` erzeugen


def run_etl_ml_job():
    """Geplanter ETL/ML-Job (Intervall).

    Ablauf:
    - Lädt ENTSO-E-Preise sowie Solar/Wind-Forecasts (Speicherung: CSV)
    - Loggt Diagnoseinfos aus CSV (letzte Zeitpunkte, Trainingsspannen)
    - Trainiert CSV-basiertes Ridge-Modell und erzeugt Forecast-CSV
    """
    now = datetime.datetime.now(tz=tz)
    fetcher.job_log(f"🚀 [{now}] Starte geplanten Job (alle 2h)...")
    
    # Zeitraum 
    # Für Demo/Tests ist der Zeitraum fix; produktiv ggf. rollierend/parametrisierbar
    start_dt = datetime.datetime(2025, 10, 7, tzinfo=tz)
    end_dt = datetime.datetime(2026, 1, 1, tzinfo=tz)
    
    # Preise
    try:
        fetcher.fetch_entsoe_prices(start=start_dt, end=end_dt)
    except Exception as e:
        fetcher.job_log(f"❌ Fehler beim Laden der Preise: {e}")
        
    # Wind & Solar Forecasts
    try:
        fetcher.fetch_entsoe_wind_and_solar_forecast(start=start_dt, end=end_dt)
    except Exception as e:
        fetcher.job_log(f"❌ Fehler beim Laden der Wind/Solar-Forecasts: {e}")

    # Preis-Forecast (Modell 2: CSV-basiert – nur Diagnoseausgabe der letzten Timestamps)
    try:
        csv_model.run_csv_based_forecast()
    except Exception as e:
        fetcher.job_log(f"❌ Fehler im CSV-Modell: {e}")

    # Trainingsspannen aus CSV loggen (Solar, Wind On/Offshore)
    try:
        csv_model.log_training_spans_from_csv()
    except Exception as e:
        fetcher.job_log(f"❌ Fehler beim Loggen der Trainingsspannen: {e}")

    # CSV-basiertes Training und Forecast
    try:
        csv_model.run_csv_training_and_forecast(n_lags=16)
    except Exception as e:
        fetcher.job_log(f"❌ Fehler beim CSV-Training/Forecast: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle-Manager für App-Start und -Stopp.

    - Initialisiert und startet den Scheduler
    - Plant einen Intervall-Job (alle 2 Stunden)
    - Stellt sicher, dass der Scheduler beim Beenden gestoppt wird
    """
    # Nutze den Uvicorn-Error-Logger, damit App-Logs im gleichen Format erscheinen wie Uvicorn-Logs
    logger = logging.getLogger("uvicorn.error")
    # File-Logging einrichten (einmalig)
    def _ensure_file_logging():
        try:
            abs_log = os.path.abspath(LOG_FILE)
            formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            for name in ("uvicorn.error", "uvicorn.access", ""):
                lg = logging.getLogger(name)
                has_file = any(
                    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_log
                    for h in lg.handlers
                )
                if not has_file:
                    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
                    fh.setLevel(logging.INFO)
                    fh.setFormatter(formatter)
                    lg.addHandler(fh)
        except Exception:
            # Fallback: kein Crash, Logging nur auf Stdout
            pass

    _ensure_file_logging()
    logger.info("[App] 🚀 FastAPI gestartet – Scheduler initialisieren...")
    
    # Intervall-Job alle 2 Stunden
    scheduler.add_job(
        run_etl_ml_job,
        "interval",
        hours=2,
        id="every_2h",
        replace_existing=True,
    )
    # Optionalen Sofortlauf beim Start planen, wenn per ENV aktiviert
    # Compose/ENV: RUN_JOB_ON_START=true|1|yes
    run_on_start = os.getenv("RUN_JOB_ON_START", "false").strip().lower() in ("1", "true", "yes")
    if run_on_start:
        try:
            scheduler.add_job(
                run_etl_ml_job,
                trigger="date",
                run_date=datetime.datetime.now(tz=tz),
                id="startup_once",
                replace_existing=True,
            )
            logger.info("[App] ▶️ RUN_JOB_ON_START aktiv – einmaligen Startlauf eingeplant.")
        except Exception as e:
            logger.error(f"[App] Fehler beim Einplanen des Startlaufs: {e}")
    if not scheduler.running:
        scheduler.start()
    logger.info("[App] 🚀 Scheduler gestartet – Job läuft alle 2 Stunden.")
    try:
        yield
    finally:
        logger.info("[App] FastAPI wird beendet – Scheduler stoppen...")
        scheduler.shutdown()


app = FastAPI(
    title=APP_INFO["app"],
    description=APP_INFO["description"],
    version=APP_INFO["version"],
    lifespan=lifespan,
    # Wichtig hinter Reverse Proxy: stellt sicher, dass Routen und OpenAPI unter /initial funktionieren
    root_path=ROOT_PATH,
)


@app.get("/info")
def info():
    """Metadaten-Endpoint der Anwendung."""
    return APP_INFO





@app.get("/trend")
def get_trend():
    """Gibt die Trend-Trefferquote (Richtungs-Hitrate) zwischen Marktpreis und Forecast zurück.

    Berechnung siehe csv_model.compute_trend_hit_rate().
    """
    result = csv_model.compute_trend_hit_rate()
    if not result or result.get("total", 0) == 0:
        # Keine ausreichenden Daten vorhanden
        return JSONResponse(
            content={"hits": 0, "total": 0, "hit_rate_percent": 0.0, "message": "Keine Vergleichsdaten vorhanden"},
            status_code=200,
        )
    return JSONResponse(content=result)


@app.get("/deviation")
def get_deviation():
    """Gibt die durchschnittliche Abweichung (MAE) in ct/kWh zwischen Marktpreis und Forecast zurück.

    Berechnung siehe csv_model.compute_average_deviation(). Niedrigere Werte
    bedeuten präzisere Vorhersagen.
    """
    result = csv_model.compute_average_deviation()
    if not result or result.get("count", 0) == 0:
        return JSONResponse(
            content={
                "avg_deviation_ct_per_kwh": 0.0,
                "count": 0,
                "message": "Keine Vergleichsdaten vorhanden",
            },
            status_code=200,
        )
    return JSONResponse(content=result)


@app.get("/forecast")
def get_forecast(
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: bool = False,
    days: int = 1,
    previous_forecast: bool = False,
):
    """Liefert Forecast-Daten (und optional Marktdaten) als JSON-Intervalle.

        Query-Parameter:
    - start: ISO-Zeitpunkt (interpretiert als lokale Zeit, falls ohne TZ). Standard: Gestern 00:00 lokale Zeit.
        - end:   ISO-Zeitpunkt (interpretiert als lokale Zeit, falls ohne TZ). Standard: letztes Zeitfenster der Forecast-CSV.
    - market: Wenn true, werden zusätzlich vorhandene Marktdaten aus entsoe_prices_*.csv ausgegeben.
    - days:  Anzahl Tage zurück für den Standard-Start (wenn "start" nicht gesetzt ist). Default: 1 ("gestern").
        - previous_forecast: Wenn false, werden Forecast-Werte vor dem Ende der Marktdaten
            im gewählten Zeitraum nicht geliefert (Filter auf t >= letztem Marktzeitpunkt).
            Default: false.

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
    # CSV-Pfade ermitteln
    forecast_csv = "/data/price_forecast.csv"
    price_path = csv_model._latest_csv("/data/entsoe_prices_*.csv")  # type: ignore[attr-defined]

    # Forecast CSV muss existieren, sonst ggf. nur Market (wenn angefordert)
    has_forecast = os.path.exists(forecast_csv)
    has_market = price_path is not None and price_path.exists()

    if not has_forecast and not (market and has_market):
        raise HTTPException(status_code=404, detail="Keine Daten gefunden (Forecast/Market)")

    # Helper: flexible Zeit-Parse-Funktion
    def _parse_dt(value: Optional[str]) -> Optional[pd.Timestamp]:
        if not value:
            return None
        try:
            ts = pd.to_datetime(value, errors="coerce")
            if pd.isna(ts):
                return None
            # Naive Zeiten als lokale Zeit interpretieren
            if ts.tzinfo is None:
                ts = ts.tz_localize(tz)
            else:
                # In System-TZ konvertieren für Konsistenz, danach UTC
                ts = ts.tz_convert(tz)
            return ts.tz_convert("UTC")
        except Exception:
            return None

    # Standard-Start: Vorgestern 00:00 (lokal)
    if start is None:
        today_local = datetime.datetime.now(tz=tz).date()
        try:
            days_back = max(0, int(days))
        except Exception:
            days_back = 1
        vorgestern = today_local - datetime.timedelta(days=days_back)
        start_local = datetime.datetime.combine(vorgestern, datetime.time(0, 0, tzinfo=tz))
        start_utc = pd.Timestamp(start_local).tz_convert("UTC")
    else:
        start_utc = _parse_dt(start)

    # Forecast-Daten einlesen (falls vorhanden)
    s_fore = pd.Series(dtype=float)
    fore_last = None
    if has_forecast:
        try:
            df_f = pd.read_csv(
                forecast_csv,
                usecols=["time", "predicted_price_eur_mwh"],
            )
            df_f["time"] = pd.to_datetime(df_f["time"], utc=True, errors="coerce")
            df_f = df_f.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
            s_fore = pd.Series(
                pd.to_numeric(df_f["predicted_price_eur_mwh"], errors="coerce").values,
                index=df_f["time"],
            ).dropna()
            s_fore = s_fore[~s_fore.index.duplicated(keep="last")].sort_index()
            fore_last = s_fore.index.max() if not s_fore.empty else None
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Fehler beim Lesen der Forecast-CSV: {e}")

    # Standard-Ende: letztes Zeitfenster der Forecast-CSV (falls vorhanden), sonst jetzt
    if end is None:
        if fore_last is not None:
            end_utc = fore_last
        else:
            end_utc = pd.Timestamp(datetime.datetime.now(datetime.timezone.utc))
    else:
        end_utc = _parse_dt(end)

    # Fallbacks, falls Parse fehlschlug
    if start_utc is None:
        raise HTTPException(status_code=400, detail="Ungültiger Start-Parameter")
    if end_utc is None:
        raise HTTPException(status_code=400, detail="Ungültiger End-Parameter")

    # Sicherstellen: start <= end
    if start_utc > end_utc:
        start_utc, end_utc = end_utc, start_utc

    # Filter Forecast auf Zeitfenster
    s_fore = s_fore.loc[(s_fore.index >= start_utc) & (s_fore.index <= end_utc)] if not s_fore.empty else s_fore

    # Marktdaten laden
    # - Für die Antwort nur schneiden, wenn market=true
    # - Für previous_forecast-Filter den global letzten Marktzeitpunkt verwenden (unabhängig vom Window)
    s_price_all = pd.Series(dtype=float)
    s_price_window = pd.Series(dtype=float)
    if has_market:
        try:
            s_price_all = csv_model._read_series(price_path)  # type: ignore[attr-defined]
            if market:
                s_price_window = s_price_all.loc[(s_price_all.index >= start_utc) & (s_price_all.index <= end_utc)]
        except Exception as e:
            # Marktdaten sind optional – Fehler nicht fatal, aber melden
            fetcher.job_log(f"⚠️ Fehler beim Lesen der Markt-CSV: {e}")
            s_price_all = pd.Series(dtype=float)
            s_price_window = pd.Series(dtype=float)

    # Optional: Forecasts vor letztem (globalen) Marktzeitpunkt ausblenden
    if not previous_forecast and not s_fore.empty:
        if not s_price_all.empty:
            last_market_ts = s_price_all.index.max()
            if pd.notna(last_market_ts):
                s_fore = s_fore.loc[s_fore.index > last_market_ts]

    # Antwortdaten bauen (Intervall-basierte Ausgabe in lokaler Zeit)
    def _infer_step(idx: pd.DatetimeIndex) -> pd.Timedelta:
        if idx.size >= 2:
            diffs = pd.Series(idx).diff().dropna()
            try:
                td = diffs.min()
                if pd.isna(td) or td <= pd.Timedelta(0):
                    return pd.Timedelta(minutes=15)
                return td
            except Exception:
                return pd.Timedelta(minutes=15)
        return pd.Timedelta(minutes=15)

    items = []

    # Forecast-Einträge
    if not s_fore.empty:
        step_fore = _infer_step(s_fore.index)
        for t, v in s_fore.items():
            start_local = t.tz_convert(tz).to_pydatetime().isoformat(timespec="milliseconds")
            end_local = (t + step_fore).tz_convert(tz).to_pydatetime().isoformat(timespec="milliseconds")
            items.append({
                "start": start_local,
                "end": end_local,
                "price": round(float(v) / 1000.0, 5),  # EUR/kWh
                "price_origin": "forecast",
            })

    # Markt-Einträge optional
    if market and not s_price_window.empty:
        step_mkt = _infer_step(s_price_window.index)
        for t, v in s_price_window.items():
            start_local = t.tz_convert(tz).to_pydatetime().isoformat(timespec="milliseconds")
            end_local = (t + step_mkt).tz_convert(tz).to_pydatetime().isoformat(timespec="milliseconds")
            items.append({
                "start": start_local,
                "end": end_local,
                "price": round(float(v) / 1000.0, 5),  # EUR/kWh
                "price_origin": "market",
            })

    # nach Startzeit sortieren
    items.sort(key=lambda x: x["start"])

    return JSONResponse(content=items)
