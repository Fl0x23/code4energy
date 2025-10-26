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

# Einheitliche Zeitzone aus dem System (z. B. via TZ=Europe/Berlin in Docker)
tz = get_localzone()
scheduler = BackgroundScheduler(timezone=tz)

# Log-Datei (einfache lokale Datei im Container)
LOG_FILE = "app.log"

# Basis-App-Infos und FastAPI-Metadaten (per ENV überschreibbar)
APP_INFO = {
    "app": os.getenv("APP_NAME", "code4energy initial API"),
    "description": os.getenv("APP_DESCRIPTION", "CSV-basierte FastAPI für ENTSO-E Daten und Preis-Forecasts"),
    "version": os.getenv("APP_VERSION", "1.0.0"),
}

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
)


@app.get("/")
def read_root():
    """Weiterleitung auf /info (relativ, damit Nginx-Prefix erhalten bleibt)."""
    return RedirectResponse(url="info", status_code=307)


@app.get("/info")
def info():
    """Metadaten-Endpoint der Anwendung."""
    return APP_INFO


@app.get("/forecast")
def get_forecast():
    """Liest die feste Forecast-CSV (`/data/price_forecast.csv`) und gibt sie
    als JSON-Zeitfenster zurück.

    Verarbeitungsschritte:
    - Forecast-CSV laden
    - Zeitstempel in lokale Zeitzone konvertieren
    - Schrittweite aus Median der Deltas bestimmen
    - Einträge in ein homogenes Intervallformat [start, end) transformieren
    - Marktpreise der letzten 2 Tage ergänzen
    """
    fixed_path = "/data/price_forecast.csv"
    if not os.path.exists(fixed_path):
        raise HTTPException(status_code=404, detail="Keine Forecast-CSV gefunden.")
    latest = fixed_path

    try:
        df = pd.read_csv(latest, parse_dates=["time"])  # erwartet Spalten: time, predicted_price_eur_mwh, ...
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV konnte nicht gelesen werden: {e}")

    if df.empty or "time" not in df.columns or "predicted_price_eur_mwh" not in df.columns:
        raise HTTPException(status_code=500, detail="CSV hat unerwartetes Format oder ist leer.")

    # Zeitspalte nach UTC interpretieren (falls naiv) und nach Europe/Berlin konvertieren
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    df["time_local"] = df["time"].dt.tz_convert(ZoneInfo("Europe/Berlin"))

    # Schrittweite bestimmen (Median der Deltas)
    deltas = df.sort_values("time_local")["time_local"].diff().dropna()
    step = deltas.median() if not deltas.empty else pd.Timedelta("1h")

    data_items = []
    df_sorted = df.sort_values("time_local").reset_index(drop=True)
    for _, row in df_sorted.iterrows():
        start = row["time_local"]
        end = start + step
        price_kwh = float(row["predicted_price_eur_mwh"]) / 1000.0
        # Variante aus CSV ableiten: wenn exogene Spalten vorhanden und nicht NaN, dann with_exo, sonst univariate
        has_exo_cols = all(col in df.columns for col in ["solar_input_mw", "wind_onshore_input_mw", "wind_offshore_input_mw"])
        if has_exo_cols:
            exo_present = (
                not pd.isna(row.get("solar_input_mw"))
                or not pd.isna(row.get("wind_onshore_input_mw"))
                or not pd.isna(row.get("wind_offshore_input_mw"))
            )
            variant = "with_exo" if exo_present else "univariate"
        else:
            variant = "univariate"
        data_items.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "price": round(price_kwh, 5),
            "price_origin": "forecast",
            "variant": variant,
        })

    # Marktpreise der letzten 2 Tage aus entsoe_prices_*.csv lesen (bis zum letzten verfügbaren Marktzeitpunkt)
    market_pattern = "/data/entsoe_prices_*.csv"
    market_files = glob.glob(market_pattern)
    market_items = []
    if market_files:
        latest_market = max(market_files, key=lambda f: os.path.getmtime(f))
        try:
            # Robust einlesen: 2 Spalten (Zeit, Preis) ohne Header oder mit Header
            try:
                dfm = pd.read_csv(latest_market)
                # Spalten bestimmen
                if "time" in dfm.columns:
                    tcol = "time"
                else:
                    tcol = dfm.columns[0]
                # Wertspalte
                vcols = [c for c in dfm.columns if c != tcol]
                vcol = vcols[0] if vcols else dfm.columns[1]
            except Exception:
                dfm = pd.read_csv(latest_market, header=None, names=["time", "value"])  # Fallback
                tcol, vcol = "time", "value"

            # Zeiten konvertieren
            dfm[tcol] = pd.to_datetime(dfm[tcol], utc=True, errors="coerce")
            dfm = dfm.dropna(subset=[tcol])
            # letzte 2 Tage bis zum letzten verfügbaren Marktzeitpunkt filtern
            if not dfm.empty:
                last_market_utc = pd.to_datetime(dfm[tcol].max()).tz_convert("UTC")
                start_utc = last_market_utc - pd.Timedelta(days=2)
                dfm = dfm[(dfm[tcol] >= start_utc) & (dfm[tcol] <= last_market_utc)]
            if not dfm.empty:
                # lokale Zeit und Schrittweite
                dfm["time_local"] = dfm[tcol].dt.tz_convert(ZoneInfo("Europe/Berlin"))
                deltas_m = dfm.sort_values("time_local")["time_local"].diff().dropna()
                step_m = deltas_m.median() if not deltas_m.empty else pd.Timedelta("1h")
                dfm = dfm.sort_values("time_local").reset_index(drop=True)
                for _, r in dfm.iterrows():
                    s = r["time_local"]
                    e = s + step_m
                    price_eur_mwh = float(r[vcol])
                    market_items.append({
                        "start": s.isoformat(),
                        "end": e.isoformat(),
                        "price": round(price_eur_mwh / 1000.0, 5),
                        "price_origin": "market",
                    })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Market-CSV konnte nicht gelesen werden: {e}")

    # Kombinierte Liste (zuerst Markt, dann Forecast), nach Startzeit sortiert
    combined = market_items + data_items
    combined.sort(key=lambda x: x["start"])

    # Lizenz-/Quellenhinweis als Header ergänzen, Body bleibt das bekannte Array
    headers = {
        "X-Data-Source": "ENTSO-E Transparency Platform",
        "X-Data-URL": "https://transparency.entsoe.eu",
        "X-Usage-Notice": "Nutzung gemäß den Nutzungsbedingungen der Plattform",
    }
    return JSONResponse(content=combined, headers=headers)



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
