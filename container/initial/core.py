from __future__ import annotations

import os
import datetime
import logging
from contextlib import asynccontextmanager

from tzlocal import get_localzone
from apscheduler.schedulers.background import BackgroundScheduler

# Einheitliche Zeitzone aus dem System (z. B. via TZ=Europe/Berlin in Docker)
# Achtung: tzlocal liefert eine ZoneInfo-kompatible TZ

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


def _ensure_file_logging() -> None:
    """Richtet einmalig File-Logging für Uvicorn-Logger ein."""
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


@asynccontextmanager
async def lifespan(app):  # FastAPI wird den Typ selbst erkennen
    """Lifecycle-Manager für App-Start und -Stopp.

    - Initialisiert und startet den Scheduler
    - Plant einen Intervall-Job (alle 2 Stunden)
    - Optional einmaliger Startlauf bei RUN_JOB_ON_START
    - Stellt sicher, dass der Scheduler beim Beenden gestoppt wird
    """
    logger = logging.getLogger("uvicorn.error")
    _ensure_file_logging()
    logger.info("[App] 🚀 FastAPI gestartet – Scheduler initialisieren...")

    # Lazy-Import, um zirkuläre Abhängigkeiten zu vermeiden
    from services import run_etl_ml_job

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
