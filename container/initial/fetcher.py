"""Fetcher-Modul: Beschafft ENTSO-E-Daten und speichert CSVs.

Zusätzlich: Hilfsfunktionen fürs Logging und robuste Zeitstempelverarbeitung.

Erwartete Umgebungsvariablen:
- ENTSOE_API_KEY       API-Schlüssel für ENTSO-E
"""

from entsoe import EntsoePandasClient
import pandas as pd
import datetime
import os
import socket
import logging


def job_log(message: str) -> None:
    """Konsistente Job-Ausgabe mit Präfix über den Uvicorn-Logger.

    Verwendung: Statt print() direkt in den Uvicorn-Error-Logger schreiben,
    um ein einheitliches Logformat im Container zu erhalten.
    """
    logger = logging.getLogger("uvicorn.error")
    logger.info(f"[Job] {message}")


def check_internet_connection(host="web-api.tp.entsoe.eu", port=443, timeout=5):
    """Prüft eine TCP-Verbindung zum angegebenen Host/Port.

    Liefert True bei Erfolg (und loggt), sonst False (und loggt Fehler).
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            job_log(f"🌐 Verbindung zu {host}:{port} erfolgreich.")
        return True
    except OSError:
        job_log(f"❌ Keine Verbindung zu {host}:{port}.")
        return False


def _to_series(obj) -> pd.Series:
    """Hilfsfunktion: DataFrame/Series zu Series vereinheitlichen.

    - DataFrame mit einer Spalte wird zur Series "gesqueezed"
    - Bei mehrspaltigem DataFrame wird die erste Spalte genutzt
    """
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.squeeze(axis=1)
        return obj.iloc[:, 0]
    raise TypeError(f"Unerwarteter Rückgabetyp: {type(obj)}")


def _ensure_timestamp(value, tz: str = "Europe/Berlin") -> pd.Timestamp:
    """Konvertiert value robust zu pandas.Timestamp mit gewünschter Zeitzone.

    Akzeptiert: None, str, datetime/date, pandas.Timestamp.
    Falls das Eingabeobjekt keine TZ hat, wird die angegebene TZ lokalisiert.
    """
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.tz_convert(tz) if value.tzinfo else value.tz_localize(tz)
    if isinstance(value, (datetime.datetime, datetime.date)):
        ts = pd.Timestamp(value)
        return ts.tz_convert(tz) if ts.tzinfo else ts.tz_localize(tz)
    # assume string
    ts = pd.Timestamp(str(value))
    return ts.tz_convert(tz) if ts.tzinfo else ts.tz_localize(tz)


def fetch_entsoe_timeseries(
    query_name: str,
    *,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    psr_type: str | None,
    csv_path: str | None,
    value_name: str = "value",
) -> int:
    """Allgemeine ENTSO-E-Fetch-Funktion (nur CSV).

    Ausführung:
    - Führt client.<query_name>(country_code, start, end, psr_type?) aus
    - Speichert zurückgegebene Zeitreihe optional als CSV

    Rückgabe: Anzahl geladener Punkte (int)
    """
    # --- Internet prüfen ---
    if not check_internet_connection():
        job_log("🚫 Keine Internetverbindung, breche Job ab.")
        return 0

    # --- API-Key ---
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise ValueError("Bitte setze die Umgebungsvariable ENTSOE_API_KEY!")

    client = EntsoePandasClient(api_key=api_key)

    # --- Abfrage durchführen ---
    job_log(f"📡 {query_name} {country_code} von {start} bis {end} (psr={psr_type}) ...")
    query_fn = getattr(client, query_name)
    try:
        if psr_type is None:
            data = query_fn(country_code, start=start, end=end)
        else:
            data = query_fn(country_code, start=start, end=end, psr_type=psr_type)
    except Exception as e:
        job_log(f"❌ Fehler bei ENTSO-E-Abfrage {query_name}: {e}")
        return 0

    series = _to_series(data)
    # Einheitliche Spalten-/Indexnamen für CSV sichern
    s = series.copy()
    try:
        s.name = value_name or "value"
        if s.index.name != "time":
            s.index.name = "time"
    except Exception:
        pass

    # --- Zeitraum der gespeicherten Daten anzeigen ---
    try:
        if series is None or len(series) == 0:
            job_log("⚠️ Leere Zeitreihe erhalten – kein Zeitraum vorhanden.")
        else:
            ts_min = series.index.min()
            ts_max = series.index.max()
            n_pts = len(series)
            job_log(f"🗓️ Speicher-Zeitraum: {ts_min} bis {ts_max} (Punkte: {n_pts})")
    except Exception as e:
        job_log(f"ℹ️ Hinweis: Zeitraum-Bestimmung fehlgeschlagen: {e}")

    # --- CSV speichern (Index konsequent als UTC, ohne Fallback) ---
    if csv_path:
        try:
            # Index robust in UTC bringen, Indexname 'time' beibehalten
            if hasattr(s, "index"):
                if getattr(s.index, "tz", None) is None:
                    msg = "Zeitstempel-Index ohne Zeitzone (naive) – erwartete tz-aware Timestamps. Abbruch."
                    job_log(f"❌ {msg}")
                    raise ValueError(msg)
                s.index = s.index.tz_convert("UTC")
                s.index.name = "time"

            s.to_csv(csv_path, header=True, index=True)
            job_log(f"✅ CSV gespeichert (UTC): {csv_path}")
        except Exception as e:
            job_log(f"⚠️ Konnte CSV nicht speichern ({csv_path}): {e}")

    # Anzahl Punkte zurückgeben
    return int(len(series)) if series is not None else 0


def fetch_entsoe_prices(*, start=None, end=None, country_code: str = "DE_LU", csv_path: str | None = None):
    """Lädt Day-Ahead-Preise und speichert als CSV.

    - Zeitfenster standardmäßig: aktuelles Jahr
    - CSV-Pfad: <CSV_BASE>/entsoe_prices_<jahr>.csv (überschreibbar über csv_path)
    """
    if start is None or end is None:
        now = datetime.datetime.now()
        year = now.year
        start_ts = pd.Timestamp(datetime.date(year, 1, 1), tz="Europe/Berlin")
        end_ts = pd.Timestamp(datetime.date(year + 1, 1, 1), tz="Europe/Berlin")
    else:
        start_ts = _ensure_timestamp(start)
        end_ts = _ensure_timestamp(end)
        year = start_ts.year

    base = os.getenv("CSV_BASE", "/data")
    csv = csv_path or f"{base}/entsoe_prices_{year}.csv"
    return fetch_entsoe_timeseries(
        "query_day_ahead_prices",
        country_code=country_code,
        start=start_ts,
        end=end_ts,
        psr_type=None,
        csv_path=csv,
        value_name="price_eur_mwh",
    )


def fetch_entsoe_wind_and_solar_forecast(*, start=None, end=None, country_code: str = "DE_LU"):
    """Lädt Solar- (B16), Wind-Onshore- (B19) und Wind-Offshore- (B18) Forecasts.

    - Speicherung jeweils als CSV unter <CSV_BASE>
    """
    base = os.getenv("CSV_BASE", "/data")
    if start is None or end is None:
        now = datetime.datetime.now()
        year = now.year
        start_ts = pd.Timestamp(datetime.date(year, 1, 1), tz="Europe/Berlin")
        end_ts = pd.Timestamp(datetime.date(year + 1, 1, 1), tz="Europe/Berlin")
    else:
        start_ts = _ensure_timestamp(start)
        end_ts = _ensure_timestamp(end)
        year = start_ts.year

    # Solar
    fetch_entsoe_timeseries(
        "query_wind_and_solar_forecast",
        country_code=country_code,
        start=start_ts,
        end=end_ts,
        psr_type="B16",
        csv_path=f"{base}/entsoe_solar_forecast_{year}.csv",
        value_name="forecast_mw",
    )

    # Wind Onshore
    fetch_entsoe_timeseries(
        "query_wind_and_solar_forecast",
        country_code=country_code,
        start=start_ts,
        end=end_ts,
        psr_type="B19",
        csv_path=f"{base}/entsoe_wind_onshore_forecast_{year}.csv",
        value_name="forecast_mw",
    )

    # Wind Offshore
    fetch_entsoe_timeseries(
        "query_wind_and_solar_forecast",
        country_code=country_code,
        start=start_ts,
        end=end_ts,
        psr_type="B18",
        csv_path=f"{base}/entsoe_wind_offshore_forecast_{year}.csv",
        value_name="forecast_mw",
    )
