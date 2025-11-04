from __future__ import annotations

from typing import Optional, List, Dict, Any
import os
import datetime

import pandas as pd

import fetcher
import model as csv_model
from core import tz


def run_etl_ml_job() -> None:
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


def compute_trend_stats() -> Dict[str, Any]:
    """Kapselt die Trend-Hitrate-Berechnung mit Default-Rückgabe."""
    result = csv_model.compute_trend_hit_rate()
    if not result or result.get("total", 0) == 0:
        return {"hits": 0, "total": 0, "hit_rate_percent": 0.0, "message": "Keine Vergleichsdaten vorhanden"}
    return result


def compute_deviation_stats() -> Dict[str, Any]:
    """Kapselt die MAE-Berechnung mit Default-Rückgabe."""
    result = csv_model.compute_average_deviation()
    if not result or result.get("count", 0) == 0:
        return {"avg_deviation_ct_per_kwh": 0.0, "count": 0, "message": "Keine Vergleichsdaten vorhanden"}
    return result


def _parse_dt(value: Optional[str]) -> Optional[pd.Timestamp]:
    """Flexible Zeit-Parse-Funktion. Liefert UTC-Timestamps zurück oder None."""
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


def build_forecast_items(
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: bool = False,
    days: int = 1,
    previous_forecast: bool = False,
) -> List[Dict[str, Any]]:
    """Erstellt die Forecast-/Markt-Intervallliste anhand der CSVs.

    Hebt die gesamte Geschäftslogik aus der Route in eine Service-Funktion.
    Gibt eine Liste von Items mit lokalen ISO-Zeitstempeln zurück.
    Kann Exceptions werfen (ValueError/RuntimeError), die der Aufrufer
    in HTTP-Fehler mappen kann.
    """
    # CSV-Pfade ermitteln
    forecast_csv = "/data/price_forecast.csv"
    price_path = csv_model._latest_csv("/data/entsoe_prices_*.csv")  # type: ignore[attr-defined]

    # Forecast CSV muss existieren, sonst ggf. nur Market (wenn angefordert)
    has_forecast = os.path.exists(forecast_csv)
    has_market = price_path is not None and price_path.exists()

    if not has_forecast and not (market and has_market):
        raise RuntimeError("Keine Daten gefunden (Forecast/Market)")

    # Marktdaten global laden (nur einmal), um u. a. das Standard-Startdatum
    # relativ zum letzten verfügbaren Marktzeitpunkt zu bestimmen
    s_price_all = pd.Series(dtype=float)
    last_market_ts = None
    if has_market:
        try:
            s_price_all = csv_model._read_series(price_path)  # type: ignore[attr-defined]
            if not s_price_all.empty:
                last_market_ts = s_price_all.index.max()
        except Exception as e:
            # Marktdaten sind optional – Fehler nicht fatal, aber melden
            fetcher.job_log(f"⚠️ Fehler beim Lesen der Markt-CSV (früh): {e}")
            s_price_all = pd.Series(dtype=float)
            last_market_ts = None

    # Start ermitteln
    if start is None:
        if last_market_ts is None or pd.isna(last_market_ts):
            raise RuntimeError("Keine Marktdaten vorhanden, um den Standard-Start zu bestimmen. Bitte 'start' angeben.")
        ref_date = last_market_ts.tz_convert(tz).date()
        try:
            days_back = max(0, int(days))
        except Exception:
            days_back = 1
        ref_day = ref_date - datetime.timedelta(days=days_back)
        start_local = datetime.datetime.combine(ref_day, datetime.time(0, 0, tzinfo=tz))
        start_utc = pd.Timestamp(start_local).tz_convert("UTC")
    else:
        start_utc = _parse_dt(start)

    # Forecast-Daten einlesen (falls vorhanden)
    s_fore = pd.Series(dtype=float)
    fore_last = None
    if has_forecast:
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
        raise ValueError("Ungültiger Start-Parameter")
    if end_utc is None:
        raise ValueError("Ungültiger End-Parameter")

    # Sicherstellen: start <= end
    if start_utc > end_utc:
        start_utc, end_utc = end_utc, start_utc

    # Filter Forecast auf Zeitfenster
    s_fore = s_fore.loc[(s_fore.index >= start_utc) & (s_fore.index <= end_utc)] if not s_fore.empty else s_fore

    # Marktdaten schneiden/aufbereiten
    # - Für die Antwort nur schneiden, wenn market=true
    # - Für previous_forecast-Filter wurde der globale letzte Marktzeitpunkt bereits oben ermittelt
    s_price_window = pd.Series(dtype=float)
    if has_market:
        try:
            if s_price_all.empty:
                s_price_all = csv_model._read_series(price_path)  # type: ignore[attr-defined]
            if market and not s_price_all.empty:
                s_price_window = s_price_all.loc[(s_price_all.index >= start_utc) & (s_price_all.index <= end_utc)]
        except Exception as e:
            # Marktdaten sind optional – Fehler nicht fatal, aber melden
            fetcher.job_log(f"⚠️ Fehler beim Lesen/Schneiden der Markt-CSV: {e}")
            s_price_window = pd.Series(dtype=float)

    # Optional: Forecasts vor letztem (globalen) Marktzeitpunkt ausblenden
    if not previous_forecast and not s_fore.empty:
        if not s_price_all.empty:
            if last_market_ts is not None and pd.notna(last_market_ts):
                s_fore = s_fore.loc[s_fore.index > last_market_ts]

    items: List[Dict[str, Any]] = []

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
    return items
