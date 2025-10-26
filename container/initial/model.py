from __future__ import annotations

"""CSV-basiertes Zusatz-Modul für Diagnose/Forecast auf Basis der von fetcher
erzeugten CSV-Dateien. Es ermittelt die zuletzt vorhandenen Zeitstempel und gibt
sie im gewünschten Log-Format aus."""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from fetcher import job_log
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def _latest_csv(pattern: str) -> Optional[Path]:
    """Findet die neueste CSV-Datei, die dem Glob-Pattern entspricht.

    Parameter:
    - pattern: Glob-Pattern (z. B. "/data/entsoe_prices_*.csv").

    Rückgabe:
    - Pfad der neuesten Datei (nach Änderungszeit) oder None, falls keine Datei
      gefunden wurde. Sucht ab Root ("/") bei absoluten Mustern, sonst im CWD.
    """
    paths = list(Path("/").glob(pattern[1:]) if pattern.startswith("/") else Path.cwd().glob(pattern))
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _read_last_time(csv_path: Path) -> Optional[pd.Timestamp]:
    """Liest robust den zuletzt verfügbaren Zeitpunkt aus einer CSV.

    Erwartung: Erste Spalte der CSV enthält Zeitstempel (mit oder ohne Headername).
    Die Werte werden als UTC interpretiert; ungültige Einträge werden ignoriert.

    Parameter:
    - csv_path: Pfad zur CSV-Datei.

    Rückgabe:
    - Maximaler Zeitstempel (UTC) oder None, wenn keine gültigen Timestamps gefunden wurden.
    """
    if not csv_path or not csv_path.exists():
        return None
    # Lies immer die erste Spalte (Zeit), unabhängig von Headernamen
    df = pd.read_csv(csv_path, usecols=[0])
    ts = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce").dropna()
    return ts.max() if not ts.empty else None


def _fmt(ts: Optional[pd.Timestamp], tz_local: str) -> str:
    """Formatiert einen UTC-Zeitstempel als "UTC=... | Local=...".

    Parameter:
    - ts: UTC-Zeitstempel oder None.
    - tz_local: Name der lokalen Zeitzone (z. B. "Europe/Berlin").

    Rückgabe:
    - Formatierter String oder "n/a" bei None.
    """
    if ts is None:
        return "n/a"
    return f"UTC={ts.isoformat()} | Local={ts.tz_convert(tz_local).isoformat()}"


def run_csv_based_forecast(*, tz_local: str = "Europe/Berlin") -> None:
    """CSV-basiertes Modell: Liest die von fetcher erzeugten CSV-Dateien und
    gibt die letzten Timestamps je Reihe im gewünschten Format aus.

    Es werden die jeweils neuesten Dateien für Preise, Solar-, Wind-Onshore- und
    Wind-Offshore-Forecasts gesucht, deren letzte Zeitpunkte bestimmt und per
    job_log ausgegeben.
    """
    base = os.getenv("CSV_BASE", "/data")
    # Label und Pattern getrennt – so können wir Label-Spaltenbreite einheitlich formatieren
    sources = [
        ("Letzter Preis", f"{base}/entsoe_prices_*.csv"),
        ("Letzter Solar", f"{base}/entsoe_solar_forecast_*.csv"),
        ("Letzter Wind Onshore", f"{base}/entsoe_wind_onshore_forecast_*.csv"),
        ("Letzter Wind Offshore", f"{base}/entsoe_wind_offshore_forecast_*.csv"),
    ]
    label_width = max(len(lbl) for lbl, _ in sources)

    for label, pattern in sources:
        path = _latest_csv(pattern)
        last_ts = _read_last_time(path) if path else None
        # Einheitliche Ausrichtung des '@' mittels fester Label-Spaltenbreite
        job_log(f"🧭 {label:<{label_width}}  @ {_fmt(last_ts, tz_local)}")


def log_training_spans_from_csv(*, tz_local: str = "Europe/Berlin") -> None:
    """Ermittelt aus den CSV-Dateien die Anzahl Punkte sowie Start-/Endzeit der
    Trainingsdaten je Quelle (Solar, Wind Onshore, Wind Offshore) und loggt sie.

    Ausgabeformat (mit ausgerichtetem '|'):
      📈 Train Solar        : 27836 Punkte | 2025-01-01T00:00:00+01:00 – 2025-10-17T23:45:00+02:00
      📈 Train Wind Onshore : 27836 Punkte | ...
    """
    base = os.getenv("CSV_BASE", "/data")
    sources = [
        ("Solar", f"{base}/entsoe_solar_forecast_*.csv"),
        ("Wind Onshore", f"{base}/entsoe_wind_onshore_forecast_*.csv"),
        ("Wind Offshore", f"{base}/entsoe_wind_offshore_forecast_*.csv"),
    ]

    entries: list[dict] = []

    for name, pattern in sources:
        path = _latest_csv(pattern)
        if not path:
            entries.append({"name": name, "status": "empty"})
            continue
        try:
            df = pd.read_csv(path, usecols=[0])
            ts = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce").dropna()
            if ts.empty:
                entries.append({"name": name, "status": "empty"})
                continue
            n = int(len(ts))
            tmin = ts.min().tz_convert(tz_local)
            tmax = ts.max().tz_convert(tz_local)
            left = f"Train {name}: {n} Punkte"
            entries.append({
                "name": name,
                "status": "ok",
                "left": left,
                "tmin": tmin,
                "tmax": tmax,
            })
        except Exception as e:
            job_log(f"⚠️ Train {name}: Fehler beim Lesen {path.name}: {e}")

    # Einheitliche Breite für den linken Teil, damit '|' vertikal ausgerichtet ist
    width = max((len(e["left"]) for e in entries if e.get("status") == "ok"), default=0)

    for e in entries:
        if e.get("status") == "ok":
            job_log(f"📈 {e['left']:<{width}} | {e['tmin'].isoformat()} – {e['tmax'].isoformat()}")
        elif e.get("status") == "empty":
            job_log(f"📉 Train {e['name']}: leer")


def _read_series(csv_path: Path) -> pd.Series:
    """Liest eine Zeitreihe aus CSV: erste Spalte als Zeit (UTC), zweite als Wert.

    - Ungültige/parsfähige Einträge werden verworfen
    - Duplikate im Index werden per "last" aufgelöst
    - Rückgabe ist aufsteigend nach Zeit sortiert
    """
    df = pd.read_csv(csv_path, usecols=[0, 1])
    t = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce")
    v = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(v.values, index=t).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _make_lags(s: pd.Series, steps: list[int]) -> pd.DataFrame:
    """Erzeugt Lag-Features für die angegebenen Schrittweiten (in Zeitschritten).

    Beispiel: steps=[1,4] erzeugt Spalten lag_1 und lag_4 basierend auf s.shift().
    """
    return pd.concat({f"lag_{k}": s.shift(k) for k in steps}, axis=1)


def run_csv_training_and_forecast(*, n_lags: Optional[int] = None, tz_local: str = "Europe/Berlin") -> Optional[pd.DataFrame]:
    """Trainiert ein Ridge-Modell aus CSV-Zeitreihen (Preis + exogene) und erzeugt
    eine Vorhersage über den in HORIZON_H konfigurierten Horizont.

    Rückgabe: DataFrame der Vorhersage (oder None bei Abbruch). Schreibt/erweitert zusätzlich /data/price_forecast.csv.
    """
    base = os.getenv("CSV_BASE", "/data")
    step_str = os.getenv("STEP", "15min")
    horizon_h = int(os.getenv("HORIZON_H", "96"))
    step = pd.to_timedelta(step_str)

    # CSV-Pfade
    price_path = _latest_csv(f"{base}/entsoe_prices_*.csv")
    solar_path = _latest_csv(f"{base}/entsoe_solar_forecast_*.csv")
    won_path = _latest_csv(f"{base}/entsoe_wind_onshore_forecast_*.csv")
    wof_path = _latest_csv(f"{base}/entsoe_wind_offshore_forecast_*.csv")

    if not price_path:
        job_log("⚠️ Keine Preis-CSV gefunden – Abbruch.")
        return None

    try:
        price_s = _read_series(price_path)
    except Exception as e:
        job_log(f"⚠️ Preis-CSV unlesbar ({price_path.name}): {e}")
        return None

    # Trainingsraster
    full_index = pd.date_range(start=price_s.index.min(), end=price_s.index.max(), freq=step, tz="UTC")
    price_s = price_s.reindex(full_index).ffill()

    # Exogene (optional)
    def _read_opt(path: Optional[Path]) -> pd.Series:
        if not path:
            return pd.Series(index=full_index, dtype=float)
        try:
            s = _read_series(path).reindex(full_index)
            return s
        except Exception:
            return pd.Series(index=full_index, dtype=float)

    solar_s = _read_opt(solar_path)
    won_s = _read_opt(won_path)
    wof_s = _read_opt(wof_path)

    # Features
    train_df = pd.DataFrame(index=full_index)
    train_df["hour"] = train_df.index.tz_convert(tz_local).hour
    train_df["dayofweek"] = train_df.index.tz_convert(tz_local).dayofweek

    steps_per_day = int(pd.Timedelta(days=1) / step)
    lag_candidates = [1, 2, 4, 8, 12, 24, 48, 72, 96, steps_per_day]
    lag_steps = [k for k in lag_candidates if k < len(price_s)]
    if n_lags is not None and n_lags > 0:
        lag_steps = lag_steps[: n_lags]
    lag_df = _make_lags(price_s, lag_steps)

    train_df = pd.concat([train_df, lag_df], axis=1)
    train_df["solar"] = solar_s
    train_df["wind_onshore"] = won_s
    train_df["wind_offshore"] = wof_s
    train_df["price"] = price_s

    # Nur Zeilen mit vollständigen Lags nutzen
    model_df = train_df.dropna(subset=list(lag_df.columns))
    if model_df.empty:
        job_log("⚠️ Zu wenig Historie für Lags – Abbruch.")
        return None

    feature_cols = ["hour", "dayofweek"] + list(lag_df.columns) + ["solar", "wind_onshore", "wind_offshore"]
    X_train = model_df[feature_cols]
    y_train = model_df["price"]

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train, y_train)
    job_log(f"🤖 CSV-Modell (Ridge) trainiert. Features: {len(feature_cols)} | Trainingszeilen: {len(model_df)}")

    # Prognose-Index
    pred_start = full_index[-1] + step
    periods = int(pd.Timedelta(hours=horizon_h) / step)
    pred_index = pd.date_range(start=pred_start, periods=periods, freq=step, tz="UTC")
    job_log(
        f"🔮 CSV-Forecast: {pred_start.tz_convert(tz_local).isoformat()} – {pred_index[-1].tz_convert(tz_local).isoformat()} | Schritt {step_str} | Punkte {periods}"
    )

    # Exogene für Prognose (auf Index reindexen)
    # Für CSV-Variante nutzen wir die letzten verfügbaren Werte (ffill), falls die Exogenen in die Zukunft reichen.
    # Wenn keine Zukunftswerte vorhanden sind, bleiben NaNs und werden imputet.
    exo_pred = pd.DataFrame(index=pred_index)
    def _future_exo(s: pd.Series) -> pd.Series:
        # Wenn s nur Vergangenheit hat, reindex auf kombinierten Index und ffill
        comb_idx = s.index.append(pred_index).unique().sort_values()
        sf = s.reindex(comb_idx).ffill().reindex(pred_index)
        return sf

    exo_pred["solar"] = _future_exo(solar_s)
    exo_pred["wind_onshore"] = _future_exo(won_s)
    exo_pred["wind_offshore"] = _future_exo(wof_s)

    # Iterativ vorhersagen, um Lags aus eigenen Prognosen zu bilden
    hist_prices = price_s.copy()
    rows = []
    for t in pred_index:
        row = {
            "hour": t.tz_convert(tz_local).hour,
            "dayofweek": t.tz_convert(tz_local).dayofweek,
            "solar": exo_pred.at[t, "solar"],
            "wind_onshore": exo_pred.at[t, "wind_onshore"],
            "wind_offshore": exo_pred.at[t, "wind_offshore"],
        }
        for k in lag_steps:
            ts_lag = t - k * step
            v = hist_prices.get(ts_lag, np.nan)
            if pd.isna(v):
                v = float(hist_prices.iloc[-1])
            row[f"lag_{k}"] = float(v)
        X = pd.DataFrame([row], columns=feature_cols)
        yhat = float(pipe.predict(X)[0])
        rows.append({
            "time": t,
            "predicted_price_eur_mwh": yhat,
            "solar_input_mw": row.get("solar", np.nan),
            "wind_onshore_input_mw": row.get("wind_onshore", np.nan),
            "wind_offshore_input_mw": row.get("wind_offshore", np.nan),
        })
        hist_prices.loc[t] = yhat

    df_pred = pd.DataFrame(rows)
    job_log(f"📊 CSV-Prognose fertig: {len(df_pred)} Punkte")

    # CSV speichern: Upsert in feste Datei (Zeitstempel-basiert: ersetzen + neue anhängen)
    out_csv = f"{base}/price_forecast.csv"
    cols = [
        "time",
        "predicted_price_eur_mwh",
        "solar_input_mw",
        "wind_onshore_input_mw",
        "wind_offshore_input_mw",
    ]
    try:
        # Vorbereiten neuer Prognose-Zeilen
        df_new = df_pred[cols].copy()
        df_new["time"] = pd.to_datetime(df_new["time"], utc=True, errors="coerce")
        df_new = df_new.dropna(subset=["time"])  # nur gültige Zeiten
        df_new = df_new.drop_duplicates(subset=["time"], keep="last").copy()
        # Index auf Zeit setzen für stabile Vergleiche
        df_new = df_new.set_index("time").sort_index()

        replaced_count = 0
        new_count = len(df_new)

        if Path(out_csv).exists():
            try:
                df_old = pd.read_csv(out_csv, parse_dates=["time"])  # erwartet kompatible Spalten inkl. "time"
            except Exception:
                # Fallback: Erste Spalte als Zeit interpretieren
                df_tmp = pd.read_csv(out_csv)
                time_col = "time" if "time" in df_tmp.columns else df_tmp.columns[0]
                df_tmp["time"] = pd.to_datetime(df_tmp[time_col], utc=True, errors="coerce")
                df_old = df_tmp[["time"] + [c for c in df_tmp.columns if c != time_col and c != "time"]]

            # Zeitspalte vereinheitlichen (UTC) und Duplikate bereinigen
            if df_old["time"].dt.tz is None:
                df_old["time"] = df_old["time"].dt.tz_localize("UTC")
            df_old = df_old.dropna(subset=["time"]).copy()
            # Sicherstellen, dass alle erforderlichen Spalten existieren
            for c in cols:
                if c not in df_old.columns:
                    df_old[c] = pd.NA
            df_old = df_old[cols]
            df_old = df_old.drop_duplicates(subset=["time"], keep="last").set_index("time").sort_index()

            overlap_idx = df_old.index.intersection(df_new.index)
            new_only_idx = df_new.index.difference(df_old.index)

            # Vergleiche nur über Werte-Spalten; NaN==NaN gilt als gleich; Floats tolerant vergleichen
            value_cols = [
                "predicted_price_eur_mwh",
                "solar_input_mw",
                "wind_onshore_input_mw",
                "wind_offshore_input_mw",
            ]

            if len(overlap_idx) > 0:
                a = df_old.loc[overlap_idx, value_cols]
                b = df_new.loc[overlap_idx, value_cols]

                eq_df = pd.DataFrame(index=overlap_idx)
                for c in value_cols:
                    av = a[c]
                    bv = b[c]
                    # Beide NaN => gleich
                    both_nan = av.isna() & bv.isna()
                    # isclose auf gefüllten Werten, um Float-Rauschen zu ignorieren
                    av_f = av.fillna(0.0).astype(float)
                    bv_f = bv.fillna(0.0).astype(float)
                    close = pd.Series(
                        pd.Series(np.isclose(av_f.values, bv_f.values, rtol=1e-05, atol=1e-08), index=overlap_idx),
                        index=overlap_idx,
                    )
                    eq_df[c] = both_nan | close
                equal_all = eq_df.all(axis=1)
                replace_idx = equal_all.index[~equal_all]
                replaced_count = int(len(replace_idx))
            else:
                replace_idx = df_old.index[:0]

            new_count = int(len(new_only_idx))

            # Upsert zusammenbauen: 
            # - alte Zeilen außerhalb Overlap beibehalten
            # - Overlap: unveränderte behalten, veränderte durch neue ersetzen
            # - neue Zeiten anhängen
            keep_old_idx = df_old.index.difference(overlap_idx).union(equal_all.index[equal_all] if len(overlap_idx) else df_old.index[:0])
            parts = [
                df_old.loc[keep_old_idx],
                df_new.loc[replace_idx],
                df_new.loc[new_only_idx],
            ]
            df_upsert = pd.concat(parts).sort_index()
        else:
            df_upsert = df_new

        # Sortieren und schreiben (vollständiger Upsert-Stand)
        df_upsert = df_upsert.sort_index().reset_index()
        df_upsert.to_csv(out_csv, index=False, mode="w", header=True)

        job_log(
            f"✅ CSV-Forecast upsert: {out_csv} | neu: {new_count} | ersetzt: {replaced_count} | gesamt: {len(df_upsert)}"
        )
    except Exception as e:
        job_log(f"⚠️ CSV-Schreibfehler (CSV-Forecast): {e}")

    return df_pred


def compute_trend_hit_rate(*, tz_local: str = "Europe/Berlin") -> Optional[dict]:
    """Berechnet die Trend-Trefferquote zwischen Marktpreis (ENTSO-E) und Forecast.

    Definition: Ein Treffer liegt vor, wenn die Richtung der Preisänderung
    (steigend/fallend/gleich) im Forecast identisch zur Marktzeitreihe ist.

    Vergleichsbasis sind nur gemeinsame Zeitstempel. Der erste Punkt entfällt
    aufgrund der Differenzbildung. Nulländerungen werden als eigener Zustand gewertet.

    Rückgabe:
      {
        "hits": int,
        "total": int,
        "hit_rate_percent": float,
        "window_start": iso_local,
        "window_end": iso_local
      }
    oder None, wenn nicht genügend Daten vorhanden sind.
    """
    base = os.getenv("CSV_BASE", "/data")
    price_path = _latest_csv(f"{base}/entsoe_prices_*.csv")
    forecast_csv = Path(f"{base}/price_forecast.csv")

    if not price_path or not forecast_csv.exists():
        return None

    try:
        # Marktpreis-Serie laden (UTC, numerisch)
        s_price = _read_series(price_path)
        # Forecast laden
        df_f = pd.read_csv(forecast_csv, usecols=["time", "predicted_price_eur_mwh"])  # erwartet diese Spalten
        df_f["time"] = pd.to_datetime(df_f["time"], utc=True, errors="coerce")
        df_f = df_f.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
        s_fore = pd.Series(
            pd.to_numeric(df_f["predicted_price_eur_mwh"], errors="coerce").values,
            index=df_f["time"]
        ).dropna()
        s_fore = s_fore[~s_fore.index.duplicated(keep="last")].sort_index()

        # Gemeinsame Zeitstempel
        idx = s_price.index.intersection(s_fore.index)
        if len(idx) < 2:
            return None
        sp = s_price.reindex(idx)
        sf = s_fore.reindex(idx)

        # Differenzen und erneute Schnittmenge (erster Punkt entfällt je Serie)
        dp = sp.diff().dropna()
        dfc = sf.diff().dropna()
        idx2 = dp.index.intersection(dfc.index)
        if len(idx2) == 0:
            return None
        dp = dp.reindex(idx2)
        dfc = dfc.reindex(idx2)

        # Richtungen vergleichen mit Toleranz für "nahe 0"
        eps = 1e-9
        sign_p = np.where(np.abs(dp.values) <= eps, 0, np.sign(dp.values))
        sign_f = np.where(np.abs(dfc.values) <= eps, 0, np.sign(dfc.values))
        hits = int(np.sum(sign_p == sign_f))
        total = int(len(idx2))
        rate = float((hits / total) * 100.0) if total > 0 else 0.0

        start_local = idx2.min().tz_convert(tz_local).isoformat()
        end_local = idx2.max().tz_convert(tz_local).isoformat()

        return {
            "hits": hits,
            "total": total,
            "hit_rate_percent": round(rate, 2),
            "window_start": start_local,
            "window_end": end_local,
        }
    except Exception:
        return None


def compute_average_deviation(*, tz_local: str = "Europe/Berlin") -> Optional[dict]:
    """Berechnet die durchschnittliche Abweichung zwischen Marktpreis und Forecast.

    Einheit: ct/kWh (Cent pro kWh). Niedrigere Werte bedeuten präzisere Vorhersagen.

    Vorgehen:
      - entsoe_prices_*.csv (EUR/MWh) und price_forecast.csv (EUR/MWh) laden
      - auf gemeinsame Zeitstempel schneiden
      - mittlere absolute Abweichung berechnen und in ct/kWh umrechnen:
            |Preis_fore - Preis_markt| [EUR/MWh] / 10 = ct/kWh
    Rückgabe:
      {
        "avg_deviation_ct_per_kwh": float,  # auf 1 Nachkommastelle gerundet
        "count": int,
        "window_start": iso_local,
        "window_end": iso_local
      }
    oder None, wenn nicht genügend Daten vorhanden sind.
    """
    base = os.getenv("CSV_BASE", "/data")
    price_path = _latest_csv(f"{base}/entsoe_prices_*.csv")
    forecast_csv = Path(f"{base}/price_forecast.csv")

    if not price_path or not forecast_csv.exists():
        return None

    try:
        s_price = _read_series(price_path)  # EUR/MWh
        df_f = pd.read_csv(forecast_csv, usecols=["time", "predicted_price_eur_mwh"])
        df_f["time"] = pd.to_datetime(df_f["time"], utc=True, errors="coerce")
        df_f = df_f.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
        s_fore = pd.Series(pd.to_numeric(df_f["predicted_price_eur_mwh"], errors="coerce").values, index=df_f["time"]).dropna()
        s_fore = s_fore[~s_fore.index.duplicated(keep="last")].sort_index()

        idx = s_price.index.intersection(s_fore.index)
        if len(idx) == 0:
            return None
        p = s_price.reindex(idx)
        f = s_fore.reindex(idx)
        # Absolute Abweichung in EUR/MWh
        abs_err_eur_mwh = np.abs(f.values - p.values)
        # Umrechnung in ct/kWh (EUR/MWh / 10)
        abs_err_ct_kwh = abs_err_eur_mwh / 10.0
        avg = float(np.mean(abs_err_ct_kwh)) if len(abs_err_ct_kwh) else 0.0

        start_local = idx.min().tz_convert(tz_local).isoformat()
        end_local = idx.max().tz_convert(tz_local).isoformat()

        return {
            "avg_deviation_ct_per_kwh": round(avg, 1),
            "count": int(len(idx)),
            "window_start": start_local,
            "window_end": end_local,
            "note": "Niedrigere Werte bedeuten präzisere Vorhersagen.",
        }
    except Exception:
        return None
