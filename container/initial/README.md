## Initial Service (FastAPI + CSV Forecast)

Basis-Root: `/initial`

Verfügbare Endpoints:
- `GET /info` – Metadaten zur App
- `GET /trend` – Trend-Hitrate zwischen Marktpreis und Forecast
- `GET /deviation` – Durchschnittliche Abweichung (MAE) in ct/kWh
- `GET /forecast` – Forecast und optional Marktdaten als Intervalle

### GET `/forecast`
Liefert Forecast-Daten (und optional Marktdaten) als JSON-Intervalle.

Query-Parameter:
- `start`: ISO-Zeitpunkt (interpretiert als lokale Zeit, falls ohne TZ).
  Standard: 00:00 lokale Zeit des Tages des letzten verfügbaren Marktwerts abzüglich `days` Tage.
  Sind keine Marktdaten vorhanden und `start` fehlt, wird `404` zurückgegeben.
- `end`: ISO-Zeitpunkt (interpretiert als lokale Zeit, falls ohne TZ).
  Standard: letztes Zeitfenster der Forecast-CSV (falls vorhanden), sonst jetzt.
- `market`: Wenn `true`, werden zusätzlich vorhandene Marktdaten aus `entsoe_prices_*.csv` ausgegeben. Default: `false`.
- `days`: Anzahl Tage zurück für den Standard-Start (wenn `start` nicht gesetzt ist). Default: `1` (bezogen auf den Tag des letzten Marktwerts).
- `previous_forecast`: Wenn `false`, werden Forecast-Werte bis einschließlich des global
  letzten Marktzeitpunkts ausgeblendet (Filter: `t > letzter Marktzeitpunkt`). Default: `false`.

Antwort (Zeitstempel in lokaler System-Zeitzone): Liste von Objekten

```jsonc
{
  "start": "ISO-LOCAL",      // inkl. Millisekunden und Offset
  "end": "ISO-LOCAL",        // = start + Schrittweite (z. B. 15min)
  "price": 0.05494,           // EUR/kWh (5 Nachkommastellen)
  "price_origin": "forecast" // oder "market"
}
```

Beispiele:

```
GET /forecast
GET /forecast?market=true&previous_forecast=true
GET /forecast?market=false&previous_forecast=true
GET /forecast?market=true&previous_forecast=false
GET /forecast?market=false&previous_forecast=false
GET /forecast?start=2025-10-28&end=2025-10-29
GET /forecast?start=2025-10-28T00:00:00+01:00&end=2025-10-28T12:00:00+01:00&market=true
GET /forecast?start=2025-10-28T00:00&market=true
```