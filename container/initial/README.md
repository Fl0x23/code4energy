## Initial Service (FastAPI + CSV Forecast)

### GET `/forecast`
Liefert Forecast-Daten (und optional Marktdaten) als JSON-Intervalle.

Query-Parameter:
- `start`: ISO-Zeitpunkt (interpretiert als lokale Zeit, falls ohne TZ). Standard: Gestern 00:00 lokale Zeit.
- `end`: ISO-Zeitpunkt (interpretiert als lokale Zeit, falls ohne TZ). Standard: letztes Zeitfenster der Forecast-CSV.
- `market`: Wenn `true`, werden zusätzlich vorhandene Marktdaten aus `entsoe_prices_*.csv` ausgegeben.
- `days`: Anzahl Tage zurück für den Standard-Start (wenn `start` nicht gesetzt ist). Default: `1` ("Gestern").
- `previous_forecast`: Wenn `false`, werden Forecast-Werte vor dem Ende der Marktdaten
  im gewählten Zeitraum nicht geliefert (t < letzter Marktzeitpunkt). Default: `false`.

Antwort (Zeitstempel in lokaler System-Zeitzone): Liste von Objekten

```json
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
GET /forecast?market=true
GET /forecast?start=2025-10-28&end=2025-10-29
GET /forecast?start=2025-10-28T00:00:00+01:00&end=2025-10-28T12:00:00+01:00&market=true
GET /forecast?start=2025-10-28T00:00&market=true
```