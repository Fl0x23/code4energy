<img width="1241" height="691" alt="image" src="https://github.com/user-attachments/assets/750ffec6-5862-47a2-91cd-ce5a2d5bc328" />

# Code 4 Energy – Die Energieprognose der nächsten Generation
Erstellen Sie ein neues Vorhersagemodell oder verbessern Sie ein bestehendes.

## Integration in EVCC (Initialmodell)
```
tariffs: 
  currency: EUR
  grid:
    type: custom
    forecast:
      source: http
      uri: https://code4energy.de/initial/forecast?market=true
      jq: 'map({start: (.start | strptime("%Y-%m-%dT%H:%M:%S.%f%z") | mktime | strftime("%Y-%m-%dT%H:%M:%SZ")), end: (.end | strptime("%Y-%m-%dT%H:%M:%S.%f%z") | mktime | strftime("%Y-%m-%dT%H:%M:%SZ")), value: .price}) | tostring'
```
