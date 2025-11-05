### Beta Service

Vorlage/Referenz für neue Vorhersagemodelle

---

#### Mindestanforderungen 
Folgende Dateien müssen im Container-Verzeichnis (z. B. `container/<container-name>/`) vorhanden sein:

- `app.py`
- `Dockerfile`
- `README.md`
- `openapi.yml`

---

#### Empfohlene Container Struktur

```
container/
└── <container-name>/
    ├── app.py          # Minimaler Einstiegspunkt (FastAPI) – bindet Router ein
    ├── routes.py       # HTTP-Endpoints (/info, /forecast)
    ├── services.py     # Geschäftslogik (z. B. CSV lesen)
    ├── core.py         # APP_INFO, ROOT_PATH u. a. Basiskonfiguration
    ├── schemas.py      # optionale Pydantic-Modelle
    └── requirements.txt
data/
└── <container-name>/   # (Mount über docker-compose optional)
    └── data.csv        # Beispiel: CSV-Datei
```

---

#### Vorlage/Referenz kopieren
1) Repository klonen

```
git clone https://github.com/Fl0x23/code4energy.git
cd code4energy
```

2) Ordner duplizieren und umbenennen

```
cp -r container/beta container/<container-name>
```

3) Container bearbeiten

```
Create your own Project 🎉
```

4) Build & Run Local

```
docker-compose up -d --build <container-name>
```
