FROM python:3.11-slim

# Arbeitsverzeichnis im Container
WORKDIR /app

# Schritt 1: Nur die Abhängigkeiten-Spezifikation kopieren (bessere Docker-Layer-Caches)
COPY requirements.txt .

# Schritt 2: Python-Abhängigkeiten installieren (Cache greift solange requirements.txt unverändert bleibt)
RUN pip install --no-cache-dir -r requirements.txt

# Schritt 3: Anwendungscode kopieren
COPY . .

# Ungepufferte STDOUT/STDERR, damit Logs sofort in "docker logs" sichtbar sind
ENV PYTHONUNBUFFERED=1

# Exponierter Port der App
EXPOSE 8000

# Serverstart (Uvicorn) mit aktivierten Lifespan-Events
# Erläuterung zu --lifespan on:
#   - FastAPI besitzt einen Lifespan-Kontext (startup/shutdown)
#   - In diesem Projekt wird im startup u. a. der Scheduler (APScheduler) gestartet
#   - "--lifespan on" stellt sicher, dass diese Events direkt beim Containerstart
#     ausgeführt werden (und nicht erst beim ersten HTTP-Request)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--lifespan", "on"]