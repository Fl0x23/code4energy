FROM python:3.12-slim

# Arbeitsdir
WORKDIR /app

# Deps (cache-freundlich)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Dateien
COPY . .

# Ungepufferte Ausgabe für unmittelbare Logs
ENV PYTHONUNBUFFERED=1

# Exponierter Port
EXPOSE 8000

# Start
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]