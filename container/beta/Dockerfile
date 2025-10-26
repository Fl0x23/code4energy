FROM python:3.12-slim

# Arbeitsdir
WORKDIR /app

# Deps (cache-freundlich)
COPY requirements.txt .
RUN pip install -r requirements.txt

# App-Dateien
COPY app.py ./
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]