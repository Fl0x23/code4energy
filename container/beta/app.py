"""FastAPI-Einstiegspunkt für den Beta-Service.

Struktur analog zu initial/:
- core.py     -> Basiskonfiguration (ROOT_PATH, APP_INFO, Lifespan)
- services.py -> Geschäftslogik (CSV lesen usw.)
- routes.py   -> HTTP-Endpunkte
- schemas.py  -> optionale Pydantic-Schemas
"""

from fastapi import FastAPI

from core import APP_INFO, ROOT_PATH
from routes import router


app = FastAPI(
    title=APP_INFO["app"],
    description=APP_INFO["description"],
    version=APP_INFO["version"],
    root_path=ROOT_PATH,
)

# Routen registrieren
app.include_router(router)