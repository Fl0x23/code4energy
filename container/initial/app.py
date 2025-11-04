"""FastAPI-Einstiegspunkt für den Initial-Service.

App-Setup, Lifecycle und Routen sind in Module aufgeteilt:
- core.py     -> Basiskonfiguration (Timezone, Scheduler, Lifespan, APP_INFO)
- services.py -> Hintergrundjobs/ETL
- routes.py   -> HTTP-Endpunkte
- schemas.py  -> (optional) Pydantic-Schemas
"""

from fastapi import FastAPI

from core import APP_INFO, ROOT_PATH, lifespan
from routes import router


app = FastAPI(
    title=APP_INFO["app"],
    description=APP_INFO["description"],
    version=APP_INFO["version"],
    lifespan=lifespan,
    root_path=ROOT_PATH,
)

# Routen registrieren
app.include_router(router)
