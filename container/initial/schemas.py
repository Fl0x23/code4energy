from __future__ import annotations

from typing import Literal
from pydantic import BaseModel


class ForecastItem(BaseModel):
    start: str
    end: str
    price: float
    price_origin: Literal["forecast", "market"]


class AppInfo(BaseModel):
    app: str
    description: str
    version: str
    root_path: str
    docs_url: str
    redoc_url: str
    openapi_url: str
