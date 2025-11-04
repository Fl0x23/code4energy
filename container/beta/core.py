from __future__ import annotations

import os

# Root-Prefix (für Reverse Proxy), z. B. "/beta"; per ENV ROOT_PATH anpassbar
ROOT_PATH = os.getenv("ROOT_PATH", "/beta")

APP_INFO = {
    "app": os.getenv("APP_NAME", "Code 4 Energy Beta API"),
    "description": os.getenv("APP_DESCRIPTION", "Dient als Vorlage"),
    "version": os.getenv("APP_VERSION", "1.0.0"),
    "root_path": ROOT_PATH,
    "docs_url": f"{ROOT_PATH}/docs",
    "redoc_url": f"{ROOT_PATH}/redoc",
    "openapi_url": f"{ROOT_PATH}/openapi.json",
}
