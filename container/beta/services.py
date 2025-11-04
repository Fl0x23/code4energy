from __future__ import annotations

import csv
from typing import List, Dict, Any


def read_forecast_demo(csv_path: str = "/data/price_forecast_demo.csv") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Preis als float, Rest als String belassen
            price_val = r.get("price")
            try:
                price = float(price_val) if price_val not in (None, "") else None
            except Exception:
                price = None
            rows.append(
                {
                    "start": r.get("start"),
                    "end": r.get("end"),
                    "price": price,
                    "price_origin": r.get("price_origin") or "forecast",
                }
            )
    return rows
