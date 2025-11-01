#!/usr/bin/env python3
"""
Generate public/docs/apis.json from subfolders in ./container/*
Each entry will be {"name": "<folder>", "url": "/<folder>/openapi.json"}
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
container_dir = ROOT / "container"
output_file = ROOT / "public" / "docs" / "apis.json"

if not container_dir.exists():
    print(f"container directory not found: {container_dir}", file=sys.stderr)
    sys.exit(1)

names = []
for p in sorted(container_dir.iterdir()):
    if not p.is_dir():
        continue
    name = p.name.strip()
    if not name or name.startswith('.'):
        continue
    names.append(name)

entries = [{"name": n, "url": f"/{n}/openapi.json"} for n in names]
output_file.parent.mkdir(parents=True, exist_ok=True)
with output_file.open("w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(entries)} entries to {output_file}")
