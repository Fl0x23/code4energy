#!/usr/bin/env python3
"""
Validate that public/docs/apis.json matches ./container/*
- Ensures each subfolder in container has an entry {name, url: /<name>/openapi.json}
- Ensures deterministic sorted order by name
- Optionally checks that container/<name>/openapi.yml exists
Exit code 1 when mismatch is found.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
container_dir = ROOT / "container"
manifest_file = ROOT / "public" / "docs" / "apis.json"

errors = []

if not container_dir.exists():
    errors.append(f"container directory not found: {container_dir}")
else:
    names = []
    for p in container_dir.iterdir():
        if p.is_dir() and not p.name.startswith('.'):
            names.append(p.name)
    names = sorted(names)

    # Build expected entries
    expected = [{"name": n, "url": f"/{n}/openapi.json"} for n in names]

    # Ensure each container has required files
    required = [
        "openapi.yml",
        "Dockerfile",
        "README.md",
        "app.py",
    ]
    for n in names:
        for fname in required:
            fpath = container_dir / n / fname
            if not fpath.exists():
                errors.append(f"missing {fname} in container/{n}")

    # Load current manifest
    if not manifest_file.exists():
        errors.append(f"manifest missing: {manifest_file}")
    else:
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                errors.append("manifest is not a JSON array")
            else:
                # Normalize and sort by name
                def norm(e):
                    return {
                        "name": str(e.get("name", "")).strip(),
                        "url": str(e.get("url", "")).strip(),
                    }
                current = [norm(e) for e in data]
                current = sorted(current, key=lambda x: x["name"])
                if current != expected:
                    errors.append(
                        "apis.json does not match container/* (names or urls or order differ)"
                    )
        except Exception as e:
            errors.append(f"failed to read manifest JSON: {e}")

if errors:
    print("[check_apis] FAILED:\n - " + "\n - ".join(errors))
    print("\nRun: python scripts/generate_apis.py and commit the updated public/docs/apis.json")
    sys.exit(1)

print("[check_apis] OK: apis.json matches container/*")
