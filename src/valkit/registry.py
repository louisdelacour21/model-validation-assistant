"""Minimal model registry: one YAML file, validated with pydantic, one entry per model."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class Entry(BaseModel):
    name: str
    kind: Literal["pricing", "var"]
    independence: Literal["self-review", "independent"]
    opinion: str
    campaign_date: str
    campaign_dir: str
    open_findings: list[str]


def update_registry(path: Path, entry: Entry) -> Path:
    path = Path(path)
    old = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else []
    rows = {e["name"]: Entry(**e) for e in old or []}
    rows[entry.name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump([rows[k].model_dump() for k in sorted(rows)], sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path
