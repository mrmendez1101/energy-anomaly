from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


# C#: public record Site(string Key, string Label, float Lat, float Lon);
class Site(BaseModel):
    key: str
    label: str
    lat: float
    lon: float


# C#: public record DateWindow(string Start, string End);
class DateWindow(BaseModel):
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def must_be_iso_date(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"Date must be YYYY-MM-DD, got: {v}")
        return v


# C#: public record AppConfig(List<Site> Sites, DateWindow Window);
class AppConfig(BaseModel):
    sites: list[Site]
    window: DateWindow


def load_config(path: Path = Path("config/sites.yaml")) -> AppConfig:
    """Read sites.yaml and return a validated AppConfig."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(
        sites=[Site(**s) for s in raw["sites"]],
        window=DateWindow(**raw["window"]),
    )
