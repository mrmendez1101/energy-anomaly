from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
from pydantic import BaseModel

from energy_monitor.config import AppConfig, DateWindow, Site

# ── Pydantic response models (C#: typed DTOs for the API response) ──────────

class HourlyUnits(BaseModel):
    time: str
    shortwave_radiation: str
    wind_speed_10m: str
    wind_speed_100m: str


class HourlyData(BaseModel):
    time: list[str]
    shortwave_radiation: list[float | None]
    wind_speed_10m: list[float | None]
    wind_speed_100m: list[float | None]


class OpenMeteoResponse(BaseModel):
    latitude: float
    longitude: float
    timezone: str
    hourly_units: HourlyUnits
    hourly: HourlyData


# ── Client (C#: class OpenMeteoClient : IOpenMeteoClient) ───────────────────

class OpenMeteoClient:
    """Fetches hourly weather data from the Open-Meteo Archive API."""

    _BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(
        self,
        base_url: str = _BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout

    def fetch_site(self, site: Site, window: DateWindow) -> pd.DataFrame:
        """Fetch one site for the fixed window. Returns a raw wide DataFrame."""
        params = {
            "latitude": site.lat,
            "longitude": site.lon,
            "hourly": "shortwave_radiation,wind_speed_10m,wind_speed_100m",
            "start_date": window.start,
            "end_date": window.end,
            "wind_speed_unit": "ms",
            "timezone": "auto",
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(self._base_url, params=params)
            response.raise_for_status()

        data = OpenMeteoResponse.model_validate(response.json())

        df = pd.DataFrame(
            {
                "time": data.hourly.time,
                "shortwave_radiation": data.hourly.shortwave_radiation,
                "wind_speed_10m": data.hourly.wind_speed_10m,
                "wind_speed_100m": data.hourly.wind_speed_100m,
            }
        )
        print(f"  {site.key}: {len(df)} rows")
        return df

    def fetch_all(self, config: AppConfig) -> dict[str, pd.DataFrame]:
        """Fetch all sites defined in config. Returns dict keyed by site_key."""
        frames: dict[str, pd.DataFrame] = {}
        for site in config.sites:
            print(f"Fetching {site.key} ...")
            frames[site.key] = self.fetch_site(site, config.window)
        return frames

    def save_snapshot(
        self,
        frames: dict[str, pd.DataFrame],
        snapshot_dir: Path,
    ) -> None:
        """Write one Parquet file per site. Creates snapshot_dir if needed."""
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for site_key, df in frames.items():
            out = snapshot_dir / f"{site_key}.parquet"
            df.to_parquet(out, index=False)
            print(f"  Saved {out}")
