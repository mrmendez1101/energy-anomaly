from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from energy_monitor.config import DateWindow, Site
from energy_monitor.ingest import OpenMeteoClient, OpenMeteoResponse

# ── Minimal fixture that matches the API contract ────────────────────────────

_VALID_RESPONSE: dict = {
    "latitude": 18.52,
    "longitude": 120.65,
    "timezone": "Asia/Manila",
    "hourly_units": {
        "time": "iso8601",
        "shortwave_radiation": "W/m²",
        "wind_speed_10m": "m/s",
        "wind_speed_100m": "m/s",
    },
    "hourly": {
        "time": ["2026-05-01T00:00", "2026-05-01T01:00"],
        "shortwave_radiation": [0.0, 10.5],
        "wind_speed_10m": [3.2, 3.8],
        "wind_speed_100m": [5.1, 6.0],
    },
}


# ── Pydantic model validation ────────────────────────────────────────────────

def test_response_model_valid() -> None:
    data = OpenMeteoResponse.model_validate(_VALID_RESPONSE)
    assert data.latitude == 18.52
    assert len(data.hourly.time) == 2


def test_response_model_rejects_missing_field() -> None:
    # hourly_units is required; drop it and expect Pydantic to raise
    bad = {k: v for k, v in _VALID_RESPONSE.items() if k != "hourly_units"}
    with pytest.raises(ValidationError):
        OpenMeteoResponse.model_validate(bad)


# ── OpenMeteoClient with mocked HTTP ─────────────────────────────────────────

def _site() -> Site:
    return Site(key="burgos_ph", label="Burgos PH", lat=18.52, lon=120.65)


def _window() -> DateWindow:
    return DateWindow(start="2026-05-01", end="2026-05-30")


def test_fetch_site_returns_expected_columns() -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = _VALID_RESPONSE
    mock_resp.raise_for_status.return_value = None

    mock_http = MagicMock()
    mock_http.get.return_value = mock_resp
    mock_http.__enter__ = MagicMock(return_value=mock_http)
    mock_http.__exit__ = MagicMock(return_value=False)

    with patch("energy_monitor.ingest.httpx.Client", return_value=mock_http):
        client = OpenMeteoClient()
        df = client.fetch_site(_site(), _window())

    assert set(df.columns) == {"time", "shortwave_radiation", "wind_speed_10m", "wind_speed_100m"}
    assert len(df) == 2
