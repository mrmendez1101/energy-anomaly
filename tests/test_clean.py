from __future__ import annotations

import pandas as pd

from energy_monitor.clean import (
    Cleaner,
    CleanSiteResult,
    DropOutliersResult,
    FillNullsResult,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _wide_df(
    times: list[str],
    radiation: list[float | None],
    wind_10m: list[float | None],
    wind_100m: list[float | None],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": times,
            "shortwave_radiation": radiation,
            "wind_speed_10m": wind_10m,
            "wind_speed_100m": wind_100m,
        }
    )


def _hours(n: int, start: str = "2026-05-01T00:00") -> list[str]:
    return pd.date_range(start, periods=n, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()


def _long_df(
    metric: str,
    values: list[float | None],
    unit: str = "W/m2",
) -> pd.DataFrame:
    n = len(values)
    ts = pd.to_datetime(_hours(n))
    return pd.DataFrame(
        {
            "site_key": ["burgos_ph"] * n,
            "ts_utc": ts,
            "ts_local": ts,
            "metric": [metric] * n,
            "value": pd.array(values, dtype="Float64"),
            "unit": [unit] * n,
        }
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_reshape_produces_long_format() -> None:
    cleaner = Cleaner()
    wide = _wide_df(
        times=_hours(2),
        radiation=[100.0, 200.0],
        wind_10m=[3.0, 4.0],
        wind_100m=[5.0, 6.0],
    )
    long = cleaner.reshape_to_long("burgos_ph", wide)

    assert len(long) == 6  # 2 timestamps × 3 metrics
    assert set(long.columns) == {"site_key", "ts_utc", "ts_local", "metric", "value", "unit"}
    assert set(long["metric"].unique()) == {"solar_radiation", "wind_speed_10m", "wind_speed_100m"}
    assert (long["site_key"] == "burgos_ph").all()


def test_nulls_short_gap_interpolated() -> None:
    cleaner = Cleaner(max_interp_gap=2)
    long = _long_df("solar_radiation", [100.0, None, 300.0, 400.0])

    result = cleaner.fill_nulls(long)

    assert isinstance(result, FillNullsResult)
    assert result.nulls_filled == 1
    assert result.data["value"].isna().sum() == 0


def test_nulls_long_gap_left_null() -> None:
    cleaner = Cleaner(max_interp_gap=2)
    long = _long_df("solar_radiation", [100.0, None, None, None, None, 600.0])

    result = cleaner.fill_nulls(long)

    assert result.nulls_filled == 0
    assert result.data["value"].isna().sum() == 4


def test_negative_radiation_clamped() -> None:
    cleaner = Cleaner()
    long = _long_df("solar_radiation", [-50.0, 100.0, 200.0])

    result = cleaner.drop_outliers(long)

    assert isinstance(result, DropOutliersResult)
    assert (result.data["value"] >= 0).all()
    assert result.rows_dropped == 0
    assert len(result.data) == 3


def test_spike_dropped() -> None:
    cleaner = Cleaner(radiation_max=1500.0)
    long = _long_df("solar_radiation", [100.0, 2000.0, 200.0])

    result = cleaner.drop_outliers(long)

    assert result.rows_dropped == 1
    assert len(result.data) == 2
    assert 2000.0 not in result.data["value"].tolist()


def test_dq_summary_counts_correct() -> None:
    cleaner = Cleaner(max_interp_gap=2, radiation_max=1500.0, wind_max=75.0)
    wide = _wide_df(
        times=_hours(4),
        radiation=[100.0, None, 300.0, 2000.0],
        wind_10m=[3.0, 4.0, 3.5, 4.5],
        wind_100m=[5.0, 6.0, 5.5, 6.5],
    )
    result = cleaner.clean_site("burgos_ph", wide)

    assert isinstance(result, CleanSiteResult)
    summary = result.summary
    assert summary.site_key == "burgos_ph"
    assert summary.rows_in == 12   # 4 timestamps × 3 metrics
    assert summary.nulls_filled == 1
    assert summary.rows_dropped == 1
    assert summary.rows_out == 11  # 12 - 1 dropped
