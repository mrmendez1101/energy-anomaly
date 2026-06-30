from __future__ import annotations

import pandas as pd

from energy_monitor.generation import GenerationPotential

# ── Helpers ───────────────────────────────────────────────────────────────────


def _solar(value: float) -> float:
    return float(GenerationPotential().solar_capacity_factor(pd.Series([value])).iloc[0])


def _wind(value: float) -> float:
    return float(GenerationPotential().wind_capacity_factor(pd.Series([value])).iloc[0])


def _long_df() -> pd.DataFrame:
    """Minimal long-format frame with the three physical metrics, one row each."""
    ts = pd.Timestamp("2026-05-01 12:00")
    rows = [
        ("burgos_ph", "solar_radiation", 1000.0, "W/m2"),
        ("burgos_ph", "wind_speed_10m", 5.0, "m/s"),
        ("burgos_ph", "wind_speed_100m", 12.0, "m/s"),
    ]
    return pd.DataFrame(
        {
            "site_key": [r[0] for r in rows],
            "ts_utc": [ts] * len(rows),
            "ts_local": [ts] * len(rows),
            "metric": [r[1] for r in rows],
            "value": [r[2] for r in rows],
            "unit": [r[3] for r in rows],
            "is_anomaly": [False] * len(rows),
            "anomaly_score": [0.0] * len(rows),
        }
    )


# ── Solar capacity factor ─────────────────────────────────────────────────────


def test_solar_cf_at_reference() -> None:
    """1000 W/m2 maps to full capacity (1.0)."""
    assert _solar(1000.0) == 1.0


def test_solar_cf_at_night() -> None:
    """0 W/m2 maps to no output (0.0)."""
    assert _solar(0.0) == 0.0


def test_solar_cf_clamped_above_reference() -> None:
    """Readings above 1000 W/m2 are clamped to 1.0, not >1."""
    assert _solar(1500.0) == 1.0


# ── Wind capacity factor ──────────────────────────────────────────────────────


def test_wind_cf_below_cut_in() -> None:
    """Below the 3 m/s cut-in: no output."""
    assert _wind(2.0) == 0.0


def test_wind_cf_at_rated() -> None:
    """At the 12 m/s rated speed: full output."""
    assert _wind(12.0) == 1.0


def test_wind_cf_ramp_midpoint() -> None:
    """Cubic ramp: 6 m/s → (6/12)**3 = 0.125."""
    assert _wind(6.0) == 0.125


def test_wind_cf_flat_region() -> None:
    """Between rated and cut-out the curve is flat at full output."""
    assert _wind(20.0) == 1.0


def test_wind_cf_above_cut_out() -> None:
    """Above the 25 m/s cut-out the turbine shuts down: no output."""
    assert _wind(30.0) == 0.0


def test_wind_cf_null_stays_null() -> None:
    """A missing wind reading is unknown, not zero output — must stay NaN."""
    import math

    assert math.isnan(_wind(float("nan")))


def test_solar_cf_null_stays_null() -> None:
    """A missing solar reading must stay NaN too (symmetry with wind)."""
    import math

    assert math.isnan(_solar(float("nan")))


# ── add_generation_metrics ────────────────────────────────────────────────────


def test_generation_metrics_present_in_output() -> None:
    """Both derived metrics appear as new rows, leaving the originals intact."""
    enriched = GenerationPotential().add_generation_metrics(_long_df())
    metrics = set(enriched["metric"])
    assert "solar_cf" in metrics
    assert "wind_cf" in metrics
    assert {"solar_radiation", "wind_speed_10m", "wind_speed_100m"} <= metrics


def test_derived_rows_are_not_flagged() -> None:
    """Capacity-factor rows are comparison metrics, never anomalies."""
    enriched = GenerationPotential().add_generation_metrics(_long_df())
    cf_rows: pd.DataFrame = enriched[enriched["metric"].isin(["solar_cf", "wind_cf"])]  # type: ignore[assignment]
    assert not bool(cf_rows["is_anomaly"].any())
    assert bool((cf_rows["unit"] == "capacity_factor").all())


def test_derived_values_use_correct_source_metric() -> None:
    """solar_cf comes from solar_radiation; wind_cf from wind_speed_100m (not 10m)."""
    enriched = GenerationPotential().add_generation_metrics(_long_df())
    solar_cf = float(enriched.loc[enriched["metric"] == "solar_cf", "value"].iloc[0])
    wind_cf = float(enriched.loc[enriched["metric"] == "wind_cf", "value"].iloc[0])
    assert solar_cf == 1.0  # from 1000 W/m2
    assert wind_cf == 1.0   # from wind_speed_100m = 12 m/s (rated), not 10m = 5 m/s
