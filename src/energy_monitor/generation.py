from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel

# C#: generation potential is its own concern, so it gets its own class —
# not a method bolted onto AnomalyDetector. Capacity factors are comparison
# metrics, not anomaly detection.


class WindPowerCurve(BaseModel):
    """Simplified turbine power curve (C#: a small config record)."""

    cut_in: float = 3.0   # below this speed: no output
    rated: float = 12.0   # at/above this (up to cut_out): full output
    cut_out: float = 25.0  # above this: turbine shuts down for safety


class SolarReference(BaseModel):
    reference_irradiance: float = 1000.0  # W/m2, standard test condition (STC)


class GenerationPotential:
    """Derives normalised capacity-factor metrics for cross-site comparison.

    Raw radiation (W/m2) and wind speed (m/s) are on different scales, so they
    cannot be compared directly. A capacity factor in [0, 1] makes every site
    comparable on one axis. These are transparent proxies, not engineering-grade
    power models — the README says so.
    """

    SOLAR_METRIC = "solar_cf"
    WIND_METRIC = "wind_cf"
    UNIT = "capacity_factor"

    def __init__(
        self,
        wind_curve: WindPowerCurve | None = None,
        solar_ref: SolarReference | None = None,
    ) -> None:
        self._wind = wind_curve or WindPowerCurve()
        self._solar = solar_ref or SolarReference()

    def solar_capacity_factor(self, radiation: pd.Series) -> pd.Series:
        """solar_cf = clamp(radiation / 1000, 0, 1). 1000 W/m2 maps to full output."""
        cf = radiation.astype("float64") / self._solar.reference_irradiance
        return cf.clip(lower=0.0, upper=1.0)

    def wind_capacity_factor(self, wind_speed: pd.Series) -> pd.Series:
        """wind_cf from a simplified power curve on hub-height (100 m) wind.

        0 below cut-in, a cubic ramp (speed/rated)**3 up to rated, flat at 1.0
        to cut-out, then 0 (shutdown). Cubic because wind power scales with the
        cube of speed.
        """
        v = wind_speed.astype("float64").to_numpy()
        cf = np.zeros_like(v)

        ramp = (v >= self._wind.cut_in) & (v <= self._wind.rated)
        flat = (v > self._wind.rated) & (v <= self._wind.cut_out)

        cf[ramp] = (v[ramp] / self._wind.rated) ** 3
        cf[flat] = 1.0
        cf = np.clip(cf, 0.0, 1.0)

        # A null wind reading is "unknown", not "zero output". The boolean masks
        # above are all False for NaN, so without this a gap would silently become
        # 0.0 — and that would disagree with solar_cf, which preserves NaN via the
        # divide-and-clip. Keep both metrics honest about missing data.
        cf[np.isnan(v)] = np.nan

        return pd.Series(cf, index=wind_speed.index)

    def add_generation_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append solar_cf and wind_cf rows to a long-format readings frame.

        solar_cf derives from solar_radiation, wind_cf from wind_speed_100m
        (closer to turbine hub height than the 10 m reading). The new rows carry
        is_anomaly=False / anomaly_score=0.0 — capacity factors are a comparison
        view, not anomaly targets.
        """
        solar_rows = self._derive_rows(
            df, "solar_radiation", self.SOLAR_METRIC, self.solar_capacity_factor
        )
        wind_rows = self._derive_rows(
            df, "wind_speed_100m", self.WIND_METRIC, self.wind_capacity_factor
        )
        return pd.concat([df, solar_rows, wind_rows], ignore_index=True)

    def _derive_rows(
        self,
        df: pd.DataFrame,
        source_metric: str,
        target_metric: str,
        transform: Callable[[pd.Series], pd.Series],
    ) -> pd.DataFrame:
        """Build derived-metric rows from one source metric's rows."""
        source: pd.DataFrame = df[df["metric"] == source_metric].copy()  # type: ignore[assignment]
        source["value"] = transform(source["value"]).to_numpy()  # type: ignore[arg-type]
        source["metric"] = target_metric
        source["unit"] = self.UNIT
        if "is_anomaly" in source.columns:
            source["is_anomaly"] = False
        if "anomaly_score" in source.columns:
            source["anomaly_score"] = 0.0
        return source.reset_index(drop=True)
