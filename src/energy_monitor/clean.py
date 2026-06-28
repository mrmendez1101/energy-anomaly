from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

# ── Result models (C#: records with named fields) ────────────────────────────

class DataQualitySummary(BaseModel):
    site_key: str
    rows_in: int
    nulls_filled: int
    rows_dropped: int
    rows_out: int


class FillNullsResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: pd.DataFrame
    nulls_filled: int


class DropOutliersResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: pd.DataFrame
    rows_dropped: int


class CleanSiteResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: pd.DataFrame
    summary: DataQualitySummary


class CleanAllResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: pd.DataFrame
    summaries: list[DataQualitySummary]


# ── Metric → unit mapping ────────────────────────────────────────────────────

_UNIT_MAP: dict[str, str] = {
    "solar_radiation": "W/m2",
    "wind_speed_10m": "m/s",
    "wind_speed_100m": "m/s",
}

_RADIATION_METRICS: list[str] = ["solar_radiation"]
_WIND_METRICS: list[str] = ["wind_speed_10m", "wind_speed_100m"]


# ── Cleaner (C#: class Cleaner — owns a single transformation concern) ───────

class Cleaner:
    def __init__(
        self,
        max_interp_gap: int = 2,
        radiation_max: float = 1500.0,
        wind_max: float = 75.0,
    ) -> None:
        self._max_interp_gap = max_interp_gap
        self._radiation_max = radiation_max
        self._wind_max = wind_max

    def reshape_to_long(self, site_key: str, df: pd.DataFrame) -> pd.DataFrame:
        """Wide Parquet row → long format: one row per (timestamp, metric)."""
        wide = df.rename(columns={"shortwave_radiation": "solar_radiation"})
        metric_cols = ["solar_radiation", "wind_speed_10m", "wind_speed_100m"]

        long = wide.melt(
            id_vars=["time"], value_vars=metric_cols, var_name="metric", value_name="value"
        )

        ts = pd.to_datetime(long["time"])
        long["site_key"] = site_key
        long["ts_utc"] = ts
        long["ts_local"] = ts
        long["unit"] = [_UNIT_MAP[m] for m in long["metric"]]
        long["value"] = long["value"].astype("float64")

        cols = ["site_key", "ts_utc", "ts_local", "metric", "value", "unit"]
        return long[cols].reset_index(drop=True)  # type: ignore[return-value]

    def fill_nulls(self, df: pd.DataFrame) -> FillNullsResult:
        """Interpolate null gaps of ≤ max_interp_gap hours; leave longer gaps null."""
        result = df.copy()
        filled = result["value"].astype("float64")
        nulls_before = int(filled.isna().to_numpy().sum())

        groups = result.groupby(["site_key", "metric"], sort=False)
        for _, group_idx in groups.groups.items():
            s = filled.loc[group_idx]
            null_mask = s.isna()

            if not null_mask.any():
                continue

            # Label each contiguous run of nulls/non-nulls with a run id
            run_id = (null_mask != null_mask.shift()).cumsum()
            # Count how many nulls are in each null run
            run_lengths = null_mask.groupby(run_id).transform("sum")
            fill_mask = null_mask & (run_lengths <= self._max_interp_gap)

            if not fill_mask.any():
                continue

            interpolated = s.interpolate(method="linear")
            fill_idx = fill_mask.index[fill_mask]
            filled.loc[fill_idx] = interpolated.loc[fill_idx]

        nulls_after = int(filled.isna().to_numpy().sum())
        result["value"] = filled
        return FillNullsResult(data=result, nulls_filled=nulls_before - nulls_after)

    def assert_units(self, df: pd.DataFrame, units_block: dict[str, str]) -> pd.DataFrame:
        """Guard: raise if the API returned unexpected units for radiation or wind."""
        expected = {
            "shortwave_radiation": ("W/m²", "W/m2"),
            "wind_speed_10m": ("m/s",),
            "wind_speed_100m": ("m/s",),
        }
        for field, valid in expected.items():
            actual = units_block.get(field)
            if actual is not None and actual not in valid:
                raise ValueError(f"Unexpected unit for {field}: {actual!r}")
        return df

    def drop_outliers(self, df: pd.DataFrame) -> DropOutliersResult:
        """Clamp negatives to 0; drop readings above physical ceilings."""
        result = df.copy()
        result["value"] = result["value"].astype("float64").clip(lower=0.0)

        radiation_spike = result["metric"].isin(_RADIATION_METRICS) & (
            result["value"] > self._radiation_max
        )
        wind_spike = result["metric"].isin(_WIND_METRICS) & (result["value"] > self._wind_max)
        spike_mask = radiation_spike | wind_spike

        rows_dropped = int(spike_mask.to_numpy().sum())
        cleaned: pd.DataFrame = result[~spike_mask].reset_index(drop=True)  # type: ignore[assignment]
        return DropOutliersResult(data=cleaned, rows_dropped=rows_dropped)

    def clean_site(self, site_key: str, df: pd.DataFrame) -> CleanSiteResult:
        """Full cleaning pipeline for one site's wide Parquet DataFrame."""
        long = self.reshape_to_long(site_key, df)
        rows_in = len(long)

        fill_result = self.fill_nulls(long)
        drop_result = self.drop_outliers(fill_result.data)

        summary = DataQualitySummary(
            site_key=site_key,
            rows_in=rows_in,
            nulls_filled=fill_result.nulls_filled,
            rows_dropped=drop_result.rows_dropped,
            rows_out=len(drop_result.data),
        )
        return CleanSiteResult(data=drop_result.data, summary=summary)

    def clean_all(self, frames: dict[str, pd.DataFrame]) -> CleanAllResult:
        """Run clean_site for every site and concatenate into one long DataFrame."""
        all_data: list[pd.DataFrame] = []
        summaries: list[DataQualitySummary] = []

        for site_key, df in frames.items():
            site_result = self.clean_site(site_key, df)
            all_data.append(site_result.data)
            summaries.append(site_result.summary)

        combined = pd.concat(all_data, ignore_index=True)
        return CleanAllResult(data=combined, summaries=summaries)
