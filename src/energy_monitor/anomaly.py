from __future__ import annotations

import pandas as pd
from pydantic import BaseModel


class IQRConfig(BaseModel):
    multiplier: float = 1.5
    variance_floor: float = 1.0
    min_daylight_radiation: float = 10.0


class AnomalyDetector:
    def __init__(self, config: IQRConfig | None = None) -> None:
        self._cfg = config or IQRConfig()

    def _compute_fences(self, values: pd.Series) -> tuple[float, float, float]:
        """Return (lower, upper, iqr_eff) for a bucket's value series."""
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr_eff = max(q3 - q1, self._cfg.variance_floor)
        lower = q1 - self._cfg.multiplier * iqr_eff
        upper = q3 + self._cfg.multiplier * iqr_eff
        return lower, upper, iqr_eff

    def flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with is_anomaly and anomaly_score filled per IQR bucket.

        Bucket = (site_key, metric, hour_of_day). Solar night buckets (all values
        below min_daylight_radiation) are skipped — their flags stay False / 0.0.
        """
        out = df.copy()
        out["is_anomaly"] = False
        out["anomaly_score"] = 0.0
        out["_hour"] = pd.to_datetime(out["ts_utc"]).dt.hour

        for keys, group in out.groupby(["site_key", "metric", "_hour"], sort=False):  # type: ignore[assignment]
            _, metric, _ = keys  # type: ignore[misc]
            values: pd.Series = group["value"]  # type: ignore[assignment]

            is_night = bool((values < self._cfg.min_daylight_radiation).all())
            if metric == "solar_radiation" and is_night:
                continue  # night bucket — leave flags at False / 0.0

            lower, upper, iqr_eff = self._compute_fences(values)

            above = values > upper
            below = values < lower

            out.loc[above[above].index, "is_anomaly"] = True
            out.loc[above[above].index, "anomaly_score"] = list(
                (values[above] - upper) / iqr_eff
            )

            out.loc[below[below].index, "is_anomaly"] = True
            out.loc[below[below].index, "anomaly_score"] = list(
                (values[below] - lower) / iqr_eff
            )

        return out.drop(columns=["_hour"])
