from __future__ import annotations

import pandas as pd

from energy_monitor.anomaly import AnomalyDetector, IQRConfig

# ── Helper ────────────────────────────────────────────────────────────────────


def _bucket_df(
    n: int = 30,
    hour: int = 12,
    metric: str = "solar_radiation",
    values: list[float] | None = None,
) -> pd.DataFrame:
    """Synthetic DataFrame with n readings all at the same hour of day."""
    if values is None:
        values = [500.0] * n
    ts = pd.date_range(f"2026-05-01 {hour:02d}:00", periods=n, freq="24h")
    return pd.DataFrame(
        {
            "site_key": ["burgos_ph"] * n,
            "ts_utc": ts,
            "ts_local": ts,
            "metric": [metric] * n,
            "value": values,
            "unit": ["W/m2"] * n,
        }
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_injected_spike_is_flagged() -> None:
    """A value 10x the median in a noon-hour bucket must be flagged."""
    values = [500.0] * 29 + [5000.0]
    df = _bucket_df(30, hour=12, values=values)
    result = AnomalyDetector(IQRConfig()).flag(df)
    assert bool(result.iloc[-1]["is_anomaly"])


def test_night_solar_zeros_not_flagged() -> None:
    """All-zero solar bucket at midnight is a night bucket — no flags allowed."""
    df = _bucket_df(30, hour=0, values=[0.0] * 30)
    result = AnomalyDetector(IQRConfig()).flag(df)
    assert not bool(result["is_anomaly"].any())


def test_zero_variance_bucket_does_not_over_flag() -> None:
    """Constant noon bucket: variance floor keeps the fence from collapsing to a point."""
    df = _bucket_df(30, hour=12, values=[500.0] * 30)
    result = AnomalyDetector(IQRConfig()).flag(df)
    assert not bool(result["is_anomaly"].any())


def test_anomaly_score_sign() -> None:
    """Value above upper fence has positive score; value below lower fence has negative score."""
    values = [500.0] * 28 + [5000.0, 0.0]
    df = _bucket_df(30, hour=12, values=values)
    result = AnomalyDetector(IQRConfig()).flag(df)
    high_score = float(result.iloc[-2]["anomaly_score"])
    low_score = float(result.iloc[-1]["anomaly_score"])
    assert high_score > 0, f"expected positive score for spike, got {high_score}"
    assert low_score < 0, f"expected negative score for drop, got {low_score}"
