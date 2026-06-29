from __future__ import annotations

import pandas as pd

from energy_monitor.clean import DataQualitySummary
from energy_monitor.store import ReadingsStore

# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean_df(n: int = 4) -> pd.DataFrame:
    """Synthetic clean DataFrame matching Cleaner output schema."""
    ts = pd.date_range("2026-05-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "site_key": ["burgos_ph"] * n,
            "ts_utc": ts,
            "ts_local": ts,
            "metric": ["solar_radiation"] * n,
            "value": [100.0, 200.0, 300.0, 400.0][:n],
            "unit": ["W/m2"] * n,
        }
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_load_inserts_correct_row_count() -> None:
    store = ReadingsStore(":memory:")
    store.load(_clean_df(4))
    assert store.row_count() == 4


def test_loaded_table_has_all_schema_columns() -> None:
    store = ReadingsStore(":memory:")
    store.load(_clean_df(2))
    expected = {
        "site_key", "ts_utc", "ts_local", "metric", "value", "unit", "is_anomaly", "anomaly_score"
    }
    assert expected == set(store.column_names())


def test_load_is_idempotent() -> None:
    """Second load replaces the table: row count stays the same, not doubled."""
    store = ReadingsStore(":memory:")
    df = _clean_df(4)
    store.load(df)
    store.load(df)
    assert store.row_count() == 4


def test_anomaly_columns_default_to_null() -> None:
    """is_anomaly and anomaly_score are NULL for a freshly loaded clean frame."""
    store = ReadingsStore(":memory:")
    store.load(_clean_df(2))
    result = store._conn.execute("SELECT is_anomaly, anomaly_score FROM readings").fetchdf()
    assert bool(result["is_anomaly"].isna().all())
    assert bool(result["anomaly_score"].isna().all())


def test_load_dq_persists_summary() -> None:
    """load_dq creates a dq_summary table with one row per site."""
    store = ReadingsStore(":memory:")
    summaries = [
        DataQualitySummary(
            site_key="site_a", rows_in=100, nulls_filled=2, rows_dropped=1, rows_out=99
        ),
        DataQualitySummary(
            site_key="site_b", rows_in=80, nulls_filled=0, rows_dropped=0, rows_out=80
        ),
    ]
    store.load_dq(summaries)
    result = store._conn.execute("SELECT * FROM dq_summary ORDER BY site_key").fetchdf()
    assert len(result) == 2
    assert list(result["site_key"]) == ["site_a", "site_b"]
    rows_out: pd.Series = result.loc[result["site_key"] == "site_a", "rows_out"]  # type: ignore[assignment]
    assert int(rows_out.iloc[0]) == 99
