from __future__ import annotations

import pandas as pd

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
