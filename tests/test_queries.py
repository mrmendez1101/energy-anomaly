from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from energy_monitor.queries import ReadingsQuery

# ── Fixture ───────────────────────────────────────────────────────────────────

TS = "2026-05-01 {h}:00:00"


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Minimal DuckDB with readings + dq_summary for unit tests."""
    db = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db))

    conn.execute("""
        CREATE TABLE readings (
            site_key VARCHAR,
            ts_utc TIMESTAMP,
            ts_local TIMESTAMP,
            metric VARCHAR,
            value DOUBLE,
            unit VARCHAR,
            is_anomaly BOOLEAN,
            anomaly_score DOUBLE
        )
    """)
    conn.executemany(
        "INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("site_a", TS.format(h="10"), TS.format(h="10"),
             "solar_radiation", 500.0, "W/m2", False, 0.0),
            ("site_a", TS.format(h="11"), TS.format(h="11"),
             "solar_radiation", 600.0, "W/m2", False, 0.0),
            ("site_a", TS.format(h="12"), TS.format(h="12"),
             "solar_radiation", 5000.0, "W/m2", True, 2.5),
            ("site_b", TS.format(h="10"), TS.format(h="10"),
             "solar_radiation", 400.0, "W/m2", False, 0.0),
            ("site_b", TS.format(h="11"), TS.format(h="11"),
             "solar_radiation", 450.0, "W/m2", False, 0.0),
            ("site_b", TS.format(h="12"), TS.format(h="12"),
             "wind_speed_10m", 5.0, "m/s", False, 0.0),
            # Derived capacity-factor rows (F4b), needed by compare_generation.
            # solar_cf: site_a avg 0.6, site_b avg 0.45; wind_cf: site_a 0.3, site_b 0.15.
            ("site_a", TS.format(h="10"), TS.format(h="10"),
             "solar_cf", 0.5, "ratio", False, 0.0),
            ("site_a", TS.format(h="11"), TS.format(h="11"),
             "solar_cf", 0.6, "ratio", False, 0.0),
            ("site_a", TS.format(h="12"), TS.format(h="12"),
             "solar_cf", 0.7, "ratio", False, 0.0),
            ("site_a", TS.format(h="10"), TS.format(h="10"),
             "wind_cf", 0.2, "ratio", False, 0.0),
            ("site_a", TS.format(h="11"), TS.format(h="11"),
             "wind_cf", 0.3, "ratio", False, 0.0),
            ("site_a", TS.format(h="12"), TS.format(h="12"),
             "wind_cf", 0.4, "ratio", False, 0.0),
            ("site_b", TS.format(h="10"), TS.format(h="10"),
             "solar_cf", 0.4, "ratio", False, 0.0),
            ("site_b", TS.format(h="11"), TS.format(h="11"),
             "solar_cf", 0.45, "ratio", False, 0.0),
            ("site_b", TS.format(h="12"), TS.format(h="12"),
             "solar_cf", 0.5, "ratio", False, 0.0),
            ("site_b", TS.format(h="10"), TS.format(h="10"),
             "wind_cf", 0.1, "ratio", False, 0.0),
            ("site_b", TS.format(h="11"), TS.format(h="11"),
             "wind_cf", 0.15, "ratio", False, 0.0),
            ("site_b", TS.format(h="12"), TS.format(h="12"),
             "wind_cf", 0.2, "ratio", False, 0.0),
        ],
    )

    conn.execute("""
        CREATE TABLE dq_summary (
            site_key VARCHAR,
            rows_in INTEGER,
            nulls_filled INTEGER,
            rows_dropped INTEGER,
            rows_out INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO dq_summary VALUES (?, ?, ?, ?, ?)",
        [
            ("site_a", 2160, 0, 0, 2160),
            ("site_b", 2160, 2, 1, 2157),
        ],
    )

    conn.close()
    return db


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_list_sites_returns_all_sites(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    assert set(q.list_sites()) == {"site_a", "site_b"}


def test_list_metrics_returns_all_metrics(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    assert set(q.list_metrics()) == {
        "solar_radiation", "wind_speed_10m", "solar_cf", "wind_cf"
    }


def test_date_bounds_correct(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    lo, hi = q.date_bounds()
    assert str(lo) == "2026-05-01"
    assert str(hi) == "2026-05-01"


def test_readings_filters_by_site(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    df = q.readings(["site_a"], ["solar_radiation"], "2026-05-01", "2026-05-01")
    assert set(df["site_key"].unique()) == {"site_a"}


def test_readings_includes_anomaly_columns(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    df = q.readings(["site_a"], ["solar_radiation"], "2026-05-01", "2026-05-01")
    assert "is_anomaly" in df.columns
    assert "anomaly_score" in df.columns


def test_anomaly_row_is_flagged(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    df = q.readings(["site_a"], ["solar_radiation"], "2026-05-01", "2026-05-01")
    flagged = df[df["is_anomaly"]]
    assert len(flagged) == 1
    value_col: pd.Series = flagged["value"]  # type: ignore[assignment]
    assert float(value_col.iloc[0]) == 5000.0


def test_dq_summary_returns_per_site_rows(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    df = q.dq_summary()
    assert len(df) == 2
    assert set(df["site_key"].unique()) == {"site_a", "site_b"}


def test_readings_raises_on_partial_date(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        q.readings(["site_a"], ["solar_radiation"], "2026-05", "2026-05-01")


def test_readings_raises_on_empty_date(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        q.readings(["site_a"], ["solar_radiation"], "", "2026-05-01")


# ── F6 agent query methods ──────────────────────────────────────────────────


def test_avg_metric_by_site_ranks_descending(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    df = q.avg_metric_by_site("solar_radiation", "2026-05-01", "2026-05-01")
    # site_a avg = (500+600+5000)/3 ≈ 2033.3, site_b avg = (400+450)/2 = 425.
    assert list(df["site_key"]) == ["site_a", "site_b"]
    avg_col: pd.Series = df["avg_value"]  # type: ignore[assignment]
    assert round(float(avg_col.iloc[0]), 1) == 2033.3
    assert float(avg_col.iloc[1]) == 425.0


def test_avg_metric_by_site_binds_metric_safely(tmp_db: Path) -> None:
    # A SQL-injection style metric string must bind as a literal, returning no rows.
    q = ReadingsQuery(tmp_db)
    df = q.avg_metric_by_site("'; DROP TABLE readings; --", "2026-05-01", "2026-05-01")
    assert df.empty
    # Table must still exist and be queryable afterwards.
    assert set(q.list_sites()) == {"site_a", "site_b"}


def test_anomalies_in_window_returns_only_flagged(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    df = q.anomalies_in_window("site_a", "solar_radiation", "2026-05-01", "2026-05-01")
    assert len(df) == 1
    value_col: pd.Series = df["value"]  # type: ignore[assignment]
    assert float(value_col.iloc[0]) == 5000.0


def test_anomalies_in_window_empty_when_none(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    df = q.anomalies_in_window("site_b", "solar_radiation", "2026-05-01", "2026-05-01")
    assert df.empty


def test_compare_generation_averages_capacity_factors(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    df = q.compare_generation("2026-05-01", "2026-05-01")
    assert list(df["site_key"]) == ["site_a", "site_b"]
    by_site = df.set_index("site_key")
    assert round(float(by_site.loc["site_a", "avg_solar_cf"]), 2) == 0.60
    assert round(float(by_site.loc["site_a", "avg_wind_cf"]), 2) == 0.30
    assert round(float(by_site.loc["site_b", "avg_solar_cf"]), 2) == 0.45
    assert round(float(by_site.loc["site_b", "avg_wind_cf"]), 2) == 0.15


def test_compare_generation_empty_window_returns_empty(tmp_db: Path) -> None:
    q = ReadingsQuery(tmp_db)
    df = q.compare_generation("2025-01-01", "2025-01-02")
    assert df.empty
