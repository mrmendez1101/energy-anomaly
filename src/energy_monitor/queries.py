from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validated_date(value: str | date, label: str) -> str:
    s = str(value)[:10]
    if not _DATE_RE.match(s):
        raise ValueError(f"Invalid {label}: {value!r} — expected YYYY-MM-DD")
    return s


class ReadingsQuery:
    """Read-only query interface over the DuckDB warehouse.

    Opened with read_only=True so the dashboard and pipeline can run concurrently
    without file-lock conflicts on Windows.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._conn = duckdb.connect(str(db_path), read_only=True)

    def list_sites(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT site_key FROM readings ORDER BY site_key"
        ).fetchall()
        return [r[0] for r in rows]

    def list_metrics(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT metric FROM readings ORDER BY metric"
        ).fetchall()
        return [r[0] for r in rows]

    def date_bounds(self) -> tuple[date, date]:
        r = self._conn.execute(
            "SELECT MIN(ts_utc)::DATE, MAX(ts_utc)::DATE FROM readings"
        ).fetchone()
        assert r is not None
        return r[0], r[1]  # type: ignore[return-value]

    def readings(
        self,
        sites: list[str],
        metrics: list[str],
        start: str | date,
        end: str | date,
    ) -> pd.DataFrame:
        """Return filtered readings including is_anomaly and anomaly_score columns."""
        site_csv = ", ".join(f"'{s}'" for s in sites)
        metric_csv = ", ".join(f"'{m}'" for m in metrics)
        return self._conn.execute(
            f"""
            SELECT * FROM readings
            WHERE site_key IN ({site_csv})
            AND metric IN ({metric_csv})
            AND ts_utc::DATE BETWEEN ? AND ?
            ORDER BY site_key, metric, ts_utc
            """,
            [_validated_date(start, "start"), _validated_date(end, "end")],
        ).fetchdf()

    def dq_summary(self) -> pd.DataFrame:
        return self._conn.execute("SELECT * FROM dq_summary").fetchdf()
