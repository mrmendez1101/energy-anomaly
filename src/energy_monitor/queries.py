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

    # ── F6 agent query methods ───────────────────────────────────────────────
    # These back the agent's whitelisted tools. The metric and site values come
    # from the model, so they are bound as ? parameters (never interpolated) so
    # the agent cannot inject SQL. C# analogy: parameterised repository methods.

    def avg_metric_by_site(
        self, metric: str, start: str | date, end: str | date
    ) -> pd.DataFrame:
        """Average a metric per site over a window, ranked highest first."""
        return self._conn.execute(
            """
            SELECT site_key, AVG(value) AS avg_value
            FROM readings
            WHERE metric = ?
            AND ts_utc::DATE BETWEEN ? AND ?
            GROUP BY site_key
            ORDER BY avg_value DESC
            """,
            [metric, _validated_date(start, "start"), _validated_date(end, "end")],
        ).fetchdf()

    def anomalies_in_window(
        self, site: str, metric: str, start: str | date, end: str | date
    ) -> pd.DataFrame:
        """Return the flagged anomaly rows for one site and metric in a window."""
        return self._conn.execute(
            """
            SELECT * FROM readings
            WHERE site_key = ?
            AND metric = ?
            AND is_anomaly = TRUE
            AND ts_utc::DATE BETWEEN ? AND ?
            ORDER BY ts_utc
            """,
            [site, metric, _validated_date(start, "start"), _validated_date(end, "end")],
        ).fetchdf()

    def compare_generation(self, start: str | date, end: str | date) -> pd.DataFrame:
        """Average each site's solar_cf and wind_cf over a window, one row per site.

        Uses DuckDB's AVG(...) FILTER to pivot the two capacity-factor metrics
        into columns. These metrics are produced by F4b.
        """
        return self._conn.execute(
            """
            SELECT
                site_key,
                AVG(value) FILTER (WHERE metric = 'solar_cf') AS avg_solar_cf,
                AVG(value) FILTER (WHERE metric = 'wind_cf') AS avg_wind_cf
            FROM readings
            WHERE metric IN ('solar_cf', 'wind_cf')
            AND ts_utc::DATE BETWEEN ? AND ?
            GROUP BY site_key
            ORDER BY site_key
            """,
            [_validated_date(start, "start"), _validated_date(end, "end")],
        ).fetchdf()
