from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# C#: class ReadingsStore : IReadingsRepository — repository pattern, owns all DuckDB access


class ReadingsStore:
    """Drop-and-load repository for the DuckDB readings table."""

    def __init__(self, db_path: str | Path) -> None:
        self._conn = duckdb.connect(str(db_path))

    def load(self, df: pd.DataFrame) -> None:
        """Drop the readings table and reload it from a clean DataFrame.

        Adds NULL is_anomaly / anomaly_score columns if not already present,
        so the schema is forward-compatible when F4 writes flags back.
        """
        data = df.copy()
        if "is_anomaly" not in data.columns:
            data["is_anomaly"] = pd.array([pd.NA] * len(data), dtype="boolean")
        if "anomaly_score" not in data.columns:
            data["anomaly_score"] = pd.array([pd.NA] * len(data), dtype="Float64")

        self._conn.execute("DROP TABLE IF EXISTS readings")
        self._conn.register("_readings_tmp", data)
        self._conn.execute("CREATE TABLE readings AS SELECT * FROM _readings_tmp")
        self._conn.unregister("_readings_tmp")

    def row_count(self) -> int:
        """Return the number of rows in the readings table."""
        result = self._conn.execute("SELECT COUNT(*) FROM readings").fetchone()
        return int(result[0])  # type: ignore[index]

    def column_names(self) -> list[str]:
        """Return column names of the readings table."""
        return self._conn.execute("DESCRIBE readings").fetchdf()["column_name"].tolist()

    def load_dq(self, summaries: list) -> None:
        """Persist data-quality summaries as a dq_summary table."""
        rows = [s.model_dump() for s in summaries]
        df = pd.DataFrame(rows)
        self._conn.execute("DROP TABLE IF EXISTS dq_summary")
        self._conn.register("_dq_tmp", df)
        self._conn.execute("CREATE TABLE dq_summary AS SELECT * FROM _dq_tmp")
        self._conn.unregister("_dq_tmp")

    def export_parquet(self, path: Path) -> None:
        """Export the readings table to a Parquet file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn.execute(f"COPY readings TO '{path}' (FORMAT PARQUET)")
