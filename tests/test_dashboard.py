from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from dash import Dash

from energy_monitor.dashboard import (
    _answer_outputs,
    _answer_text,
    _build_app,
    _default_metric,
)
from energy_monitor.queries import ReadingsQuery

# These tests pin the dashboard's default-metric policy. After F4b added the
# capacity-factor metrics (solar_cf, wind_cf, both always is_anomaly=False),
# defaulting to the alphabetically-first metric opened the dashboard on solar_cf,
# which has no anomalies, so the red "x" anomaly markers were absent on load.
# The default must be a measured metric that carries anomaly flags.


def test_default_metric_prefers_solar_radiation() -> None:
    # The real production metric list, alphabetically ordered by list_metrics().
    metrics = ["solar_cf", "solar_radiation", "wind_cf", "wind_speed_100m", "wind_speed_10m"]
    assert _default_metric(metrics) == "solar_radiation"


def test_default_metric_skips_capacity_factors() -> None:
    # solar_radiation absent: must still avoid solar_cf / wind_cf.
    metrics = ["solar_cf", "wind_cf", "wind_speed_100m"]
    assert _default_metric(metrics) == "wind_speed_100m"


def test_default_metric_falls_back_when_only_capacity_factors() -> None:
    metrics = ["solar_cf", "wind_cf"]
    assert _default_metric(metrics) == "solar_cf"


def test_default_metric_none_on_empty() -> None:
    assert _default_metric([]) is None


# ── F6 chat panel ────────────────────────────────────────────────────────────


class _StubAgent:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def ask(self, question: str) -> str:
        self.asked.append(question)
        return f"answer to: {question}"


def test_answer_text_ignores_empty_input() -> None:
    agent = _StubAgent()
    assert _answer_text(agent, "   ") == ""
    assert agent.asked == []  # the agent is never called on blank input


def test_answer_text_calls_agent() -> None:
    agent = _StubAgent()
    assert _answer_text(agent, "Which site is sunniest?") == \
        "answer to: Which site is sunniest?"
    assert agent.asked == ["Which site is sunniest?"]


def test_answer_outputs_clears_waiting_indicator() -> None:
    # The server callback must return the answer AND the values that reset the
    # waiting indicator the clientside callback raised: blank status, button
    # re-enabled, label restored to "Ask".
    agent = _StubAgent()
    answer, status, disabled, label = _answer_outputs(agent, "Which site is sunniest?")
    assert answer == "answer to: Which site is sunniest?"
    assert status == ""
    assert disabled is False
    assert label == "Ask"


class _BoomAgent:
    def ask(self, question: str) -> str:
        raise RuntimeError("API key invalid")


def test_answer_text_handles_agent_error_gracefully() -> None:
    # An agent/API failure must surface as a message, never crash the callback.
    out = _answer_text(_BoomAgent(), "anything")
    assert "could not" in out.lower()


@pytest.fixture
def dashboard_db(tmp_path: Path) -> Path:
    db = tmp_path / "dash.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute("""
        CREATE TABLE readings (
            site_key VARCHAR, ts_utc TIMESTAMP, ts_local TIMESTAMP,
            metric VARCHAR, value DOUBLE, unit VARCHAR,
            is_anomaly BOOLEAN, anomaly_score DOUBLE
        )
    """)
    conn.execute(
        "INSERT INTO readings VALUES "
        "('burgos_ph', '2026-05-01 10:00:00', '2026-05-01 10:00:00', "
        "'solar_radiation', 500.0, 'W/m2', FALSE, 0.0)"
    )
    conn.execute("""
        CREATE TABLE dq_summary (
            site_key VARCHAR, rows_in INTEGER, nulls_filled INTEGER,
            rows_dropped INTEGER, rows_out INTEGER
        )
    """)
    conn.execute("INSERT INTO dq_summary VALUES ('burgos_ph', 24, 0, 0, 24)")
    conn.close()
    return db


def test_build_app_without_agent(dashboard_db: Path) -> None:
    app = _build_app(ReadingsQuery(dashboard_db), agent=None)
    assert isinstance(app, Dash)


def test_build_app_with_agent(dashboard_db: Path) -> None:
    app = _build_app(ReadingsQuery(dashboard_db), agent=_StubAgent())
    assert isinstance(app, Dash)
