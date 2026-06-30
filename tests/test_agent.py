from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from energy_monitor.agent import AnalystAgent
from energy_monitor.queries import ReadingsQuery

# ── Fixtures ────────────────────────────────────────────────────────────────

TS = "2026-05-01 {h}:00:00"


@pytest.fixture
def query(tmp_path: Path) -> ReadingsQuery:
    """Small DuckDB with one anomaly and capacity-factor rows for two sites."""
    db = tmp_path / "agent.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute("""
        CREATE TABLE readings (
            site_key VARCHAR, ts_utc TIMESTAMP, ts_local TIMESTAMP,
            metric VARCHAR, value DOUBLE, unit VARCHAR,
            is_anomaly BOOLEAN, anomaly_score DOUBLE
        )
    """)
    conn.executemany(
        "INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("burgos_ph", TS.format(h="10"), TS.format(h="10"),
             "solar_radiation", 500.0, "W/m2", False, 0.0),
            ("burgos_ph", TS.format(h="12"), TS.format(h="12"),
             "solar_radiation", 5000.0, "W/m2", True, 2.5),
            ("sevilla_es", TS.format(h="10"), TS.format(h="10"),
             "solar_radiation", 800.0, "W/m2", False, 0.0),
            ("burgos_ph", TS.format(h="10"), TS.format(h="10"),
             "solar_cf", 0.5, "ratio", False, 0.0),
            ("burgos_ph", TS.format(h="12"), TS.format(h="12"),
             "wind_cf", 0.3, "ratio", False, 0.0),
            ("sevilla_es", TS.format(h="10"), TS.format(h="10"),
             "solar_cf", 0.8, "ratio", False, 0.0),
        ],
    )
    conn.close()
    return ReadingsQuery(db)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use(tool_id: str, name: str, tool_input: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def _response(content: list, stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason)


class _FakeMessages:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list) -> None:
        self.messages = _FakeMessages(responses)


def _agent(query: ReadingsQuery, responses: list) -> AnalystAgent:
    return AnalystAgent(query, client=_FakeClient(responses))


# ── Tool definitions and system prompt ──────────────────────────────────────


def test_build_tools_exposes_the_three_whitelisted_tools(query: ReadingsQuery) -> None:
    agent = AnalystAgent(query, client=_FakeClient([]))
    names = {t["name"] for t in agent._build_tools()}
    assert names == {"avg_metric_by_site", "anomalies_in_window", "compare_generation"}


def test_system_prompt_injects_grounding_context(query: ReadingsQuery) -> None:
    agent = AnalystAgent(query, client=_FakeClient([]))
    prompt = agent._build_system_prompt()
    assert "burgos_ph" in prompt and "sevilla_es" in prompt  # site keys
    assert "2026-05-01" in prompt  # data bounds
    assert "solar_cf" in prompt  # metric names


# ── Tool dispatch ────────────────────────────────────────────────────────────


def test_dispatch_routes_to_avg_metric_by_site(query: ReadingsQuery) -> None:
    agent = AnalystAgent(query, client=_FakeClient([]))
    out = agent._dispatch_tool(
        "avg_metric_by_site",
        {"metric": "solar_radiation", "start": "2026-05-01", "end": "2026-05-01"},
    )
    # sevilla_es avg 800 outranks burgos_ph avg (500+5000)/2 = 2750? burgos higher.
    assert "burgos_ph" in out


def test_dispatch_anomalies_returns_compact_summary(query: ReadingsQuery) -> None:
    agent = AnalystAgent(query, client=_FakeClient([]))
    out = agent._dispatch_tool(
        "anomalies_in_window",
        {"site": "burgos_ph", "metric": "solar_radiation",
         "start": "2026-05-01", "end": "2026-05-01"},
    )
    assert "1" in out  # one anomaly counted


def test_dispatch_empty_result_says_no_data(query: ReadingsQuery) -> None:
    agent = AnalystAgent(query, client=_FakeClient([]))
    out = agent._dispatch_tool(
        "anomalies_in_window",
        {"site": "sevilla_es", "metric": "solar_radiation",
         "start": "2026-05-01", "end": "2026-05-01"},
    )
    assert out == "No data found for the given parameters."


def test_dispatch_unknown_tool_is_an_error_string(query: ReadingsQuery) -> None:
    agent = AnalystAgent(query, client=_FakeClient([]))
    out = agent._dispatch_tool("delete_everything", {})
    assert "unknown tool" in out.lower()


def test_dispatch_bad_date_returns_error_not_raises(query: ReadingsQuery) -> None:
    # The model can emit a non-zero-padded date; this must not crash the loop.
    agent = AnalystAgent(query, client=_FakeClient([]))
    out = agent._dispatch_tool(
        "avg_metric_by_site",
        {"metric": "solar_radiation", "start": "2026-5-1", "end": "2026-05-01"},
    )
    assert "invalid" in out.lower()


def test_dispatch_missing_parameter_returns_error_not_raises(query: ReadingsQuery) -> None:
    agent = AnalystAgent(query, client=_FakeClient([]))
    out = agent._dispatch_tool(
        "avg_metric_by_site", {"start": "2026-05-01", "end": "2026-05-01"}
    )
    assert "invalid" in out.lower()


# ── ask() loop ───────────────────────────────────────────────────────────────


def test_ask_dispatches_tool_then_returns_final_text(query: ReadingsQuery) -> None:
    responses = [
        _response(
            [_tool_use("t1", "compare_generation",
                       {"start": "2026-05-01", "end": "2026-05-01"})],
            "tool_use",
        ),
        _response([_text("sevilla_es has the highest solar capacity factor.")], "end_turn"),
    ]
    agent = _agent(query, responses)
    answer = agent.ask("Compare generation potential across all sites.")
    assert answer == "sevilla_es has the highest solar capacity factor."
    # Two API calls: the tool request, then the grounded answer.
    fake: _FakeClient = agent._client  # type: ignore[assignment]
    assert len(fake.messages.calls) == 2


def test_ask_threads_tool_result_back_to_model(query: ReadingsQuery) -> None:
    responses = [
        _response(
            [_tool_use("t1", "avg_metric_by_site",
                       {"metric": "solar_radiation",
                        "start": "2026-05-01", "end": "2026-05-01"})],
            "tool_use",
        ),
        _response([_text("burgos_ph had the highest average.")], "end_turn"),
    ]
    agent = _agent(query, responses)
    agent.ask("Which site had the highest average solar radiation?")
    fake: _FakeClient = agent._client  # type: ignore[assignment]
    # The second call must carry a tool_result for the tool the model requested.
    second_messages = fake.messages.calls[1]["messages"]
    tool_results = [
        block
        for msg in second_messages
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "t1"


def test_ask_returns_text_without_tool_use(query: ReadingsQuery) -> None:
    responses = [_response([_text("I can only answer from the data tools.")], "end_turn")]
    agent = _agent(query, responses)
    answer = agent.ask("Hello")
    assert answer == "I can only answer from the data tools."


def test_ask_marks_tool_error_so_model_can_self_correct(query: ReadingsQuery) -> None:
    # Model first emits a bad date, then (after the error result) answers in text.
    responses = [
        _response(
            [_tool_use("t1", "avg_metric_by_site",
                       {"metric": "solar_radiation", "start": "2026-5-1", "end": "2026-05-01"})],
            "tool_use",
        ),
        _response([_text("Corrected and answered.")], "end_turn"),
    ]
    agent = _agent(query, responses)
    answer = agent.ask("Which site is sunniest last week?")
    assert answer == "Corrected and answered."  # loop survived the bad date
    fake: _FakeClient = agent._client  # type: ignore[assignment]
    second_messages = fake.messages.calls[1]["messages"]
    tool_results = [
        block
        for msg in second_messages
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0].get("is_error") is True  # flagged for self-correction
