from __future__ import annotations

from typing import Any

import anthropic
import pandas as pd

from energy_monitor.queries import ReadingsQuery

# C# analogy: a service that orchestrates an AI call with function-calling.
# AnalystAgent maps a plain-English question to one of three whitelisted SQL
# tools, runs it, and lets the model phrase the grounded result. It never writes
# SQL: the model only picks a tool name and supplies parameters.

_MAX_TOOL_TURNS = 5  # guard against a runaway tool-use loop


class AnalystAgent:
    """Grounded conversational agent over three whitelisted ReadingsQuery tools."""

    def __init__(
        self,
        query: ReadingsQuery,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._query = query
        self._model = model
        # client is injectable for offline tests; otherwise build the real SDK
        # client, which reads ANTHROPIC_API_KEY from the environment when api_key
        # is None. C# analogy: constructor injection of a dependency.
        # Typed Any so the duck-typed fake client used in tests type-checks the same
        # as the real SDK client, and content-block access in the loop stays uniform.
        self._client: Any = (
            client if client is not None else anthropic.Anthropic(api_key=api_key)
        )
        self._system = self._build_system_prompt()
        self._tools = self._build_tools()

    # ── Tool definitions ─────────────────────────────────────────────────────

    def _build_tools(self) -> list[dict]:
        date = {"type": "string", "description": "Date as YYYY-MM-DD"}
        return [
            {
                "name": "avg_metric_by_site",
                "description": (
                    "Average one metric per site over a date window, ranked highest "
                    "first. Use for 'which site had the highest average X' questions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "description": "Metric name"},
                        "start": date,
                        "end": date,
                    },
                    "required": ["metric", "start", "end"],
                },
            },
            {
                "name": "anomalies_in_window",
                "description": (
                    "List flagged anomalies for one site and metric in a date window. "
                    "Use for 'were there anomalous X readings at site Y' questions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "site": {"type": "string", "description": "Site key"},
                        "metric": {"type": "string", "description": "Metric name"},
                        "start": date,
                        "end": date,
                    },
                    "required": ["site", "metric", "start", "end"],
                },
            },
            {
                "name": "compare_generation",
                "description": (
                    "Compare average solar and wind capacity factors across all sites "
                    "over a date window. Use for 'compare generation potential' questions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"start": date, "end": date},
                    "required": ["start", "end"],
                },
            },
        ]

    # ── System prompt (grounding) ────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        lo, hi = self._query.date_bounds()
        sites = ", ".join(self._query.list_sites())
        metrics = ", ".join(self._query.list_metrics())
        return (
            "You are an analyst assistant for a renewable-energy monitoring tool. "
            "Answer ONLY from the results of the provided tools. Never invent numbers "
            "or sites. If a tool returns no data, say there is no data for that window.\n\n"
            f"The data covers {lo} to {hi} (a fixed historical window, not today). "
            "Resolve relative dates like 'last week' or 'the past month' against that "
            "window, not the current calendar date.\n"
            f"Available sites: {sites}.\n"
            f"Available metrics: {metrics}.\n"
            "Pick exactly one tool per question, then phrase the result in one or two "
            "plain sentences with the relevant numbers."
        )

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    def _dispatch_tool(self, tool_name: str, tool_input: dict) -> str:
        """Route a tool call to its query method, returning a result string."""
        return self._run_tool(tool_name, tool_input)[0]

    def _run_tool(self, tool_name: str, tool_input: dict) -> tuple[str, bool]:
        """Run a tool and return (result_text, is_error).

        Bad parameters from the model (a non-zero-padded date, a missing key) must
        not crash the loop: we catch them and return an error string flagged so the
        model can self-correct on the next turn. C#: a try/catch at the dispatch
        boundary that converts an exception into a typed failure result.
        """
        try:
            if tool_name == "avg_metric_by_site":
                df = self._query.avg_metric_by_site(
                    tool_input["metric"], tool_input["start"], tool_input["end"]
                )
                return self._format_avg(df), False
            if tool_name == "anomalies_in_window":
                df = self._query.anomalies_in_window(
                    tool_input["site"], tool_input["metric"],
                    tool_input["start"], tool_input["end"],
                )
                return self._format_anomalies(df), False
            if tool_name == "compare_generation":
                df = self._query.compare_generation(
                    tool_input["start"], tool_input["end"]
                )
                return self._format_generation(df), False
            return f"Unknown tool: {tool_name}", True
        except (ValueError, KeyError) as exc:
            return f"Invalid parameters: {exc}", True

    @staticmethod
    def _no_data() -> str:
        return "No data found for the given parameters."

    def _format_avg(self, df: pd.DataFrame) -> str:
        if df.empty:
            return self._no_data()
        rows = [
            f"{row['site_key']}: {float(row['avg_value']):.2f}"
            for row in df.to_dict("records")
        ]
        return "Average per site (highest first): " + "; ".join(rows)

    def _format_anomalies(self, df: pd.DataFrame) -> str:
        # Compact summary, not the full frame: count plus up to three timestamps,
        # so we don't feed a large table back into the model's context.
        if df.empty:
            return self._no_data()
        stamps = [str(ts) for ts in df["ts_utc"].head(3)]
        more = "" if len(df) <= 3 else f" (showing first 3 of {len(df)})"
        return f"{len(df)} anomaly reading(s) found at: " + ", ".join(stamps) + more

    def _format_generation(self, df: pd.DataFrame) -> str:
        if df.empty:
            return self._no_data()
        rows = [
            f"{row['site_key']}: solar_cf {float(row['avg_solar_cf']):.3f}, "
            f"wind_cf {float(row['avg_wind_cf']):.3f}"
            for row in df.to_dict("records")
        ]
        return "Generation potential per site: " + "; ".join(rows)

    # ── Single-turn ask loop ──────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """Answer one question, resolving any tool calls, and return grounded text."""
        messages: list[dict] = [{"role": "user", "content": question}]
        for _ in range(_MAX_TOOL_TURNS):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=self._system,
                tools=self._tools,
                messages=messages,
            )
            if response.stop_reason != "tool_use":
                return self._final_text(response)

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    content, is_error = self._run_tool(block.name, dict(block.input))
                    result: dict = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                    }
                    if is_error:
                        # Let the model see the failure and retry with fixed params.
                        result["is_error"] = True
                    tool_results.append(result)
            messages.append({"role": "user", "content": tool_results})

        return "Sorry, I could not resolve that question within the tool limit."

    @staticmethod
    def _final_text(response: Any) -> str:
        parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()
