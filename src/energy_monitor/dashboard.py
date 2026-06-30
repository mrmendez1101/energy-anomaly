from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from energy_monitor.agent import AnalystAgent
from energy_monitor.queries import ReadingsQuery

_METRIC_LABELS: dict[str, str] = {
    "solar_radiation": "Solar Radiation (W/m²)",
    "wind_speed_10m": "Wind Speed 10 m (m/s)",
    "wind_speed_100m": "Wind Speed 100 m (m/s)",
    "solar_cf": "Solar Capacity Factor (0–1)",
    "wind_cf": "Wind Capacity Factor (0–1)",
}

# Capacity-factor metrics (F4b) are comparison views and never carry anomaly
# flags, so they must not be the dashboard's opening metric, otherwise the red
# "x" anomaly markers are absent on load.
_CAPACITY_FACTOR_METRICS = frozenset({"solar_cf", "wind_cf"})
_PREFERRED_DEFAULT_METRIC = "solar_radiation"


def _default_metric(metrics: list[str]) -> str | None:
    """Pick the metric the dashboard opens on.

    Prefer solar_radiation (a measured metric that carries anomaly flags) so the
    anomaly markers are visible on first load. list_metrics() orders metrics
    alphabetically, which after F4b puts solar_cf first, so we cannot just take
    metrics[0]. C# analogy: a small policy/strategy method.
    """
    if not metrics:
        return None
    if _PREFERRED_DEFAULT_METRIC in metrics:
        return _PREFERRED_DEFAULT_METRIC
    measured = [m for m in metrics if m not in _CAPACITY_FACTOR_METRICS]
    return measured[0] if measured else metrics[0]


def _answer_text(agent: Any, question: str | None) -> str:
    """Pure callback body: blank input is a no-op, otherwise ask the agent.

    Kept separate from the Dash callback so the branching logic is unit-testable
    without spinning up the app. C# analogy: a thin service method behind a thin
    controller action.
    """
    if not question or not question.strip():
        return ""
    try:
        return agent.ask(question.strip())
    except Exception as exc:  # UI boundary: the bonus must never crash the core tool
        return f"Sorry, the agent could not answer that right now ({exc})."


def _answer_outputs(agent: Any, question: str | None) -> tuple[str, str, bool, str]:
    """Server-callback body: the answer plus the values that *clear* the waiting
    indicator the clientside callback raised.

    Returns (answer, status_text, ask_disabled, ask_label). Status is blanked and
    the button restored/re-enabled once the answer lands. Kept as a pure helper so
    the indicator-reset contract is unit-testable without a browser. C# analogy: a
    DTO returned by the service method, assembled by the thin controller.
    """
    return _answer_text(agent, question), "", False, "Ask"


def _chat_panel(enabled: bool) -> html.Div:
    """The chat input, or a disabled notice when no agent is configured."""
    if not enabled:
        return html.Div(
            "Agent disabled: set ANTHROPIC_API_KEY and restart to enable the chat.",
            style={"color": "#888", "fontStyle": "italic"},
        )
    return html.Div([
        dcc.Input(
            id="chat-input",
            type="text",
            placeholder="e.g. Which site had the highest average solar radiation?",
            debounce=True,
            style={"width": "70%", "padding": "6px"},
        ),
        html.Button("Ask", id="chat-ask", n_clicks=0, style={"marginLeft": "8px"}),
        # Waiting indicator: a clientside callback raises this the instant Ask is
        # clicked; the server callback blanks it when the answer returns.
        html.Div(
            id="chat-status",
            style={"marginTop": "0.5rem", "color": "#888", "fontStyle": "italic"},
        ),
        # dcc.Loading tracks chat-answer's loading state client-side and overlays a
        # spinner while the (blocking) agent.ask() server callback runs.
        dcc.Loading(
            id="chat-loading",
            type="dot",
            children=html.Div(
                id="chat-answer",
                style={"marginTop": "0.75rem", "whiteSpace": "pre-wrap"},
            ),
        ),
    ])


def _build_app(q: ReadingsQuery, agent: Any | None = None) -> Dash:
    sites = q.list_sites()
    metrics = q.list_metrics()
    lo, hi = q.date_bounds()

    metric_opts = [{"label": _METRIC_LABELS.get(m, m), "value": m} for m in metrics]

    app = Dash(__name__, title="Energy Monitor")

    app.layout = html.Div(
        style={"fontFamily": "sans-serif", "maxWidth": "1200px", "margin": "0 auto"},
        children=[
            html.H2("Energy Monitor — Anomaly Dashboard"),

            # ── Controls ──────────────────────────────────────────────────────
            html.Div(
                style={
                    "display": "flex", "gap": "1.5rem", "flexWrap": "wrap", "marginBottom": "1rem"
                },
                children=[
                    html.Div([
                        html.Label("Sites", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="site-select",
                            options=[{"label": s, "value": s} for s in sites],
                            value=sites,
                            multi=True,
                            style={"minWidth": "300px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Metric", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="metric-select",
                            options=metric_opts,
                            value=_default_metric(metrics),
                            style={"minWidth": "220px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Date range", style={"fontWeight": "bold"}),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=str(lo),
                            max_date_allowed=str(hi),
                            start_date=str(lo),
                            end_date=str(hi),
                        ),
                    ]),
                ],
            ),

            # ── Chart ─────────────────────────────────────────────────────────
            dcc.Graph(id="time-series", style={"height": "480px"}),

            # ── Data Quality ──────────────────────────────────────────────────
            html.H3("Data Quality", style={"marginTop": "1.5rem"}),
            html.Div(id="dq-table"),

            # ── Analyst chat (F6 bonus) ───────────────────────────────────────
            html.H3("Ask the Analyst Agent", style={"marginTop": "1.5rem"}),
            _chat_panel(agent is not None),
        ],
    )

    dq_df = q.dq_summary()

    @app.callback(
        Output("time-series", "figure"),
        [
            Input("site-select", "value"),
            Input("metric-select", "value"),
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
        ],
    )
    def update_chart(
        selected_sites: list[str] | None,
        metric: str | None,
        start: str | None,
        end: str | None,
    ) -> go.Figure:
        fig = go.Figure()
        if not selected_sites or not metric or not start or not end:
            return fig

        df = q.readings(selected_sites, [metric], start, end)
        if df.empty:
            return fig

        for site in selected_sites:
            site_df = df[df["site_key"] == site]
            is_anom_col: pd.Series = site_df["is_anomaly"]  # type: ignore[assignment]
            is_anom = is_anom_col.fillna(False)
            normal = site_df[~is_anom]
            anomalies: pd.DataFrame = site_df[is_anom]  # type: ignore[assignment]

            fig.add_trace(
                go.Scatter(
                    x=normal["ts_utc"],
                    y=normal["value"],
                    mode="lines",
                    name=site,
                    hovertemplate="%{x}<br>%{y:.1f}<extra>" + site + "</extra>",
                )
            )
            if not anomalies.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies["ts_utc"],
                        y=anomalies["value"],
                        mode="markers",
                        marker={"symbol": "x", "size": 10, "color": "red", "line": {"width": 2}},
                        name=f"{site} anomaly",
                        hovertemplate="%{x}<br>%{y:.1f}<extra>" + site + " anomaly</extra>",
                    )
                )

        fig.update_layout(
            xaxis_title="Time (UTC)",
            yaxis_title=_METRIC_LABELS.get(metric, metric),
            legend_title="Site",
            hovermode="x unified",
            margin={"l": 60, "r": 20, "t": 30, "b": 60},
        )
        return fig

    @app.callback(Output("dq-table", "children"), Input("site-select", "value"))
    def update_dq(_: object) -> html.Table:
        header = html.Tr([
            html.Th(c, style={"textAlign": "left", "padding": "4px 12px"})
            for c in ["Site", "Rows in", "Nulls filled", "Rows dropped", "Rows out"]
        ])
        rows = [
            html.Tr([
                html.Td(str(row["site_key"]), style={"padding": "2px 12px"}),
                html.Td(str(row["rows_in"]), style={"padding": "2px 12px"}),
                html.Td(str(row["nulls_filled"]), style={"padding": "2px 12px"}),
                html.Td(str(row["rows_dropped"]), style={"padding": "2px 12px"}),
                html.Td(str(row["rows_out"]), style={"padding": "2px 12px"}),
            ])
            for _, row in dq_df.iterrows()
        ]
        return html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"borderCollapse": "collapse", "fontSize": "0.9rem"},
        )

    # Only register the chat callbacks when an agent exists; otherwise the
    # chat-input/chat-ask components are absent and the callbacks have nothing to bind.
    if agent is not None:
        # Instant, browser-side feedback the moment Ask is clicked: show "Analyzing…"
        # and disable/relabel the button. A synchronous server callback cannot do this
        # itself — the browser only re-renders when the call returns — so we raise the
        # indicator clientside and let the server callback below clear it. Empty input
        # is a no-op (no_update) so a blank click doesn't flash the indicator.
        app.clientside_callback(
            """
            function(n_clicks, question) {
                const noUpdate = window.dash_clientside.no_update;
                if (!question || !question.trim()) {
                    return [noUpdate, noUpdate, noUpdate];
                }
                return ['Analyzing…', true, 'Asking…'];
            }
            """,
            Output("chat-status", "children"),
            Output("chat-ask", "disabled"),
            Output("chat-ask", "children"),
            Input("chat-ask", "n_clicks"),
            State("chat-input", "value"),
            prevent_initial_call=True,
        )

        # Server callback: runs the (blocking) agent, then clears the indicator the
        # clientside callback raised. chat-status/chat-ask.disabled/chat-ask.children
        # are written by both callbacks, so the server side declares them
        # allow_duplicate=True (Dash requires prevent_initial_call with it).
        @app.callback(
            Output("chat-answer", "children"),
            Output("chat-status", "children", allow_duplicate=True),
            Output("chat-ask", "disabled", allow_duplicate=True),
            Output("chat-ask", "children", allow_duplicate=True),
            Input("chat-ask", "n_clicks"),
            State("chat-input", "value"),
            prevent_initial_call=True,
        )
        def answer_question(_: int, question: str | None) -> tuple[str, str, bool, str]:
            return _answer_outputs(agent, question)

    return app


def _build_agent(q: ReadingsQuery) -> AnalystAgent | None:
    """Construct the agent only if an API key is present, else degrade gracefully."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return AnalystAgent(q)


def run_dashboard(db_path: str | Path = "data/warehouse.duckdb") -> None:
    q = ReadingsQuery(db_path)
    agent = _build_agent(q)
    app = _build_app(q, agent=agent)
    status = "enabled" if agent is not None else "disabled (no ANTHROPIC_API_KEY)"
    print(f"Dashboard running at http://127.0.0.1:8050/  (agent {status})")
    app.run(debug=False)
