from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

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


def _build_app(q: ReadingsQuery) -> Dash:
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

    return app


def run_dashboard(db_path: str | Path = "data/warehouse.duckdb") -> None:
    q = ReadingsQuery(db_path)
    app = _build_app(q)
    print("Dashboard running at http://127.0.0.1:8050/")
    app.run(debug=False)
