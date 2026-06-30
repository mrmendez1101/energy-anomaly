# Energy Anomaly Monitor

ETL pipeline + dashboard that pulls 30 days of hourly weather for three renewable sites,
cleans and stores it in DuckDB, flags anomalies (IQR), and serves an interactive Dash
dashboard with an optional grounded chat agent.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python      | >= 3.12 | 3.13 used in dev |
| `uv`        | latest  | dependency + venv manager ([install](https://docs.astral.sh/uv/)) |
| Network     | —       | only for the first data fetch; a snapshot is committed so offline runs work |

No database server and no API key are required for the core pipeline and dashboard.
An `ANTHROPIC_API_KEY` is only needed for the optional chat agent (Step 6).

---

## 2. Install

From the project root:

```bash
uv sync
```

This creates `.venv/` and installs all runtime + dev dependencies from `pyproject.toml`.
Run every command below through `uv run` so it uses that environment.

---

## 3. Generate the data (run the ETL pipeline)

```bash
uv run python -m energy_monitor
```

This is the entry point in [`src/energy_monitor/__main__.py`](src/energy_monitor/__main__.py).
It runs the full pipeline in order:

| Stage | Module | What it does |
|-------|--------|--------------|
| F1 Ingest | `ingest.py` | Fetch hourly `shortwave_radiation`, `wind_speed_10m`, `wind_speed_100m` per site from the Open-Meteo Archive API |
| F2 Clean | `clean.py` | Reshape to long format, interpolate short null gaps, enforce units, drop impossible values, emit a data-quality summary |
| F4 Anomaly | `anomaly.py` | Flag anomalies with IQR per `(site, metric, hour_of_day)` |
| F4b Generation | `generation.py` | Derive `solar_cf` / `wind_cf` capacity factors |
| F3 Store | `store.py` | Drop-and-load into DuckDB `readings` + `dq_summary` tables |

### Data source and reproducibility

- **First run:** if `data/snapshot/*.parquet` is **absent**, the pipeline calls the
  Open-Meteo Archive API (one request per site, no key) and writes the raw snapshot to
  `data/snapshot/`.
- **Subsequent runs:** if the snapshot **already exists**, the API fetch is skipped and the
  committed snapshot is reused — so every run is deterministic and reproducible offline.
- The fetch window is **fixed in config** (`config/sites.yaml`), not rolling:

  ```yaml
  window:
    start: "2026-03-01"
    end:   "2026-03-31"
  ```

- To force a fresh fetch, delete the snapshot first:

  ```bash
  rm data/snapshot/*.parquet
  uv run python -m energy_monitor
  ```

### Expected output

```
Snapshot already exists in data/snapshot, skipping fetch.
Cleaning and normalising...
  burgos_ph: ... in, ... filled, ... dropped, ... out
  ...
Flagging anomalies...
  N anomalies flagged across M readings.
Deriving generation potential (solar_cf, wind_cf)...
Loading ... rows into data/warehouse.duckdb...
  readings table: ... rows loaded
  dq_summary table: 3 site(s) recorded.
```

After this step `data/warehouse.duckdb` exists and is populated. (It is gitignored —
regenerate it by re-running the pipeline.)

---

## 4. Launch the dashboard

```bash
uv run python -m energy_monitor --dashboard
```

Open **http://127.0.0.1:8050** in a browser.

> Run Step 3 at least once before launching, so `data/warehouse.duckdb` exists.

The dashboard provides:

- Site multi-select and metric select
- Date-range picker
- Time-series chart with anomalies marked as distinct points
- A data-quality summary line
- Generation-potential view (`solar_cf`, `wind_cf`)
- A chat panel (enabled only when an API key is set — see Step 6)

---

## 5. Configuration

Edit [`config/sites.yaml`](config/sites.yaml) to change the date window or swap sites.
Each site needs `key`, `label`, `lat`, `lon`. Re-run the pipeline (Step 3) after editing;
delete the snapshot first if you change anything that affects the fetched data.

---

## 6. Optional: enable the chat agent

The chat panel is enabled only when `ANTHROPIC_API_KEY` is present. Without it, the rest
of the dashboard works unchanged and the panel shows a disabled notice.

```bash
cp .env.example .env
# edit .env and set:
# ANTHROPIC_API_KEY=sk-ant-...
```

`main()` calls `load_dotenv()` at startup, so the key loads automatically — no manual
`export`. Restart the dashboard; the startup line reads `(agent enabled)`. The `.env` file
is gitignored — never commit it.

The agent uses tool-calling over a fixed set of parameterised SQL queries
(`avg_metric_by_site`, `anomalies_in_window`, `compare_generation` in `queries.py`), so
answers are grounded in stored data — no text-to-SQL, no hallucinated numbers.

---

## 7. Quality gates

```bash
uv run ruff check .     # lint
uv run pyright          # type-check
uv run pytest           # tests
```

---

## 8. Why IQR over z-score

The dashboard flags anomalies with the **IQR** method, computed per site, per metric, per
hour-of-day. We chose IQR over z-score for two reasons:

1. **The data is not normal.** Solar and wind are skewed and heavy-tailed. The z-score
   depends on the mean and standard deviation, both of which are inflated by the very
   outliers we want to detect — which masks real anomalies. IQR rests on the median and
   quartiles, which are robust to extreme values.
2. **We condition on hour-of-day.** Solar has a strong daily cycle (zero at night, peaking
   near noon), so a single global threshold would flag every night and every midday.
   Comparing each reading against the normal range *for that hour at that site* is what an
   analyst actually means by an anomaly.

Two safeguards make this hold up on real data:

- **Daylight filter:** solar is detected on daylight readings only, so all-zero night
  buckets don't collapse the fences to `[0, 0]`.
- **Variance floor:** `IQR_eff = max(IQR, floor)`, so a near-constant bucket doesn't flag
  every tiny deviation.

A reading is flagged when it falls outside `[Q1 - 1.5*IQR_eff, Q3 + 1.5*IQR_eff]` for its
bucket. The `1.5` multiplier and the floor are configurable in `IQRConfig` (`anomaly.py`).
With ~30 samples per bucket over 30 days, flags are **indicative rather than precise** —
another reason the rule is kept deliberately simple.

---

## 9. Project layout

```
energy-anomaly-monitor/
├── config/sites.yaml          # sites + fixed date window
├── src/energy_monitor/
│   ├── __main__.py            # CLI entry: ETL pipeline, or --dashboard
│   ├── config.py              # settings, sites, window
│   ├── ingest.py              # Open-Meteo client + snapshot
│   ├── clean.py               # normalise + data-quality summary
│   ├── anomaly.py             # IQR detection
│   ├── generation.py          # solar_cf / wind_cf capacity factors
│   ├── store.py               # DuckDB load, dq_summary, Parquet export
│   ├── queries.py             # read-only query class (dashboard + agent)
│   ├── dashboard.py           # Dash app
│   └── agent.py               # grounded conversational agent
├── data/
│   ├── snapshot/              # committed raw snapshot (reproducible)
│   └── warehouse.duckdb       # gitignored — regenerate via the pipeline
└── tests/
```

---

## 10. Command reference

| Goal | Command |
|------|---------|
| Install | `uv sync` |
| Generate data (ETL) | `uv run python -m energy_monitor` |
| Force fresh fetch | `rm data/snapshot/*.parquet && uv run python -m energy_monitor` |
| Run dashboard | `uv run python -m energy_monitor --dashboard` |
| Lint / type / test | `uv run ruff check .` · `uv run pyright` · `uv run pytest` |
