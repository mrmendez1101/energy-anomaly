import argparse
from pathlib import Path

import pandas as pd

from energy_monitor.anomaly import AnomalyDetector, IQRConfig
from energy_monitor.clean import Cleaner
from energy_monitor.config import load_config
from energy_monitor.generation import GenerationPotential
from energy_monitor.ingest import OpenMeteoClient
from energy_monitor.store import ReadingsStore

_SNAPSHOT_DIR = Path("data/snapshot")
_DB_PATH = Path("data/warehouse.duckdb")


def _load_snapshots(snapshot_dir: Path) -> dict[str, pd.DataFrame]:
    """Read committed Parquet snapshots from disk keyed by site_key."""
    return {p.stem: pd.read_parquet(p) for p in sorted(snapshot_dir.glob("*.parquet"))}


def run_pipeline() -> None:
    config = load_config(Path("config/sites.yaml"))

    # F1: fetch and save snapshot (skipped if snapshots already exist)
    if not any(_SNAPSHOT_DIR.glob("*.parquet")):
        print("Fetching from Open-Meteo API...")
        client = OpenMeteoClient()
        frames = client.fetch_all(config)
        client.save_snapshot(frames, _SNAPSHOT_DIR)
    else:
        print(f"Snapshot already exists in {_SNAPSHOT_DIR}, skipping fetch.")

    # F2: clean
    print("Cleaning and normalising...")
    raw_frames = _load_snapshots(_SNAPSHOT_DIR)
    cleaner = Cleaner()
    clean_result = cleaner.clean_all(raw_frames)
    for s in clean_result.summaries:
        print(f"  {s.site_key}: {s.rows_in} in, {s.nulls_filled} filled, "
              f"{s.rows_dropped} dropped, {s.rows_out} out")

    # F4: flag anomalies
    print("Flagging anomalies...")
    detector = AnomalyDetector(IQRConfig())
    flagged = detector.flag(clean_result.data)
    n_anomalies = int(flagged["is_anomaly"].sum())  # type: ignore[arg-type]
    print(f"  {n_anomalies} anomalies flagged across {len(flagged)} readings.")

    # F4b: derive generation-potential capacity factors (comparison metrics)
    print("Deriving generation potential (solar_cf, wind_cf)...")
    enriched = GenerationPotential().add_generation_metrics(flagged)
    n_derived = len(enriched) - len(flagged)
    print(f"  {n_derived} capacity-factor rows added ({len(enriched)} total).")

    # F3: store
    print(f"Loading {len(enriched)} rows into {_DB_PATH}...")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = ReadingsStore(_DB_PATH)
    store.load(enriched)
    store.load_dq(clean_result.summaries)
    print(f"  readings table: {store.row_count()} rows loaded "
          f"(incl. {n_derived} capacity-factor rows).")
    print(f"  dq_summary table: {len(clean_result.summaries)} site(s) recorded.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="energy_monitor")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Dash dashboard instead of running the ETL pipeline.",
    )
    args = parser.parse_args()

    if args.dashboard:
        from energy_monitor.dashboard import run_dashboard
        run_dashboard(_DB_PATH)
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
