from pathlib import Path

import pandas as pd

from energy_monitor.clean import Cleaner
from energy_monitor.config import load_config
from energy_monitor.ingest import OpenMeteoClient
from energy_monitor.store import ReadingsStore

_SNAPSHOT_DIR = Path("data/snapshot")
_DB_PATH = Path("data/warehouse.duckdb")


def _load_snapshots(snapshot_dir: Path) -> dict[str, pd.DataFrame]:
    """Read committed Parquet snapshots from disk keyed by site_key."""
    return {p.stem: pd.read_parquet(p) for p in sorted(snapshot_dir.glob("*.parquet"))}


def main() -> None:
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

    # F3: store
    print(f"Loading {len(clean_result.data)} rows into {_DB_PATH}...")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = ReadingsStore(_DB_PATH)
    store.load(clean_result.data)
    print(f"  readings table: {store.row_count()} rows loaded.")


if __name__ == "__main__":
    main()
