from pathlib import Path

from energy_monitor.config import load_config
from energy_monitor.ingest import OpenMeteoClient


def main() -> None:
    config = load_config(Path("config/sites.yaml"))
    client = OpenMeteoClient()
    frames = client.fetch_all(config)
    client.save_snapshot(frames, Path("data/snapshot"))


if __name__ == "__main__":
    main()
